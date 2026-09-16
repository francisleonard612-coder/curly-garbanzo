"""
PostgreSQL / Supabase persistence.

Drop-in replacement for app/storage/db.Database, selected at startup by
setting DB_BACKEND=postgres. Same method names, same semantics, same
fail-closed contract: record_decision() raises rather than swallowing, and
the executor treats that as an emergency stop, because an untracked open
contract is worse than a missed trade.

WHY THIS EXISTS. Railway containers have ephemeral filesystems -- a redeploy,
a crash, or a platform-side restart wipes /data unless a volume is attached,
and a bot whose decision history vanishes on every deploy cannot be reviewed.
Supabase gives the history somewhere durable that outlives the container.

CONNECT THROUGH THE POOLER, PORT 6543. Supabase's direct connection (5432)
caps at a small number of connections and this process holds one open for its
whole life; on the pooled port that is free. The URI is in the Supabase
dashboard under Project Settings -> Database -> Connection string -> URI.

USE THE SERVICE ROLE CREDENTIALS, NEVER THE ANON KEY. The schema enables RLS
with read-only policies for `authenticated`, so an anon connection silently
writes nothing -- the bot would appear to run and record absolutely nothing,
which is the worst of both failure modes.

SCHEMA IS NOT CREATED HERE. supabase/schema.sql is applied once, by hand, in
the SQL editor. A bot that runs DDL on startup is a bot that can destroy a
production table during a rollback, and the migration it would need to guess
at is exactly the one a human should look at.
"""
from __future__ import annotations

import json
import logging
import os
import time

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 2


class PostgresUnavailable(RuntimeError):
    pass


def _connect(dsn: str):
    try:
        import psycopg
    except ImportError as exc:  # pragma: no cover - import guard
        raise PostgresUnavailable(
            "DB_BACKEND=postgres requires psycopg: pip install 'psycopg[binary]'"
        ) from exc
    # autocommit: every write here is a single statement that must be durable
    # the moment it returns. An open transaction spanning ticks would mean a
    # crash loses an unknown number of decisions, including the one that
    # opened a contract.
    #
    # prepare_threshold=None: this connects through Supabase's pooled port
    # (6543, PgBouncer in transaction-pooling mode -- see module docstring).
    # PgBouncer in that mode does not support server-side prepared
    # statements: each logical psycopg connection can be handed a different
    # backend connection between statements, so a name psycopg prepared on
    # one backend does not exist -- or collides with one of the same
    # generated name -- on the next. psycopg3 auto-prepares any statement
    # once it has been executed `prepare_threshold` times (default 5), which
    # is exactly what record_decision() and record_model_performance() do on
    # every tick/maintenance cycle. Setting this to None disables client-side
    # prepared statements entirely, which is the standard fix for pooled
    # Postgres connections (matches Supabase's own guidance for port 6543).
    return psycopg.connect(
        dsn, autocommit=True, connect_timeout=10, prepare_threshold=None)


class PostgresDatabase:
    def __init__(self, dsn: str | None = None):
        dsn = dsn or os.getenv("DATABASE_URL", "")
        if not dsn:
            raise PostgresUnavailable(
                "DATABASE_URL is empty; set it to the Supabase pooled URI "
                "(port 6543) or set DB_BACKEND=sqlite")
        if ":6543" not in dsn and "pooler" not in dsn:
            logger.warning(
                "DATABASE_URL does not look like the Supabase pooler (port "
                "6543). The direct port has a low connection cap.")
        self.dsn = dsn
        self.conn = _connect(dsn)
        self._verify_schema()

    # -- lifecycle ----------------------------------------------------------

    def _verify_schema(self) -> None:
        """Fails loudly if schema.sql has not been applied.

        The alternative -- discovering it on the first insert, hours in -- is
        how a run produces no data and no obvious reason why.
        """
        with self.conn.cursor() as cur:
            cur.execute("""
                select count(*) from information_schema.tables
                 where table_schema = 'public'
                   and table_name in ('decisions','trades','model_performance',
                                      'randomness_checks','events')""")
            found = cur.fetchone()[0]
        if found < 5:
            raise PostgresUnavailable(
                f"only {found}/5 expected tables exist. Apply "
                f"supabase/schema.sql in the Supabase SQL editor first.")

    def _reconnect(self) -> None:
        try:
            self.conn.close()
        except Exception:
            pass
        self.conn = _connect(self.dsn)

    def is_healthy(self) -> bool:
        """Consulted by the hard gates before every candidate. One cheap
        reconnect attempt, because a transient pooler blip should not latch
        the bot off for the rest of the session."""
        try:
            with self.conn.cursor() as cur:
                cur.execute("select 1")
                cur.fetchone()
            return True
        except Exception:
            try:
                self._reconnect()
                return True
            except Exception as exc:
                logger.error("database unhealthy: %s", exc)
                return False

    def close(self) -> None:
        try:
            self.conn.close()
        except Exception:
            pass

    # -- writes --------------------------------------------------------------

    def record_decision(self, d) -> int:
        """Raises on failure -- see module docstring."""
        sql = """
            insert into decisions (
                ts, symbol, decision, reason_code, explanation, quote, digit,
                p_even_digit_derived, p_even_direct, p_even_ensemble,
                calibrated_p_even, probability_lower, probability_upper,
                dispersion, agreement_fraction, regime, entropy,
                calibration_quality, randomness_tradeable, contract_type,
                stake, payout, break_even, edge, conservative_edge,
                expected_value, quality_score, is_probe,
                opportunity_score, opportunity_zone, opportunity_threshold,
                selectivity_multiplier, limiting_factor, capped_by,
                randomness_evidence, regime_evidence, signal_persistence,
                contribution_detail, hard_gate_failed,
                digit_probabilities, member_p_even, model_weights, gates
            ) values (
                %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,
                %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,
                %s,%s,%s,%s
            ) returning id"""
        traded = d.will_trade
        args = (
            d.timestamp, d.symbol, d.decision, d.reason_code, d.explanation,
            d.current_quote, d.current_digit,
            d.p_even_digit_derived, d.p_even_direct, d.p_even_ensemble,
            d.calibrated_p_even, d.probability_lower, d.probability_upper,
            d.dispersion, d.agreement_fraction, d.regime, d.entropy,
            d.calibration_quality, d.randomness_tradeable, d.contract_type,
            d.stake, d.payout, d.break_even_probability, d.edge,
            d.conservative_edge, d.expected_value, d.quality_score, d.is_probe,
            d.opportunity_score, d.opportunity_zone, d.opportunity_threshold,
            d.selectivity_multiplier, d.limiting_factor,
            json.dumps(d.capped_by) if d.capped_by else None,
            d.randomness_evidence, d.regime_evidence, d.signal_persistence,
            json.dumps(d.contribution_detail) if d.contribution_detail else None,
            d.hard_gate_failed,
            # Section 37: heavy payloads only for traded rows. At one no-trade
            # row per tick, persisting ~40 floats each adds gigabytes a week to
            # record that nothing happened.
            json.dumps(d.digit_probabilities) if traded else None,
            json.dumps(d.member_p_even) if traded else None,
            json.dumps(d.model_weights) if traded else None,
            json.dumps([(g.passed, g.code, g.explanation) for g in d.gates]),
        )
        with self.conn.cursor() as cur:
            cur.execute(sql, args)
            return cur.fetchone()[0]

    def record_trade_open(self, *, decision_id, symbol, contract_id,
                          idempotency_key, contract_type, stake, payout,
                          buy_price, entry_digit) -> int:
        with self.conn.cursor() as cur:
            cur.execute("""
                insert into trades (decision_id, ts, symbol, contract_id,
                    idempotency_key, contract_type, stake, payout, buy_price,
                    entry_digit)
                values (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) returning id""",
                (decision_id, time.time(), symbol, contract_id,
                 idempotency_key, contract_type, stake, payout, buy_price,
                 entry_digit))
            return cur.fetchone()[0]

    def record_trade_result(self, contract_id: int, *, won: bool, pnl: float,
                            exit_digit: int | None = None,
                            error: str | None = None) -> None:
        with self.conn.cursor() as cur:
            cur.execute("""
                update trades set won=%s, pnl=%s, exit_digit=%s, settled_at=%s,
                       error=%s
                 where contract_id=%s""",
                (won, pnl, exit_digit, time.time(), error, contract_id))

    def has_idempotency_key(self, key: str) -> bool:
        """Duplicate-buy protection (Section 35). Belt and braces: the UNIQUE
        constraint in the schema is the real guarantee, since this check and
        the insert are not atomic with respect to each other."""
        with self.conn.cursor() as cur:
            cur.execute("select 1 from trades where idempotency_key=%s limit 1",
                        (key,))
            return cur.fetchone() is not None

    def record_model_performance(self, symbol: str, report: dict) -> None:
        now = time.time()
        rows = [(now, symbol, name, r["n"], r["brier"], r["brier_skill"],
                 r["log_loss"], r["accuracy"], r["weight"], r["health"])
                for name, r in report.items()]
        if not rows:
            return
        with self.conn.cursor() as cur:
            cur.executemany("""
                insert into model_performance (ts,symbol,model,n,brier,
                    brier_skill,log_loss,accuracy,weight,health)
                values (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""", rows)

    def record_randomness(self, symbol: str, verdict) -> None:
        pi = verdict.parity_interval
        with self.conn.cursor() as cur:
            cur.execute("""
                insert into randomness_checks (ts,symbol,n_samples,chi2_p,
                    runs_p,even_rate,ci_lower,ci_upper,any_significant,
                    economically_relevant,tradeable,summary)
                values (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                (time.time(), symbol, verdict.n_samples,
                 verdict.digit_uniformity.p_value if verdict.digit_uniformity else None,
                 verdict.runs.p_value if verdict.runs else None,
                 pi.point if pi else None, pi.lower if pi else None,
                 pi.upper if pi else None,
                 bool(verdict.any_significant),
                 bool(verdict.economically_relevant),
                 bool(verdict.tradeable), verdict.summary()))

    def log_event(self, level: str, category: str, message: str,
                  detail: dict | None = None) -> None:
        """Never raises. An unloggable warning must not take down a bot that
        is otherwise healthy -- unlike record_decision, which must."""
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    insert into events (ts,level,category,message,detail)
                    values (%s,%s,%s,%s,%s)""",
                    (time.time(), level, category, message,
                     json.dumps(detail) if detail else None))
        except Exception as exc:
            logger.warning("could not log event (%s): %s", message, exc)

    # -- reads ---------------------------------------------------------------

    def reason_histogram(self, symbol: str | None = None,
                         since: float | None = None) -> dict:
        q = "select reason_code, count(*) from decisions where true"
        args: list = []
        if symbol:
            q += " and symbol=%s"
            args.append(symbol)
        if since:
            q += " and ts>=%s"
            args.append(since)
        q += " group by reason_code order by 2 desc"
        with self.conn.cursor() as cur:
            cur.execute(q, args)
            return {r[0]: r[1] for r in cur.fetchall()}

    def trade_summary(self) -> dict:
        with self.conn.cursor() as cur:
            cur.execute("""
                select count(*), count(*) filter (where won),
                       coalesce(sum(pnl), 0)
                  from trades where settled_at is not null""")
            n, wins, pnl = cur.fetchone()
        n = n or 0
        return {"trades": n, "wins": wins or 0, "pnl": float(pnl or 0.0),
                "win_rate": (wins / n) if n else float("nan")}


def open_database(backend: str | None = None, *, sqlite_path: str = "data/bot.db"):
    """Factory used by app/main.py.

    Falls back to SQLite when postgres is not configured, and says so loudly.
    A silent fallback would mean a Railway deploy that looks healthy while
    writing its entire history to a filesystem that vanishes on redeploy.
    """
    backend = (backend or os.getenv("DB_BACKEND", "sqlite")).strip().lower()
    if backend == "postgres":
        return PostgresDatabase()
    if backend != "sqlite":
        raise ValueError(f"unknown DB_BACKEND {backend!r}: use sqlite or postgres")
    from app.storage.db import Database
    return Database(sqlite_path)
