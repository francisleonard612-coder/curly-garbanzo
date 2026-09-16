"""
SQLite persistence (spec Sections 23, 37, 61).

Stores every decision -- traded or not -- so any outcome can be
reconstructed forensically. The no-trade rows are the valuable ones on this
instrument: a bot that correctly waits produces a database full of reasons
it waited, and that record is the evidence that the veto is working rather
than the bot being broken.

WRITE FAILURE IS A TRADING HALT (Section 59: "database cannot record a
trade" is on the fail-closed list). record_decision() raises rather than
swallowing, and the executor treats that as an emergency stop. An untracked
open contract is worse than a missed trade.

Feature vectors are stored as JSON only for TRADED decisions. Section 37
warns against storing massive redundant feature arrays; at one no-trade
decision per tick, persisting ~40 floats each would add gigabytes per week
to record that nothing happened.
"""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

SCHEMA_VERSION = 2

SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_version (version INTEGER PRIMARY KEY);

CREATE TABLE IF NOT EXISTS decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    symbol TEXT NOT NULL,
    decision TEXT NOT NULL,
    reason_code TEXT NOT NULL,
    explanation TEXT,
    quote REAL, digit INTEGER,
    p_even_digit_derived REAL, p_even_direct REAL, p_even_ensemble REAL,
    calibrated_p_even REAL, probability_lower REAL, probability_upper REAL,
    dispersion REAL, agreement_fraction REAL,
    regime TEXT, entropy REAL, calibration_quality REAL, randomness_tradeable INTEGER,
    contract_type TEXT, stake REAL, payout REAL, break_even REAL,
    edge REAL, conservative_edge REAL, expected_value REAL, quality_score REAL,
    is_probe INTEGER DEFAULT 0,
    -- v2: the Section 13-19 continuous layer. These are what make the
    -- deadlock and shadow reports reconstructable from the database alone.
    opportunity_score REAL, opportunity_zone TEXT, opportunity_threshold REAL,
    selectivity_multiplier REAL, limiting_factor TEXT, capped_by TEXT,
    randomness_evidence REAL, regime_evidence REAL, signal_persistence REAL,
    contribution_detail TEXT, hard_gate_failed INTEGER DEFAULT 0,
    digit_probabilities TEXT, member_p_even TEXT, model_weights TEXT, gates TEXT
);
CREATE INDEX IF NOT EXISTS idx_decisions_ts ON decisions(ts);
CREATE INDEX IF NOT EXISTS idx_decisions_symbol_ts ON decisions(symbol, ts);
CREATE INDEX IF NOT EXISTS idx_decisions_reason ON decisions(reason_code);
CREATE INDEX IF NOT EXISTS idx_decisions_zone ON decisions(opportunity_zone);
CREATE INDEX IF NOT EXISTS idx_decisions_limiting ON decisions(limiting_factor);

CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    decision_id INTEGER REFERENCES decisions(id),
    ts REAL NOT NULL,
    symbol TEXT NOT NULL,
    contract_id INTEGER UNIQUE,
    idempotency_key TEXT UNIQUE,
    contract_type TEXT, stake REAL, payout REAL, buy_price REAL,
    entry_digit INTEGER, exit_digit INTEGER,
    won INTEGER, pnl REAL, settled_at REAL, error TEXT
);
CREATE INDEX IF NOT EXISTS idx_trades_ts ON trades(ts);

CREATE TABLE IF NOT EXISTS model_performance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL, symbol TEXT, model TEXT,
    n INTEGER, brier REAL, brier_skill REAL, log_loss REAL,
    accuracy REAL, weight REAL, health TEXT
);
CREATE INDEX IF NOT EXISTS idx_perf_ts ON model_performance(ts);

CREATE TABLE IF NOT EXISTS randomness_checks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL, symbol TEXT, n_samples INTEGER,
    chi2_p REAL, runs_p REAL, even_rate REAL, ci_lower REAL, ci_upper REAL,
    any_significant INTEGER, economically_relevant INTEGER, tradeable INTEGER,
    summary TEXT
);

CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL, level TEXT, category TEXT, message TEXT, detail TEXT
);
CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts);
"""


class Database:
    def __init__(self, path: str = "data/bot.db"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=FULL")   # a trade record must survive a crash
        self._migrate()

    #: v1 -> v2 columns, added in place so an existing database keeps its
    #: history. ALTER TABLE ADD COLUMN is the only schema change SQLite does
    #: cheaply; anything else would mean rewriting a multi-gigabyte decisions
    #: table on startup.
    _V2_COLUMNS = (
        ("opportunity_score", "REAL"), ("opportunity_zone", "TEXT"),
        ("opportunity_threshold", "REAL"), ("selectivity_multiplier", "REAL"),
        ("limiting_factor", "TEXT"), ("capped_by", "TEXT"),
        ("randomness_evidence", "REAL"), ("regime_evidence", "REAL"),
        ("signal_persistence", "REAL"), ("contribution_detail", "TEXT"),
        ("hard_gate_failed", "INTEGER DEFAULT 0"),
    )

    def _migrate(self) -> None:
        self.conn.executescript(SCHEMA)
        existing = {r["name"] for r in
                    self.conn.execute("PRAGMA table_info(decisions)")}
        for col, decl in self._V2_COLUMNS:
            if col not in existing:
                self.conn.execute(f"ALTER TABLE decisions ADD COLUMN {col} {decl}")
        cur = self.conn.execute("SELECT version FROM schema_version LIMIT 1")
        row = cur.fetchone()
        if row is None:
            self.conn.execute("INSERT INTO schema_version(version) VALUES (?)",
                              (SCHEMA_VERSION,))
        else:
            self.conn.execute("UPDATE schema_version SET version=?", (SCHEMA_VERSION,))
        self.conn.commit()

    def is_healthy(self) -> bool:
        """Consulted by the hard gates: a trade we cannot record is a trade we
        cannot reconcile."""
        try:
            self.conn.execute("SELECT 1").fetchone()
            return True
        except Exception:
            return False

    def record_decision(self, d) -> int:
        """Raises on failure -- see module docstring."""
        cur = self.conn.execute(
            """INSERT INTO decisions (
                ts,symbol,decision,reason_code,explanation,quote,digit,
                p_even_digit_derived,p_even_direct,p_even_ensemble,calibrated_p_even,
                probability_lower,probability_upper,dispersion,agreement_fraction,
                regime,entropy,calibration_quality,randomness_tradeable,
                contract_type,stake,payout,break_even,edge,conservative_edge,
                expected_value,quality_score,is_probe,
                opportunity_score,opportunity_zone,opportunity_threshold,
                selectivity_multiplier,limiting_factor,capped_by,
                randomness_evidence,regime_evidence,signal_persistence,
                contribution_detail,hard_gate_failed,
                digit_probabilities,member_p_even,model_weights,gates
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,
                      ?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (d.timestamp, d.symbol, d.decision, d.reason_code, d.explanation,
             d.current_quote, d.current_digit,
             d.p_even_digit_derived, d.p_even_direct, d.p_even_ensemble, d.calibrated_p_even,
             d.probability_lower, d.probability_upper, d.dispersion, d.agreement_fraction,
             d.regime, d.entropy, d.calibration_quality,
             None if d.randomness_tradeable is None else int(d.randomness_tradeable),
             d.contract_type, d.stake, d.payout, d.break_even_probability,
             d.edge, d.conservative_edge, d.expected_value, d.quality_score, int(d.is_probe),
             d.opportunity_score, d.opportunity_zone, d.opportunity_threshold,
             d.selectivity_multiplier, d.limiting_factor,
             json.dumps(d.capped_by) if d.capped_by else None,
             d.randomness_evidence, d.regime_evidence, d.signal_persistence,
             json.dumps(d.contribution_detail) if d.contribution_detail else None,
             int(d.hard_gate_failed),
             json.dumps(d.digit_probabilities) if d.will_trade else None,
             json.dumps(d.member_p_even) if d.will_trade else None,
             json.dumps(d.model_weights) if d.will_trade else None,
             json.dumps([(g.passed, g.code, g.explanation) for g in d.gates])))
        self.conn.commit()
        return cur.lastrowid

    def record_trade_open(self, *, decision_id, symbol, contract_id, idempotency_key,
                          contract_type, stake, payout, buy_price, entry_digit) -> int:
        cur = self.conn.execute(
            """INSERT INTO trades (decision_id,ts,symbol,contract_id,idempotency_key,
               contract_type,stake,payout,buy_price,entry_digit)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (decision_id, time.time(), symbol, contract_id, idempotency_key,
             contract_type, stake, payout, buy_price, entry_digit))
        self.conn.commit()
        return cur.lastrowid

    def record_trade_result(self, contract_id: int, *, won: bool, pnl: float,
                            exit_digit: int | None = None, error: str | None = None) -> None:
        self.conn.execute(
            """UPDATE trades SET won=?, pnl=?, exit_digit=?, settled_at=?, error=?
               WHERE contract_id=?""",
            (int(won), pnl, exit_digit, time.time(), error, contract_id))
        self.conn.commit()

    def has_idempotency_key(self, key: str) -> bool:
        """Duplicate-buy protection (Section 35)."""
        cur = self.conn.execute("SELECT 1 FROM trades WHERE idempotency_key=? LIMIT 1", (key,))
        return cur.fetchone() is not None

    def record_model_performance(self, symbol: str, report: dict) -> None:
        now = time.time()
        rows = [(now, symbol, name, r["n"], r["brier"], r["brier_skill"],
                 r["log_loss"], r["accuracy"], r["weight"], r["health"])
                for name, r in report.items()]
        self.conn.executemany(
            """INSERT INTO model_performance (ts,symbol,model,n,brier,brier_skill,
               log_loss,accuracy,weight,health) VALUES (?,?,?,?,?,?,?,?,?,?)""", rows)
        self.conn.commit()

    def record_randomness(self, symbol: str, verdict) -> None:
        pi = verdict.parity_interval
        self.conn.execute(
            """INSERT INTO randomness_checks (ts,symbol,n_samples,chi2_p,runs_p,
               even_rate,ci_lower,ci_upper,any_significant,economically_relevant,
               tradeable,summary) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (time.time(), symbol, verdict.n_samples,
             verdict.digit_uniformity.p_value if verdict.digit_uniformity else None,
             verdict.runs.p_value if verdict.runs else None,
             pi.point if pi else None, pi.lower if pi else None, pi.upper if pi else None,
             int(verdict.any_significant), int(verdict.economically_relevant),
             int(verdict.tradeable), verdict.summary()))
        self.conn.commit()

    def log_event(self, level: str, category: str, message: str, detail: dict | None = None) -> None:
        self.conn.execute(
            "INSERT INTO events (ts,level,category,message,detail) VALUES (?,?,?,?,?)",
            (time.time(), level, category, message, json.dumps(detail) if detail else None))
        self.conn.commit()

    def reason_histogram(self, symbol: str | None = None, since: float | None = None) -> dict:
        q = "SELECT reason_code, COUNT(*) c FROM decisions WHERE 1=1"
        args: list = []
        if symbol:
            q += " AND symbol=?"
            args.append(symbol)
        if since:
            q += " AND ts>=?"
            args.append(since)
        q += " GROUP BY reason_code ORDER BY c DESC"
        return {r["reason_code"]: r["c"] for r in self.conn.execute(q, args)}

    def trade_summary(self) -> dict:
        r = self.conn.execute(
            """SELECT COUNT(*) n, SUM(COALESCE(won,0)) wins, SUM(COALESCE(pnl,0)) pnl
               FROM trades WHERE settled_at IS NOT NULL""").fetchone()
        n = r["n"] or 0
        return {"trades": n, "wins": r["wins"] or 0, "pnl": r["pnl"] or 0.0,
                "win_rate": (r["wins"] / n) if n else float("nan")}

    def close(self) -> None:
        self.conn.close()
