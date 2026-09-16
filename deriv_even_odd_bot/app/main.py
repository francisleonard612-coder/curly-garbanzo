"""
Main orchestrator (spec Sections 33, 34, 36, 62, 64, 65, 66).

Event-driven and fully asynchronous. Trading logic never runs inside a
WebSocket callback (Section 34): the client pushes ticks onto a queue and
per-symbol workers consume them, so a slow model refit cannot stall the
socket reader and cause the keepalive timeout that killed a sibling bot.

STARTUP SEQUENCE, in mandated order (Sections 1, 48):
  1. connect + authorize
  2. discover active symbols, resolve pip_size per symbol
  3. verify DIGITEVEN and DIGITODD are actually offered -- refuse otherwise
  4. verify balance
  5. load historical ticks and seed models (cold start)
  6. subscribe and begin observing

THE EXPECTED STEADY STATE IS "WAIT". On a CSPRNG-driven instrument the
randomness battery will not find an exploitable departure and the bot will
never trade. The dashboard is written to make that legible rather than
looking like a hang.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sys
import time

from app.api.deriv_client import BuyAmbiguousError, DerivAPIError, DerivClient
from app.config.settings import MODE_LIVE, get_settings
from app.economics.edge import Proposal, ProposalError
from app.execution.engine import SymbolPipeline, new_idempotency_key
from app.monitoring.dashboard import Dashboard
from app.risk.manager import RiskManager
from app.risk.staking import StakingEngine
from app.storage.postgres import open_database

logger = logging.getLogger("astra.evenodd")


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-8s %(name)s %(message)s",
        stream=sys.stderr)


class Bot:
    def __init__(self, settings, *, render: bool = True):
        self.settings = settings
        # DB_BACKEND=postgres routes to Supabase; sqlite is the default and
        # is fine locally. On Railway the container filesystem is ephemeral,
        # so sqlite there means the decision history dies on every redeploy.
        self.db = open_database(sqlite_path=settings.db_path)
        d = settings.deriv
        self.client = DerivClient(
            app_id=d.app_id, api_token=d.api_token, ws_url=d.ws_url,
            request_timeout=d.request_timeout,
            api_base_url=d.api_base_url, auth_mode=d.auth_mode,
            account_id=d.account_id or None,
            # The client resolves demo vs real itself under OTP auth. It is
            # the SAME flag the two-switch live guard reads, so the account
            # the bot authenticates against can never disagree with the mode
            # it believes it is running in.
            use_real_account=d.use_real_account,
            max_requests_per_minute=d.max_requests_per_minute)
        r = settings.risk
        self.risk = RiskManager(
            base_stake=r["base_stake"], max_stake=r["max_stake"],
            max_daily_loss=r["max_daily_loss"], max_drawdown=r["max_drawdown"],
            max_consecutive_losses=r["max_consecutive_losses"],
            max_trades_per_day=r["max_trades_per_day"],
            max_concurrent_trades=r["max_concurrent_trades"],
            cooldown_seconds=r["cooldown_seconds"])
        s = settings.staking
        self.staking = StakingEngine(
            method=s["method"], base_stake=r["base_stake"], max_stake=r["max_stake"],
            kelly_fraction=s["kelly_fraction"],
            martingale_enabled=s["martingale_enabled"],
            martingale_factor=s["martingale_factor"],
            martingale_max_steps=s["martingale_max_steps"],
            min_consecutive_losses=s["martingale_trigger_losses"])
        self.dashboard = Dashboard(settings.mode)
        self.render = render
        self.pipelines: dict[str, SymbolPipeline] = {}
        self.last_decision = {}
        self._stop = asyncio.Event()

    async def start(self) -> None:
        await self.client.connect()
        self.db.log_event("INFO", "lifecycle", f"connected, mode={self.settings.mode}")

        await self.client.active_symbols()

        balance = await self.client.balance()
        self.risk.update_balance(balance)
        logger.info("balance verified: %.2f %s", balance, self.settings.currency)

        usable = []
        for symbol in self.settings.symbols:
            ok, why = await self.client.verify_even_odd_available(symbol, self.settings.currency)
            logger.info("contract check %s", why)
            if not ok:
                self.db.log_event("WARNING", "discovery", why)
                continue
            usable.append(symbol)
        if not usable:
            raise RuntimeError("no configured symbol offers DIGITEVEN/DIGITODD -- refusing to run")

        for symbol in usable:
            pip = await self.client.get_pip_size(symbol)
            pipe = SymbolPipeline(symbol, pip, self.settings, db=self.db)
            # Marks this gate as live so app/backtest/simulator.py refuses to
            # replay through it. A simulated run sharing a live TradeGate would
            # pour synthetic rejections into the live DeadlockMonitor, and the
            # operator would read a deadlock diagnosis describing a backtest.
            pipe.gate._live = True
            history = await self.client.tick_history(symbol, count=5000)
            seeded = pipe.seed(history)
            logger.info("%s seeded with %d historical ticks (pip_size=%s)", symbol, seeded, pip)
            self.pipelines[symbol] = pipe

        tasks = []
        for symbol in usable:
            queue = await self.client.subscribe_ticks(symbol)
            tasks.append(asyncio.create_task(self._worker(symbol, queue), name=f"worker-{symbol}"))
        tasks.append(asyncio.create_task(self._renderer(), name="renderer"))
        tasks.append(asyncio.create_task(self._maintenance(), name="maintenance"))

        await self._stop.wait()
        for t in tasks:
            t.cancel()
        await self.client.close()
        self.db.log_event("INFO", "lifecycle", "shutdown")
        self.db.close()

    def stop(self) -> None:
        self._stop.set()

    async def _worker(self, symbol: str, queue: asyncio.Queue) -> None:
        pipe = self.pipelines[symbol]
        while not self._stop.is_set():
            try:
                tick = await asyncio.wait_for(queue.get(), timeout=60.0)
            except asyncio.TimeoutError:
                logger.warning("%s: no tick for 60s", symbol)
                continue
            except asyncio.CancelledError:
                return
            try:
                await self._handle_tick(pipe, tick)
            except Exception as exc:
                logger.exception("%s: tick handling failed", symbol)
                self.db.log_event("ERROR", "tick", str(exc), {"symbol": symbol})

    async def _handle_tick(self, pipe: SymbolPipeline, tick) -> None:
        # STEP 1-3: predict and gate BEFORE the tick is observed (Section 22).
        decision, quote_request = pipe.evaluate(
            tick.quote_raw, tick.pip_size, tick.epoch,
            risk_manager=self.risk, staking=self.staking,
            api_connected=self.client.is_connected,
            db_healthy=self.db.is_healthy() if hasattr(self.db, "is_healthy") else True,
            open_contract=pipe.symbol in getattr(self, "open_contracts", set()))
        pipe.gate.deadlock.note_tick()

        if quote_request is not None:
            side, is_probe = quote_request
            try:
                raw = await self.client.proposal(
                    symbol=pipe.symbol, contract_type=side,
                    amount=self.staking.base_stake, currency=self.settings.currency,
                    duration=1, duration_unit="t")
                proposal = Proposal(
                    contract_type=side, symbol=pipe.symbol,
                    stake=float(raw.get("ask_price", self.staking.base_stake)),
                    payout=float(raw.get("payout", 0.0)),
                    ask_price=float(raw.get("ask_price", 0.0)),
                    currency=self.settings.currency,
                    proposal_id=raw.get("id", ""), spot=raw.get("spot"),
                    received_at=time.time())
                decision = pipe.finalize_with_quote(
                    decision, proposal, risk_manager=self.risk,
                    staking=self.staking, side=side, is_probe=is_probe)
            except (DerivAPIError, ProposalError, ValueError) as exc:
                decision.reason_code = "NO_TRADE_BAD_PROPOSAL"
                decision.explanation = f"proposal unusable: {exc}"
                decision.hard_gate_failed = True
                pipe.gate.deadlock.note_rejection(
                    reason_code="NO_TRADE_BAD_PROPOSAL", hard=True)

        decision_id = self.db.record_decision(decision)
        self.last_decision[pipe.symbol] = decision

        if decision.will_trade and self.settings.mode == MODE_LIVE:
            await self._execute(pipe, decision, decision_id)

        # STEP 4-5: only now does the model see the tick.
        pipe.learn(tick.quote_raw, tick.pip_size, tick.epoch)

    async def _execute(self, pipe: SymbolPipeline, decision, decision_id: int) -> None:
        key = new_idempotency_key(pipe.symbol, decision.contract_type)
        if self.db.has_idempotency_key(key):
            logger.error("duplicate idempotency key -- refusing buy")
            return
        try:
            raw = await self.client.proposal(
                symbol=pipe.symbol, contract_type=decision.contract_type,
                amount=decision.stake, currency=self.settings.currency,
                duration=1, duration_unit="t")
            buy = await self.client.buy(raw.get("id", ""), float(raw.get("ask_price", 0.0)),
                                        idempotency_key=key)
        except BuyAmbiguousError as exc:
            # Never retry (Section 35) -- reconcile instead.
            logger.error("ambiguous buy: %s", exc)
            self.db.log_event("CRITICAL", "buy", str(exc))
            self.risk.emergency_stop("ambiguous buy outcome -- manual reconciliation required")
            return
        except DerivAPIError as exc:
            self.db.log_event("ERROR", "buy", str(exc))
            return

        contract_id = buy.get("contract_id")
        if not contract_id:
            return
        self.risk.register_open(decision.stake)
        self.db.record_trade_open(
            decision_id=decision_id, symbol=pipe.symbol, contract_id=contract_id,
            idempotency_key=key, contract_type=decision.contract_type,
            stake=decision.stake, payout=decision.payout,
            buy_price=float(buy.get("buy_price", decision.stake)),
            entry_digit=decision.current_digit)
        asyncio.create_task(self._monitor(contract_id, decision))

    async def _monitor(self, contract_id: int, decision) -> None:
        poc = await self.client.wait_for_settlement(contract_id)
        if not poc:
            # Outcome is genuinely unknown -- the subscription went quiet
            # rather than confirming is_sold. Recorded as a loss for pnl
            # accounting because that is the conservative assumption, and
            # register_result MUST agree: leaving the martingale/risk state
            # un-updated here would desync it from what the trades table
            # says happened, which is exactly the kind of mismatch that
            # makes an escalating stake untrustworthy -- the very thing
            # martingale needs most is an accurate loss-streak count.
            self.risk.register_result(0.0)
            self.staking.register_result(False)
            self.db.record_trade_result(contract_id, won=False, pnl=0.0,
                                        error="settlement timeout")
            return
        profit = float(poc.get("profit", 0.0))
        won = profit > 0
        self.risk.register_result(profit)
        self.staking.register_result(won)
        self.db.record_trade_result(contract_id, won=won, pnl=profit)
        for cal in self.pipelines[decision.symbol].calibrators.values():
            cal.note_settled()

    async def _maintenance(self) -> None:
        """Periodic work kept OFF the tick path (Section 62)."""
        while not self._stop.is_set():
            await asyncio.sleep(30)
            for symbol, pipe in self.pipelines.items():
                try:
                    pipe.maybe_refit()
                    self.db.record_model_performance(symbol, pipe.ensemble.health_report())
                except Exception:
                    logger.exception("maintenance failed for %s", symbol)
            try:
                self.risk.update_balance(await self.client.balance())
            except DerivAPIError:
                logger.warning("balance refresh failed")

    async def _renderer(self) -> None:
        while not self._stop.is_set():
            await asyncio.sleep(1.0)
            if not self.render or not self.pipelines:
                continue
            symbol = next(iter(self.pipelines))
            print("\033[2J\033[H" + self.dashboard.render(
                pipeline=self.pipelines[symbol], client=self.client, risk=self.risk,
                last_decision=self.last_decision.get(symbol),
                db_summary=self.db.trade_summary()), flush=True)


async def run(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Deriv Even/Odd quantitative engine")
    parser.add_argument("--config", default=None)
    parser.add_argument("--no-dashboard", action="store_true")
    args = parser.parse_args(argv)

    settings = get_settings(args.config)
    setup_logging(settings.log_level)
    logger.info("mode=%s symbols=%s", settings.mode, settings.symbols)
    if settings.mode == MODE_LIVE:
        logger.warning("LIVE MODE: real money is at risk")

    bot = Bot(settings, render=not args.no_dashboard)
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, bot.stop)
        except NotImplementedError:
            pass
    try:
        await bot.start()
    except Exception:
        logger.exception("fatal error")
        return 1
    return 0


def main() -> int:
    try:
        return asyncio.run(run())
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
