"""
Risk management (spec Sections 31, 59).

INDEPENDENT OF PREDICTION BY CONSTRUCTION. This module never sees a
probability, a model, or an edge. It answers exactly one question -- "is
the account currently allowed to place a trade of this size?" -- from
account state alone. That separation is the point: a prediction engine that
could talk the risk engine into an exception is not a risk engine.

FAIL CLOSED (Section 59). Every unknown resolves to "no trade". If balance
cannot be verified, if the tick stream is stale, if the database refuses a
write, the answer is no. A risk check that errors open is worse than none,
because it creates false confidence.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

BLOCK_DAILY_LOSS = "NO_TRADE_RISK_DAILY_LOSS"
BLOCK_DRAWDOWN = "NO_TRADE_RISK_DRAWDOWN"
BLOCK_CONSECUTIVE = "NO_TRADE_RISK_CONSECUTIVE_LOSSES"
BLOCK_TRADE_COUNT = "NO_TRADE_RISK_TRADE_COUNT"
BLOCK_CONCURRENT = "NO_TRADE_RISK_CONCURRENT"
BLOCK_STAKE = "NO_TRADE_RISK_STAKE_LIMIT"
BLOCK_COOLDOWN = "NO_TRADE_COOLDOWN"
BLOCK_BALANCE = "NO_TRADE_RISK_BALANCE_UNVERIFIED"
BLOCK_STOPPED = "NO_TRADE_EMERGENCY_STOP"


@dataclass
class RiskDecision:
    allowed: bool
    reason: str
    stake: float = 0.0


@dataclass
class RiskManager:
    base_stake: float = 1.0
    max_stake: float = 5.0
    max_daily_loss: float = 25.0
    max_drawdown: float = 50.0
    max_consecutive_losses: int = 8
    max_trades_per_day: int = 200
    max_concurrent_trades: int = 1
    cooldown_seconds: float = 0.0

    balance: float | None = None
    peak_balance: float | None = None
    session_pnl: float = 0.0
    daily_pnl: float = 0.0
    consecutive_losses: int = 0
    trades_today: int = 0
    open_trades: int = 0
    emergency_stopped: bool = False
    stop_reason: str = ""
    _day: str = field(default_factory=lambda: time.strftime("%Y-%m-%d"))
    _last_trade_at: float = 0.0

    def _roll_day(self) -> None:
        today = time.strftime("%Y-%m-%d")
        if today != self._day:
            self._day = today
            self.daily_pnl = 0.0
            self.trades_today = 0

    def update_balance(self, balance: float) -> None:
        self.balance = float(balance)
        self.peak_balance = self.balance if self.peak_balance is None \
            else max(self.peak_balance, self.balance)

    @property
    def drawdown(self) -> float:
        if self.balance is None or self.peak_balance is None:
            return 0.0
        return max(0.0, self.peak_balance - self.balance)

    def emergency_stop(self, reason: str) -> None:
        """Section 59. One-way latch -- only an operator restart clears it.
        Anything that could auto-clear would let a transient condition
        reopen trading during the very incident it fired for."""
        self.emergency_stopped = True
        self.stop_reason = reason

    def can_trade(self, stake: float, *, balance_verified: bool = True) -> RiskDecision:
        self._roll_day()
        if self.emergency_stopped:
            return RiskDecision(False, f"{BLOCK_STOPPED}: {self.stop_reason}")
        if not balance_verified or self.balance is None:
            return RiskDecision(False, BLOCK_BALANCE)
        if stake <= 0:
            return RiskDecision(False, f"{BLOCK_STAKE}: non-positive stake")
        if stake > self.max_stake:
            return RiskDecision(False, f"{BLOCK_STAKE}: {stake:.2f} > max {self.max_stake:.2f}")
        if stake > self.balance:
            return RiskDecision(False, f"{BLOCK_STAKE}: stake exceeds balance")
        if self.daily_pnl <= -abs(self.max_daily_loss):
            return RiskDecision(False, f"{BLOCK_DAILY_LOSS}: {self.daily_pnl:.2f}")
        if self.drawdown >= abs(self.max_drawdown):
            return RiskDecision(False, f"{BLOCK_DRAWDOWN}: {self.drawdown:.2f}")
        if self.consecutive_losses >= self.max_consecutive_losses:
            return RiskDecision(False, f"{BLOCK_CONSECUTIVE}: {self.consecutive_losses}")
        if self.trades_today >= self.max_trades_per_day:
            return RiskDecision(False, f"{BLOCK_TRADE_COUNT}: {self.trades_today}")
        if self.open_trades >= self.max_concurrent_trades:
            return RiskDecision(False, f"{BLOCK_CONCURRENT}: {self.open_trades}")
        if self.cooldown_seconds > 0 and (time.time() - self._last_trade_at) < self.cooldown_seconds:
            remaining = self.cooldown_seconds - (time.time() - self._last_trade_at)
            return RiskDecision(False, f"{BLOCK_COOLDOWN}: {remaining:.1f}s remaining")
        return RiskDecision(True, "risk checks passed", stake)

    def register_open(self, stake: float) -> None:
        self.open_trades += 1
        self.trades_today += 1
        self._last_trade_at = time.time()

    def register_result(self, pnl: float) -> None:
        self.open_trades = max(0, self.open_trades - 1)
        self.session_pnl += pnl
        self.daily_pnl += pnl
        if self.balance is not None:
            self.update_balance(self.balance + pnl)
        if pnl < 0:
            self.consecutive_losses += 1
        elif pnl > 0:
            self.consecutive_losses = 0
        if self.daily_pnl <= -abs(self.max_daily_loss):
            self.emergency_stop(f"daily loss limit hit ({self.daily_pnl:.2f})")
        elif self.drawdown >= abs(self.max_drawdown):
            self.emergency_stop(f"max drawdown hit ({self.drawdown:.2f})")

    def snapshot(self) -> dict:
        return {
            "balance": self.balance, "peak": self.peak_balance,
            "session_pnl": self.session_pnl, "daily_pnl": self.daily_pnl,
            "drawdown": self.drawdown, "consecutive_losses": self.consecutive_losses,
            "trades_today": self.trades_today, "open": self.open_trades,
            "stopped": self.emergency_stopped, "stop_reason": self.stop_reason,
        }
