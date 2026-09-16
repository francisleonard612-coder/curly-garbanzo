"""
Configuration (spec Sections 56, 60, 65, 66).

YAML holds the research parameters; environment variables hold credentials
and the handful of deployment switches. No parameter in this file requires
a source edit to change.

TWO SAFETY RULES CARRIED FORWARD FROM PRIOR INCIDENTS IN THIS ACCOUNT:

1. STAKE IS NEVER BALANCE-SCALED AT THE PREDICTION LAYER. A sibling bot had
   a real production incident where a balance-scaled base stake and an
   independent risk allocator disagreed by 527x, caught only by a hard
   ceiling. base_stake here is a fixed currency amount. Percentage- and
   Kelly-based sizing live in app/risk/staking.py and are bounded there,
   independently, by max_stake.

2. REAL-MONEY MODE IS OPT-IN THROUGH TWO INDEPENDENT SWITCHES (Section 65).
   DERIV_USE_REAL=true alone is not enough; TRADING_MODE must also be
   explicitly "live". Research and demo cannot reach a buy call at all. The
   default of every switch is the safe one, so a missing or misspelled env
   var degrades toward not trading rather than toward trading real money.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

load_dotenv()


def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    return default if v is None else v.strip().lower() in ("1", "true", "yes", "on")


def _env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    return float(v) if v not in (None, "") else default


def _env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    return int(v) if v not in (None, "") else default


class ConfigError(RuntimeError):
    pass


# Trading modes, in ascending order of danger.
MODE_RESEARCH = "research"   # never calls buy, not even on demo (Section 64)
MODE_DEMO = "demo"           # real order flow, play money (Section 65)
MODE_LIVE = "live"           # real money (Section 66)
VALID_MODES = (MODE_RESEARCH, MODE_DEMO, MODE_LIVE)


@dataclass
class DerivSettings:
    app_id: str = field(default_factory=lambda: os.getenv("DERIV_APP_ID", "1089"))
    api_token: str = field(default_factory=lambda: os.getenv("DERIV_API_TOKEN", ""))
    ws_url: str = field(default_factory=lambda: os.getenv(
        "DERIV_WS_URL", "wss://ws.derivws.com/websockets/v3"))
    use_real_account: bool = field(default_factory=lambda: _env_bool("DERIV_USE_REAL", False))
    request_timeout: float = field(default_factory=lambda: _env_float("DERIV_REQUEST_TIMEOUT", 15.0))


class Settings:
    def __init__(self, path: str | None = None):
        path = path or os.getenv("CONFIG_PATH", "config.yaml")
        p = Path(path)
        if not p.is_absolute():
            p = Path(__file__).resolve().parents[2] / path
        if not p.exists():
            raise ConfigError(f"config file not found: {p}")
        with open(p) as f:
            self.raw: dict[str, Any] = yaml.safe_load(f) or {}

        self.deriv = DerivSettings()

        mode = os.getenv("TRADING_MODE", self.raw.get("mode", MODE_RESEARCH)).strip().lower()
        if mode not in VALID_MODES:
            raise ConfigError(f"TRADING_MODE must be one of {VALID_MODES}, got {mode!r}")
        self.mode: str = mode

        # Section 65: make it hard to point a research/demo config at real
        # money by accident. Both switches must agree, and disagreement is a
        # hard startup failure rather than a silent downgrade -- a downgrade
        # would let a config that LOOKS live run quietly on demo, which is
        # its own (opposite) kind of nasty surprise.
        if self.mode == MODE_LIVE and not self.deriv.use_real_account:
            raise ConfigError(
                "TRADING_MODE=live requires DERIV_USE_REAL=true. Refusing to start: "
                "one switch says real money and the other says demo.")
        if self.deriv.use_real_account and self.mode != MODE_LIVE:
            raise ConfigError(
                f"DERIV_USE_REAL=true but TRADING_MODE={self.mode}. Refusing to start: "
                "set TRADING_MODE=live to trade real money, or DERIV_USE_REAL=false.")

        self.symbols: list[str] = self._symbols()
        self.currency: str = os.getenv("CURRENCY", self.raw.get("currency", "USD"))
        self.log_level: str = os.getenv("LOG_LEVEL", self.raw.get("logging", {}).get("level", "INFO"))
        self.db_path: str = os.getenv("DB_PATH", self.raw.get("storage", {}).get("path", "data/bot.db"))

        self._apply_env_overrides()

    def _symbols(self) -> list[str]:
        env = os.getenv("SYMBOLS", "").strip()
        if env:
            return [s.strip() for s in env.split(",") if s.strip()]
        return list(self.raw.get("symbols", ["R_100"]))

    def _apply_env_overrides(self) -> None:
        g = self.raw.setdefault("gating", {})
        g["min_edge"] = _env_float("MIN_EDGE", g.get("min_edge", 0.02))
        g["min_expected_value"] = _env_float("MIN_EV", g.get("min_expected_value", 0.0))
        g["min_samples"] = _env_int("MIN_SAMPLES", g.get("min_samples", 2000))
        g["min_model_agreement"] = _env_float(
            "MIN_MODEL_AGREEMENT", g.get("min_model_agreement", 0.60))
        g["max_probability_uncertainty"] = _env_float(
            "MAX_PROB_UNCERTAINTY", g.get("max_probability_uncertainty", 0.02))
        g["min_calibration_quality"] = _env_float(
            "MIN_CALIBRATION_QUALITY", g.get("min_calibration_quality", 0.50))
        g["randomness_alpha"] = _env_float("RANDOMNESS_ALPHA", g.get("randomness_alpha", 0.01))

        r = self.raw.setdefault("risk", {})
        r["base_stake"] = _env_float("BASE_STAKE", r.get("base_stake", 1.0))
        r["max_stake"] = _env_float("MAX_STAKE", r.get("max_stake", 5.0))
        r["max_daily_loss"] = _env_float("MAX_DAILY_LOSS", r.get("max_daily_loss", 25.0))
        r["max_drawdown"] = _env_float("MAX_DRAWDOWN", r.get("max_drawdown", 50.0))
        r["max_consecutive_losses"] = _env_int(
            "MAX_CONSECUTIVE_LOSSES", r.get("max_consecutive_losses", 8))
        r["max_trades_per_day"] = _env_int("MAX_TRADES_PER_DAY", r.get("max_trades_per_day", 200))
        r["max_concurrent_trades"] = _env_int("MAX_CONCURRENT_TRADES", r.get("max_concurrent_trades", 1))
        r["cooldown_seconds"] = _env_float("COOLDOWN_SECONDS", r.get("cooldown_seconds", 0.0))
        r["stale_tick_seconds"] = _env_float("STALE_TICK_SECONDS", r.get("stale_tick_seconds", 30.0))

        s = self.raw.setdefault("staking", {})
        # Spec Section 31: progressive staking is separate from prediction
        # and disabled by default. Prior live evidence in this account: at
        # negative expectancy a martingale converts a slow bleed into a fast
        # one, so this stays off unless deliberately switched on.
        s["method"] = os.getenv("STAKING_METHOD", s.get("method", "fixed"))
        s["kelly_fraction"] = _env_float("KELLY_FRACTION", s.get("kelly_fraction", 0.25))

    @property
    def gating(self) -> dict:
        return self.raw["gating"]

    @property
    def risk(self) -> dict:
        return self.raw["risk"]

    @property
    def staking(self) -> dict:
        return self.raw["staking"]

    @property
    def is_research(self) -> bool:
        return self.mode == MODE_RESEARCH

    def get(self, *keys: str, default: Any = None) -> Any:
        node: Any = self.raw
        for k in keys:
            if not isinstance(node, dict) or k not in node:
                return default
            node = node[k]
        return node


_SETTINGS: Settings | None = None


def get_settings(path: str | None = None) -> Settings:
    global _SETTINGS
    if _SETTINGS is None:
        _SETTINGS = Settings(path)
    return _SETTINGS
