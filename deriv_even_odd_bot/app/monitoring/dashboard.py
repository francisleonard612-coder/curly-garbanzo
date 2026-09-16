"""
Terminal dashboard (spec Sections 38, 39, 61).

Renders the full decision context every tick. The design priority is that
a human can tell, at a glance, WHY the bot is not trading -- which on this
instrument is what it will be doing essentially always. A dashboard that
only lights up on trades would look broken for weeks at a time.
"""
from __future__ import annotations

import shutil
import time


def _bar(value: float, lo: float, hi: float, width: int = 20) -> str:
    if value != value:
        return " " * width
    frac = max(0.0, min(1.0, (value - lo) / (hi - lo) if hi > lo else 0.0))
    filled = int(frac * width)
    return "#" * filled + "-" * (width - filled)


def _fmt(v, spec="{:.4f}", dash="--"):
    if v is None or (isinstance(v, float) and v != v):
        return dash
    try:
        return spec.format(v)
    except (TypeError, ValueError):
        return str(v)


class Dashboard:
    def __init__(self, mode: str):
        self.mode = mode
        self.started = time.time()

    def render(self, *, pipeline, client, risk, last_decision, db_summary=None) -> str:
        w = min(shutil.get_terminal_size((100, 40)).columns, 100)
        line = "=" * w
        thin = "-" * w
        st = pipeline.state
        last = st.last
        d = last_decision
        out: list[str] = [line]
        out.append(f" DERIV EVEN/ODD ENGINE   mode={self.mode.upper()}   "
                   f"uptime={int(time.time() - self.started)}s   state={pipeline.state_name}")
        out.append(line)

        conn = "CONNECTED" if client and client.is_connected else "DISCONNECTED"
        age = client.seconds_since_last_message() if client else float("inf")
        out.append(f" CONNECTION  {conn}   last msg {_fmt(age, '{:.1f}')}s ago   "
                   f"ticks={st.total_count}")

        if last:
            out.append(f" MARKET      {st.symbol}   quote={last.quote_raw}   "
                       f"digit={last.digit}   parity={'EVEN' if last.parity == 0 else 'ODD'}"
                       f"   exact={last.exact}")
        out.append(thin)

        if d:
            out.append(f" PREDICTION  P(EVEN)={_fmt(d.p_even_ensemble)}  "
                       f"P(ODD)={_fmt(1 - d.p_even_ensemble if d.p_even_ensemble else None)}")
            out.append(f"             calibrated={_fmt(d.calibrated_p_even)}  "
                       f"CI=[{_fmt(d.probability_lower)}, {_fmt(d.probability_upper)}]")
            out.append(f"             derived={_fmt(d.p_even_digit_derived)}  "
                       f"direct={_fmt(d.p_even_direct)}  "
                       f"agreement={_fmt(d.agreement_fraction, '{:.2f}')}  "
                       f"dispersion={_fmt(d.dispersion)}")
            out.append(f"             regime={d.regime or '--'}  "
                       f"entropy={_fmt(d.entropy)}  "
                       f"cal_quality={_fmt(d.calibration_quality, '{:.3f}')}  "
                       f"random_tradeable={d.randomness_tradeable}")
            if d.digit_probabilities:
                probs = "  ".join(f"{i}:{p:.3f}" for i, p in enumerate(d.digit_probabilities))
                out.append(f"             {probs}")
            out.append(thin)
            out.append(f" ECONOMICS   stake={_fmt(d.stake, '{:.2f}')}  "
                       f"payout={_fmt(d.payout, '{:.2f}')}  "
                       f"break_even={_fmt(d.break_even_probability)}")
            out.append(f"             edge={_fmt(d.edge, '{:+.4f}')}  "
                       f"conservative={_fmt(d.conservative_edge, '{:+.4f}')}  "
                       f"EV={_fmt(d.expected_value, '{:+.4f}')}  "
                       f"quality={_fmt(d.quality_score, '{:.1f}')}")
        out.append(thin)

        r = risk.snapshot()
        out.append(f" TRADING     balance={_fmt(r['balance'], '{:.2f}')}  "
                   f"session={_fmt(r['session_pnl'], '{:+.2f}')}  "
                   f"daily={_fmt(r['daily_pnl'], '{:+.2f}')}  "
                   f"dd={_fmt(r['drawdown'], '{:.2f}')}")
        out.append(f"             trades_today={r['trades_today']}  open={r['open']}  "
                   f"consec_losses={r['consecutive_losses']}"
                   + ("  [EMERGENCY STOP: " + r['stop_reason'] + "]" if r['stopped'] else ""))
        if db_summary:
            out.append(f"             settled={db_summary['trades']}  "
                       f"wins={db_summary['wins']}  "
                       f"win_rate={_fmt(db_summary['win_rate'], '{:.3f}')}  "
                       f"pnl={_fmt(db_summary['pnl'], '{:+.2f}')}")
        out.append(thin)

        report = pipeline.ensemble.health_report()
        out.append(" MODELS      name                   n     brier  skill    weight  health")
        for name, m in sorted(report.items()):
            out.append(f"             {name:<22} {m['n']:>5}  "
                       f"{_fmt(m['brier'], '{:.4f}')}  {_fmt(m['brier_skill'], '{:+.3f}')}  "
                       f"{_fmt(m['weight'], '{:.3f}')}   {m['health']}")
        out.append(thin)

        if d:
            out.append(f" DECISION    {d.decision}")
            out.append(f" REASON      {d.reason_code}")
            for chunk in _wrap(d.explanation or "", w - 14):
                out.append(f"             {chunk}")
            failed = [g for g in d.gates if not g.passed]
            if failed:
                out.append(f" BLOCKED BY  {failed[0].code}: {failed[0].explanation[:w-26]}")
        out.append(line)
        return "\n".join(out)


def _wrap(text: str, width: int) -> list[str]:
    words = text.split()
    lines, cur = [], ""
    for word in words:
        if len(cur) + len(word) + 1 > width:
            lines.append(cur)
            cur = word
        else:
            cur = f"{cur} {word}".strip()
    if cur:
        lines.append(cur)
    return lines or [""]
