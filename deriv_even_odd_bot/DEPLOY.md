# Deploying to Railway with Supabase

Read the whole page before the first deploy. Two things on it (replica count,
and what a restart does to the models) will cost real money if skipped.

---

## 1. Supabase

1. Create a project. Any region; pick one near `ws.derivws.com` if you care
   about write latency, though the bot never blocks a decision on a write.
2. **SQL Editor → New query →** paste all of `supabase/schema.sql` → **Run**.
   The bot verifies the five tables exist on startup and refuses to run if they
   do not, so if this step is skipped you will find out immediately rather than
   after a silent, dataless night.
3. **Project Settings → Database → Connection string → URI.** Take the one on
   **port 6543** (the pooler), not 5432. Replace `[YOUR-PASSWORD]` with your
   database password.

   ```
   postgresql://postgres.abcdefgh:PASSWORD@aws-0-eu-west-1.pooler.supabase.com:6543/postgres
   ```

   Port 5432 caps at a small number of direct connections and this process
   holds one open for its whole life. On 6543 that is free.

4. Use the **service role** credentials. `supabase/schema.sql` enables RLS with
   read-only policies for `authenticated`, so an anon connection writes nothing
   and raises nothing — the bot would appear to run perfectly and record
   absolutely nothing.

---

## 2. Railway

1. **New Project → Deploy from GitHub repo** (or `railway up` from the repo
   root). `railway.json` and `runtime.txt` are already present; Nixpacks reads
   `requirements.txt` and the start command is `python -m app.main`.
2. **Settings → Deploy → Replicas: 1.** Non-negotiable — see below.
3. Add the variables in section 3.
4. Deploy, then watch the logs for the startup banner. It prints the resolved
   mode and backend. If it says `sqlite` you have not set `DB_BACKEND`.

### Replicas must stay at 1

Railway will happily run two copies. Two copies means two bots watching the
same tick stream with the same account token, each unaware of the other's open
contracts. The `idempotency_key` UNIQUE constraint stops one *process* from
retrying into a duplicate; it does nothing about two processes that generated
different keys for two genuinely separate buys. `max_concurrent_trades: 1` is
enforced per-process, in memory.

The same applies to redeploys: Railway's default is to start the new instance
before draining the old one, so during a rollout you briefly have two. Set
**Settings → Deploy → Overlap: 0** (or accept the window only in `research`
mode).

### Auth, and what to do if it 401s

The client defaults to the REST OTP exchange: it calls
`/trading/v1/options/accounts` to resolve your demo or real account, then
`/trading/v1/options/accounts/{id}/otp` to get a pre-authenticated WebSocket
URL. The OTP is single-use and expires in **120 seconds**, so a fresh one is
minted on every connect and every reconnect.

If startup fails with `AuthFailed`, read the message before changing anything —
it reports `token_len` and `token_has_surrounding_whitespace` (never the token
itself). A trailing newline pasted into a Railway variable is invisible in the
dashboard and produces a 401 identical to a revoked token. `DerivSettings` now
strips both `DERIV_APP_ID` and `DERIV_API_TOKEN`, but the diagnostic stays
because it distinguishes a locally malformed token from a genuinely rejected
one.

If the accounts endpoint returns 404 for your app_id, set
`DERIV_AUTH_MODE=legacy`.

### Rate limits

Deriv counts `proposal`, `proposal_open_contract`, `buy` and `sell` against
**one shared budget of 360 requests/minute per connection** — not 360 each.
This engine requests a proposal on nearly every tick per symbol, so at two
symbols the budget is gone in well under a minute without pacing. The client
paces to 300/min before sending rather than absorbing rejections.

A consequence worth knowing: **at two symbols the pacing is the binding
constraint on evaluation rate, not the models.** `client.rate_limit_status()`
reports how often and how long it has waited. If `waits` climbs steadily, add
a proposal cache or reduce symbols rather than raising the cap.

### What a restart costs you

Railway restarts containers — on deploy, on OOM, on platform maintenance. Every
restart loses **all in-memory state**: model accumulators, the calibrator's
fitted curve, the ensemble's learned weights, the CUSUM, the persistence
trackers. The bot reseeds from Deriv's tick history on boot, but the calibrator
needs its own minimum sample before `is_fitted` goes true, and until then
`NO_TRADE_CALIBRATION_UNFITTED` is a hard gate.

Practical consequence: **a bot that restarts every few hours will never trade,
and this is indistinguishable from a deadlock unless you look.** Check
`v_rejection_breakdown` — if `NO_TRADE_CALIBRATION_UNFITTED` dominates and the
`events` table shows repeated startups, the problem is restart frequency, not
configuration. Do not respond by lowering a threshold.

Only the Supabase tables survive a restart. That is the entire reason for
`DB_BACKEND=postgres`.

---

## 3. Railway variables

### Required

| Variable | Example | Notes |
|---|---|---|
| `DERIV_API_TOKEN` | `a1b2c3...` | Deriv API token. Scopes: **Read** and **Trade**. Do **not** grant Payments or Admin. |
| `DERIV_APP_ID` | `1089` | `1089` is Deriv's public demo app id; register your own for live. |
| `DB_BACKEND` | `postgres` | Anything else means the history dies on redeploy. |
| `DATABASE_URL` | `postgresql://postgres.xxx:PASS@...pooler.supabase.com:6543/postgres` | Supabase **pooled** URI, service role. Mark as secret. |
| `TRADING_MODE` | `research` | `research` \| `demo` \| `live`. Start here. |
| `DERIV_USE_REAL` | `false` | Second switch. See the two-switch table below. |

### Recommended

| Variable | Default | Notes |
|---|---|---|
| `SYMBOLS` | `R_100,1HZ100V` | Comma-separated. Each is a separate pipeline and a separate memory footprint. |
| `CURRENCY` | `USD` | Must match the account. |
| `BASE_STAKE` | `1.0` | Fixed currency amount, never a percentage at this layer. |
| `MAX_STAKE` | `5.0` | Hard ceiling regardless of what staking computes. |
| `MAX_DAILY_LOSS` | `25.0` | Triggers emergency stop, which does **not** auto-clear. |
| `MAX_DRAWDOWN` | `50.0` | Same. |
| `MIN_SAMPLES` | `2000` | Ticks before any trade is considered. |
| `RANDOMNESS_ALPHA` | `0.01` | Family-wide alpha for the battery. |
| `LOG_LEVEL` | `INFO` | `DEBUG` is very chatty at ~2 ticks/sec. |
| `DERIV_AUTH_MODE` | `otp` | `otp` exchanges the token for a pre-authenticated WS URL via the Options REST API. `legacy` sends an authorize message instead — use it if you are on app_id `1089` and the accounts endpoint 404s. |
| `DERIV_API_BASE_URL` | `https://api.derivws.com` | REST base for the OTP exchange. Only used when `DERIV_AUTH_MODE=otp`. |
| `DERIV_ACCOUNT_ID` | *(blank)* | Pin one Options account. Blank auto-resolves to demo or real from `DERIV_USE_REAL`. |
| `DERIV_MAX_REQUESTS_PER_MINUTE` | `300` | Client-side pacing for Deriv's shared 360/min budget. Lower it if you see rate-limit errors; do not raise it. |
| `CONFIG_PATH` | `config.yaml` | Everything in `config.yaml` is env-overridable. |
| `TZ` | `Africa/Nairobi` | Only affects log timestamps and the daily-loss rollover boundary. |

### The two-switch live guard

Real money requires **both** switches to agree. Disagreement is a hard startup
failure, never a silent downgrade.

| `TRADING_MODE` | `DERIV_USE_REAL` | Behaviour |
|---|---|---|
| `research` | `false` | Full pipeline, full logging. **BUY is never called.** |
| `demo` | `false` | Real order flow, demo balance. |
| `live` | `true` | Real money. |
| `live` | `false` | **Refuses to start.** |
| anything else | `true` | **Refuses to start.** |

Set both as Railway variables, not in a committed `.env`. `.env` is gitignored;
keep it that way.

---

## 4. First-run checklist

Run in this order. Do not skip to `live` because the earlier stages were quiet —
quiet is the expected output.

1. **`research`, 24 hours.** Confirm ticks arrive, the calibrator fits, and
   `decisions` fills up. Then query:

   ```sql
   select * from v_rejection_breakdown;
   select * from v_shadow_thresholds;
   ```

   Expect `NO_TRADE_NEGATIVE_EV` to dominate with a mean edge near −0.013 at a
   1.95x payout. That is the house edge, measured on your own stream.

2. **Check `qualifying_positive_ev` in `v_shadow_thresholds`.** If it is zero at
   every threshold, no configuration change will produce a profitable trade.
   The levers that actually move break-even are payout and instrument, not the
   cutoff.

3. **`demo`, a week.** Only if step 2 showed something. Compare realized win
   rate against `v_score_vs_outcome`.

4. **`live`** — only with both switches, a stake you would shrug at, and
   `MAX_DAILY_LOSS` set to a number you have actually decided on.

---

## 5. Operating it

```sql
-- why is it not trading?
select * from v_rejection_breakdown;

-- does a higher opportunity score predict a higher win rate?
select * from v_score_vs_outcome;

-- daily P/L
select * from v_daily_pnl;

-- has the randomness battery ever claimed a departure?
select * from randomness_checks where tradeable order by ts desc limit 20;

-- restarts and errors
select * from events where level in ('ERROR','WARNING') order by ts desc limit 50;
```

An emergency stop does not auto-clear. Clear it deliberately, after reading why
it fired, by restarting the service — and read `events` first.

### Offline analysis

```bash
python -m app.backtest.simulator            # not a script; import it
```

```python
from app.backtest.simulator import walk_forward
from app.execution.engine import SymbolPipeline
from app.config.settings import Settings

s = Settings()
print(walk_forward(lambda: SymbolPipeline("R_100", 0.01, s), digits,
                   n_blocks=5, payout_multiple=1.95).report())
```

The simulator refuses to run through a live pipeline (`gate._live`), so it
cannot pollute the running bot's deadlock diagnostics. Run it locally or in a
Railway one-off command, not inside the worker process.

---

## 6. Costs

| | |
|---|---|
| Railway worker | ~$5/mo on Hobby; one always-on container, no HTTP service needed |
| Supabase | Free tier is 500 MB. At ~2 ticks/sec across two symbols, `decisions` grows ~350k rows/day and fills it in roughly three weeks |

When it fills: export first, then read the retention block at the bottom of
`supabase/schema.sql` before deleting anything. The no-trade rows are the
evidence the engine is declining correctly.
