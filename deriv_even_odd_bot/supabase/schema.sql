-- ===========================================================================
-- Deriv Even/Odd engine -- Supabase (PostgreSQL) schema
--
-- Apply once, in the Supabase SQL editor, BEFORE the first deploy:
--   Supabase dashboard -> SQL Editor -> New query -> paste -> Run
--
-- Mirrors app/storage/db.py at SCHEMA_VERSION 2. If you change one, change
-- the other: app/storage/postgres.py writes the same column list to both.
--
-- WHY THE no-trade ROWS ARE THE VALUABLE ONES. On this instrument a correctly
-- operating bot produces a table full of reasons it waited. That record is the
-- evidence the engine is working rather than broken, and it is what the
-- deadlock and shadow reports are reconstructed from. Do not prune it to save
-- space without first exporting it.
--
-- RETENTION. At ~2 ticks/second across two symbols this table grows by roughly
-- 350k rows/day. Supabase's free tier (500 MB) holds about three weeks of it.
-- The retention policy at the bottom is commented out on purpose -- read it and
-- decide, rather than inheriting someone else's answer.
-- ===========================================================================

create table if not exists schema_version (
    version integer primary key
);
insert into schema_version (version) values (2)
    on conflict (version) do nothing;

-- ---------------------------------------------------------------------------
-- every decision, traded or not (Section 23)
-- ---------------------------------------------------------------------------
create table if not exists decisions (
    id                      bigserial primary key,
    ts                      double precision not null,
    created_at              timestamptz not null default now(),
    symbol                  text not null,
    decision                text not null,
    reason_code             text not null,
    explanation             text,

    quote                   double precision,
    digit                   smallint,

    -- probability stack (Sections 7-10)
    p_even_digit_derived    double precision,
    p_even_direct           double precision,
    p_even_ensemble         double precision,
    calibrated_p_even       double precision,
    probability_lower       double precision,
    probability_upper       double precision,
    dispersion              double precision,
    agreement_fraction      double precision,

    -- diagnostics
    regime                  text,
    entropy                 double precision,
    calibration_quality     double precision,
    randomness_tradeable    boolean,

    -- economics, from the REAL proposal (Sections 11, 12)
    contract_type           text,
    stake                   double precision,
    payout                  double precision,
    break_even              double precision,
    edge                    double precision,
    conservative_edge       double precision,
    expected_value          double precision,
    quality_score           double precision,
    is_probe                boolean default false,

    -- the continuous layer (Sections 13-19)
    opportunity_score       double precision,
    opportunity_zone        text,
    opportunity_threshold   double precision,
    selectivity_multiplier  double precision,
    limiting_factor         text,
    capped_by               jsonb,
    randomness_evidence     double precision,
    regime_evidence         double precision,
    signal_persistence      double precision,
    contribution_detail     jsonb,
    hard_gate_failed        boolean default false,

    -- heavy payloads, stored only for TRADED rows (Section 37)
    digit_probabilities     jsonb,
    member_p_even           jsonb,
    model_weights           jsonb,
    gates                   jsonb
);

create index if not exists idx_decisions_ts        on decisions (ts desc);
create index if not exists idx_decisions_symbol_ts on decisions (symbol, ts desc);
create index if not exists idx_decisions_reason    on decisions (reason_code);
create index if not exists idx_decisions_zone      on decisions (opportunity_zone);
create index if not exists idx_decisions_limiting  on decisions (limiting_factor);
-- Partial index: trades are a tiny fraction of rows, and every review query
-- starts by filtering to them.
create index if not exists idx_decisions_traded
    on decisions (ts desc) where decision <> 'NO_TRADE';

-- ---------------------------------------------------------------------------
-- executed contracts
-- ---------------------------------------------------------------------------
create table if not exists trades (
    id               bigserial primary key,
    decision_id      bigint references decisions (id),
    ts               double precision not null,
    created_at       timestamptz not null default now(),
    symbol           text not null,
    contract_id      bigint unique,
    -- Duplicate-buy protection (Section 35). The UNIQUE constraint is the
    -- real defence: a retry at ANY layer hits it and fails loudly rather than
    -- opening a second contract.
    idempotency_key  text unique,
    contract_type    text,
    stake            double precision,
    payout           double precision,
    buy_price        double precision,
    entry_digit      smallint,
    exit_digit       smallint,
    won              boolean,
    pnl              double precision,
    settled_at       double precision,
    error            text
);

create index if not exists idx_trades_ts on trades (ts desc);
create index if not exists idx_trades_open
    on trades (symbol) where settled_at is null;

-- ---------------------------------------------------------------------------
-- per-model health over time (Sections 20, 21)
-- ---------------------------------------------------------------------------
create table if not exists model_performance (
    id           bigserial primary key,
    ts           double precision not null,
    created_at   timestamptz not null default now(),
    symbol       text,
    model        text,
    n            integer,
    brier        double precision,
    brier_skill  double precision,
    log_loss     double precision,
    accuracy     double precision,
    weight       double precision,
    health       text
);
create index if not exists idx_perf_ts on model_performance (ts desc);
create index if not exists idx_perf_model on model_performance (model, ts desc);

-- ---------------------------------------------------------------------------
-- the randomness battery's log (Section 3)
-- ---------------------------------------------------------------------------
-- `tradeable` is retained as a column but IS NO LONGER A VETO -- it records
-- what the battery concluded, while the engine consumes the continuous
-- evidence instead. If this column ever turns true, treat it as an
-- extraordinary claim and confirm it persists across independent re-scans
-- before believing it.
create table if not exists randomness_checks (
    id                     bigserial primary key,
    ts                     double precision not null,
    created_at             timestamptz not null default now(),
    symbol                 text,
    n_samples              integer,
    chi2_p                 double precision,
    runs_p                 double precision,
    even_rate              double precision,
    ci_lower               double precision,
    ci_upper               double precision,
    any_significant        boolean,
    economically_relevant  boolean,
    tradeable              boolean,
    summary                text
);
create index if not exists idx_randomness_ts on randomness_checks (ts desc);

-- ---------------------------------------------------------------------------
-- operational events
-- ---------------------------------------------------------------------------
create table if not exists events (
    id          bigserial primary key,
    ts          double precision not null,
    created_at  timestamptz not null default now(),
    level       text,
    category    text,
    message     text,
    detail      jsonb
);
create index if not exists idx_events_ts on events (ts desc);
create index if not exists idx_events_level on events (level, ts desc);

-- ===========================================================================
-- Views the operator actually reads
-- ===========================================================================

-- Section 16: why is it not trading?
create or replace view v_rejection_breakdown as
select
    symbol,
    reason_code,
    limiting_factor,
    count(*)                                        as n,
    round(avg(opportunity_score)::numeric, 2)       as mean_score,
    round(max(opportunity_score)::numeric, 2)       as best_score,
    round(avg(edge)::numeric, 5)                    as mean_edge,
    round(avg(expected_value)::numeric, 5)          as mean_ev,
    round(avg(payout / nullif(stake, 0))::numeric, 4) as mean_payout_multiple
from decisions
where decision = 'NO_TRADE'
group by symbol, reason_code, limiting_factor
order by n desc;

-- Section 17: what would a different threshold admit?
create or replace view v_shadow_thresholds as
select
    t.threshold,
    count(*) filter (where d.opportunity_score >= t.threshold)          as qualifying,
    count(*) filter (where d.opportunity_score >= t.threshold
                       and d.expected_value > 0)                        as qualifying_positive_ev,
    round(avg(d.edge) filter (where d.opportunity_score >= t.threshold)::numeric, 5)
                                                                        as mean_edge
from decisions d
cross join (values (40.0), (50.0), (55.0), (62.0), (70.0), (80.0)) as t(threshold)
where d.opportunity_score is not null
group by t.threshold
order by t.threshold;

-- Does a higher opportunity score actually predict a higher win rate?
-- The same question the walk-forward simulator answers offline, asked of live
-- data. Needs settled trades to be meaningful; it will be empty until then.
create or replace view v_score_vs_outcome as
select
    width_bucket(d.opportunity_score, 0, 100, 10) * 10 as score_bucket,
    count(*)                                            as n,
    round(avg(case when tr.won then 1.0 else 0.0 end)::numeric, 4) as win_rate,
    round(avg(d.edge)::numeric, 5)                      as mean_edge
from decisions d
join trades tr on tr.decision_id = d.id
where tr.settled_at is not null and d.opportunity_score is not null
group by 1
order by 1;

create or replace view v_daily_pnl as
select
    date_trunc('day', to_timestamp(ts)) as day,
    symbol,
    count(*)                            as trades,
    count(*) filter (where won)         as wins,
    round(sum(pnl)::numeric, 2)         as pnl
from trades
where settled_at is not null
group by 1, 2
order by 1 desc;

-- ===========================================================================
-- Row Level Security
--
-- The bot connects with the SERVICE ROLE key, which bypasses RLS. These
-- policies exist so that if you ever point a dashboard, a Supabase client, or
-- the anon key at this project, it cannot write. Enable them.
-- ===========================================================================
alter table decisions         enable row level security;
alter table trades            enable row level security;
alter table model_performance enable row level security;
alter table randomness_checks enable row level security;
alter table events            enable row level security;

do $$
declare t text;
begin
    foreach t in array array['decisions','trades','model_performance',
                             'randomness_checks','events']
    loop
        execute format(
            'drop policy if exists %I on %I', 'read_only_authenticated', t);
        execute format(
            'create policy %I on %I for select to authenticated using (true)',
            'read_only_authenticated', t);
    end loop;
end $$;

-- ===========================================================================
-- Retention -- READ BEFORE ENABLING
--
-- Deleting no-trade decisions destroys the evidence that the engine is
-- declining correctly, which is the most valuable thing in this database on a
-- CSPRNG instrument. If you must prune, prune the heavy JSONB first (it is
-- already null on no-trade rows) and keep the scalar columns, which are what
-- every view above reads.
--
-- Requires pg_cron: Supabase dashboard -> Database -> Extensions -> pg_cron.
-- ===========================================================================
-- create extension if not exists pg_cron;
-- select cron.schedule(
--     'prune-no-trade-decisions', '0 4 * * *',
--     $$delete from decisions
--        where decision = 'NO_TRADE'
--          and created_at < now() - interval '30 days'$$);
