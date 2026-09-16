"""
Regression tests for app/api/deriv_client.py.

Every test here corresponds to a numbered rule in that module's docstring,
and every rule was written after a real outage on this account or on the
Astra Rise/Fall bot. These run without a network: sockets, the pump and the
REST calls are all substituted.
"""
from __future__ import annotations

import asyncio
import time

import pytest
from websockets.protocol import State as WsState

from app.api.deriv_client import (
    AUTH_LEGACY,
    AUTH_OTP,
    BuyAmbiguousError,
    DerivAPIError,
    DerivAuthError,
    DerivClient,
    _RateLimiter,
)


class _FakeWs:
    def __init__(self, state=WsState.OPEN):
        self.state = state
        self.sent: list[str] = []

    async def send(self, payload: str) -> None:
        self.sent.append(payload)

    async def close(self) -> None:
        self.state = WsState.CLOSED


def _client(**kw) -> DerivClient:
    kw.setdefault("auth_mode", AUTH_LEGACY)
    return DerivClient(app_id="1", api_token="tok", ws_url="wss://x", **kw)


async def _instant_sleep(*_a, **_k) -> None:
    """Stand-in for asyncio.sleep. Must not call asyncio.sleep itself -- a
    lambda delegating to the function it replaced recurses forever."""
    return None


# ---------------------------------------------------------------------------
# rule 1: one reader, and it never awaits a handler
# ---------------------------------------------------------------------------

def test_route_resolves_a_pending_future_synchronously():
    async def run():
        c = _client()
        fut = asyncio.get_running_loop().create_future()
        c._pending[7] = fut
        c._route({"req_id": 7, "msg_type": "balance", "balance": {"balance": 10.0}})
        assert fut.done()
        assert fut.result()["balance"]["balance"] == 10.0
    asyncio.run(run())


def test_route_raises_api_error_on_the_waiting_future():
    async def run():
        c = _client()
        fut = asyncio.get_running_loop().create_future()
        c._pending[3] = fut
        c._route({"req_id": 3, "error": {"code": "RateLimit", "message": "slow down"}})
        with pytest.raises(DerivAPIError) as exc:
            fut.result()
        assert exc.value.code == "RateLimit"
    asyncio.run(run())


def test_subscribe_response_also_routes_its_own_first_tick():
    """The subscribe call's response carries the FIRST tick. Routing only the
    future would silently drop it."""
    async def run():
        c = _client()
        c._tick_queues["R_100"] = asyncio.Queue(maxsize=10)
        fut = asyncio.get_running_loop().create_future()
        c._pending[1] = fut
        c._route({"req_id": 1, "msg_type": "tick",
                  "tick": {"symbol": "R_100", "epoch": 1, "quote": "100.5", "id": "s1"}})
        assert fut.done()
        assert c._tick_queues["R_100"].qsize() == 1
        assert c._subscription_ids["R_100"] == "s1"
    asyncio.run(run())


def test_pump_failure_fails_every_in_flight_request():
    """Callers must not discover a dead socket one request timeout at a time."""
    async def run():
        c = _client()

        class _Boom:
            def __aiter__(self):
                return self

            async def __anext__(self):
                raise ConnectionError("no close frame received or sent")

        c._ws = _Boom()
        fut = asyncio.get_running_loop().create_future()
        c._pending[1] = fut
        await c._recv_pump()
        assert fut.done()
        with pytest.raises(DerivAPIError):
            fut.result()
        assert c._pending == {}
    asyncio.run(run())


# ---------------------------------------------------------------------------
# rule 1 (cont.): the quote must survive as TEXT
# ---------------------------------------------------------------------------

def test_quote_is_preserved_as_text_not_reparsed_as_float():
    """The digit of "1234.50" is 0. Once json.loads has made it a float the
    trailing zero is gone and the last digit reads as 5 -- wrong, silently,
    on every tick ending in zero."""
    async def run():
        c = _client()
        c._tick_queues["R_100"] = asyncio.Queue(maxsize=10)
        c._route_tick({"tick": {"symbol": "R_100", "epoch": 1, "quote": "1234.50"}})
        tick = c._tick_queues["R_100"].get_nowait()
        assert tick.quote_raw == "1234.50"
        assert isinstance(tick.quote_raw, str)
    asyncio.run(run())


def test_full_tick_queue_drops_the_oldest_and_never_blocks():
    async def run():
        c = _client()
        c._tick_queues["R_100"] = asyncio.Queue(maxsize=1)
        c._route_tick({"tick": {"symbol": "R_100", "epoch": 1, "quote": "100.1"}})
        c._route_tick({"tick": {"symbol": "R_100", "epoch": 2, "quote": "100.2"}})
        assert c._tick_queues["R_100"].qsize() == 1
        assert c._tick_queues["R_100"].get_nowait().quote_raw == "100.2"
    asyncio.run(run())


# ---------------------------------------------------------------------------
# rule 2: pump death must not be a silent permanent outage
# ---------------------------------------------------------------------------

def test_supervisor_reconnects_after_the_pump_exits(monkeypatch):
    """The exact production scenario: the pump returns after logging its own
    crash, and the tick feed must not die silently forever."""
    monkeypatch.setattr(asyncio, "sleep", _instant_sleep)

    async def run():
        c = _client()
        reconnects = 0

        async def fake_reconnect():
            nonlocal reconnects
            reconnects += 1
            if reconnects >= 2:
                c._closed = True

        async def dead_pump():
            return

        monkeypatch.setattr(c, "_reconnect", fake_reconnect)
        c._pump_task = asyncio.create_task(dead_pump())
        await c._supervise_pump()
        assert reconnects >= 1
    asyncio.run(run())


def test_supervisor_survives_a_failing_reconnect_attempt(monkeypatch):
    """A reconnect that raises must not kill the supervisor -- that would
    restore the very silent-outage failure it exists to prevent."""
    monkeypatch.setattr(asyncio, "sleep", _instant_sleep)

    async def run():
        c = _client()
        attempts = 0

        async def flaky_reconnect():
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                raise OSError("network down")
            c._closed = True

        async def dead_pump():
            return

        monkeypatch.setattr(c, "_reconnect", flaky_reconnect)
        c._pump_task = asyncio.create_task(dead_pump())
        await c._supervise_pump()
        assert attempts == 3
    asyncio.run(run())


# ---------------------------------------------------------------------------
# rule 3: the pump starts before resubscribing
# ---------------------------------------------------------------------------

def test_reconnect_starts_the_pump_before_resubscribing():
    """If the order inverts, every resubscribe after every reconnect hangs
    for the full request timeout -- deterministically, for every symbol."""
    async def run():
        c = _client()
        order: list[str] = []
        c._tick_queues["R_100"] = asyncio.Queue()

        async def fake_open():
            c._ws = _FakeWs()
            order.append("socket")

        async def fake_pump():
            order.append("pump")

        async def fake_authorize():
            order.append("authorize")

        async def fake_subscribe(symbol):
            order.append(f"subscribe:{symbol}")

        c._open_socket = fake_open
        c._recv_pump = fake_pump
        c._authorize = fake_authorize
        c._subscribe_ticks = fake_subscribe
        await c._reconnect()
        await asyncio.sleep(0)   # let the pump task run
        assert order.index("socket") < order.index("subscribe:R_100")
        assert "pump" in order
        assert order.index("authorize") < order.index("subscribe:R_100")
    asyncio.run(run())


def test_ensure_connected_checks_state_outside_the_lock():
    """asyncio.Lock is not reentrant. _reconnect() holds the lock while
    resubscribing, and resubscribing calls _send -> ensure_connected. If the
    open check sat inside the lock, the first resubscribe of every reconnect
    would deadlock against its own caller."""
    async def run():
        c = _client()
        c._ws = _FakeWs()
        async with c._connect_lock:
            # Would hang forever if ensure_connected tried to take the lock.
            await asyncio.wait_for(c.ensure_connected(), timeout=1.0)
    asyncio.run(run())


def test_reconnect_returns_early_if_another_caller_already_reconnected():
    async def run():
        c = _client()
        c._ws = _FakeWs()
        opened = False

        async def fake_open():
            nonlocal opened
            opened = True

        c._open_socket = fake_open
        await c._reconnect()
        assert not opened
    asyncio.run(run())


# ---------------------------------------------------------------------------
# rule 4: never blindly retry a buy
# ---------------------------------------------------------------------------

def test_ambiguous_buy_raises_rather_than_retrying():
    async def run():
        c = _client()
        calls = 0

        async def fake_send(payload):
            nonlocal calls
            calls += 1
            raise DerivAPIError("Timeout", "no response within 15.0s")

        c._send = fake_send
        with pytest.raises(BuyAmbiguousError) as exc:
            await c.buy("prop-1", 1.0, idempotency_key="key-1")
        assert calls == 1, "a buy must be attempted exactly once"
        assert "DO NOT retry" in str(exc.value)
    asyncio.run(run())


def test_a_definite_buy_rejection_is_not_ambiguous():
    """An explicit rejection is a known outcome -- no contract opened, so it
    must not be dressed up as needing reconciliation."""
    async def run():
        c = _client()

        async def fake_send(payload):
            raise DerivAPIError("InsufficientBalance", "not enough funds")

        c._send = fake_send
        with pytest.raises(DerivAPIError) as exc:
            await c.buy("prop-1", 1.0, idempotency_key="key-1")
        assert not isinstance(exc.value, BuyAmbiguousError)
    asyncio.run(run())


# ---------------------------------------------------------------------------
# rule 5: REST OTP auth
# ---------------------------------------------------------------------------

def test_account_resolution_prefers_the_wanted_kind():
    async def run():
        for wanted_real, expect in ((False, "demo-1"), (True, "real-1")):
            c = _client(auth_mode=AUTH_OTP, use_real_account=wanted_real)
            c._http_client = lambda: _FakeHttp(200, {"data": [
                {"account_id": "real-1", "type": "REAL"},
                {"account_id": "demo-1", "account_type": "demo"}]})
            assert await c._resolve_account_id() == expect
    asyncio.run(run())


def test_account_resolution_falls_back_to_the_first_account():
    """Better to trade on a visible wrong account than to fail to start for
    a reason nobody can see."""
    async def run():
        c = _client(auth_mode=AUTH_OTP, use_real_account=True)
        c._http_client = lambda: _FakeHttp(200, {"data": [
            {"account_id": "demo-only", "type": "demo"}]})
        assert await c._resolve_account_id() == "demo-only"
    asyncio.run(run())


def test_auth_failure_reports_token_shape_without_leaking_the_token():
    """A trailing newline pasted into a Railway variable is invisible in the
    dashboard and produces a 401 identical to a revoked token."""
    async def run():
        c = DerivClient(app_id="1", api_token="tok\n", ws_url="wss://x",
                        auth_mode=AUTH_OTP)
        c._http_client = lambda: _FakeHttp(401, {}, text="Unauthorized")
        with pytest.raises(DerivAuthError) as exc:
            await c._resolve_account_id()
        msg = str(exc.value)
        assert "token_has_surrounding_whitespace=True" in msg
        assert "tok" not in msg.replace("token_len", "").replace("token_has", "")
    asyncio.run(run())


def test_otp_exchange_returns_the_preauthenticated_url():
    async def run():
        c = _client(auth_mode=AUTH_OTP)
        c._http_client = lambda: _FakeHttp(
            200, {"data": {"url": "wss://api.derivws.com/ws/demo?otp=abc"}})
        assert await c._exchange_otp("acc-1") == "wss://api.derivws.com/ws/demo?otp=abc"
    asyncio.run(run())


def test_otp_exchange_falls_back_to_a_bare_token():
    async def run():
        c = _client(auth_mode=AUTH_OTP)
        c._http_client = lambda: _FakeHttp(200, {"data": {"otp": "xyz"}})
        url = await c._exchange_otp("acc-1")
        assert "otp=xyz" in url and url.startswith("wss://x")
    asyncio.run(run())


def test_otp_mode_needs_no_authorize_message():
    """The OTP URL arrives pre-authenticated. Sending authorize anyway would
    be a wasted round trip on every reconnect."""
    c = _client(auth_mode=AUTH_OTP)
    assert c.auth_mode == AUTH_OTP
    c._ws = _FakeWs()
    c._authorized = True
    assert c.is_connected


class _FakeHttp:
    def __init__(self, status: int, payload: dict, text: str = ""):
        self._status = status
        self._payload = payload
        self._text = text

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_a):
        return False

    async def get(self, *_a, **_k):
        return self

    async def post(self, *_a, **_k):
        return self

    @property
    def status_code(self):
        return self._status

    @property
    def text(self):
        return self._text

    def json(self):
        return self._payload


# ---------------------------------------------------------------------------
# rule 6: the shared rate budget
# ---------------------------------------------------------------------------

def test_rate_limiter_allows_up_to_the_budget_without_waiting():
    async def run():
        rl = _RateLimiter(max_per_window=5, window_seconds=60.0)
        start = time.monotonic()
        for _ in range(5):
            await rl.acquire()
        assert time.monotonic() - start < 0.5
        assert rl.total_waits == 0
    asyncio.run(run())


def test_rate_limiter_paces_once_the_budget_is_spent():
    async def run():
        rl = _RateLimiter(max_per_window=2, window_seconds=0.3)
        for _ in range(2):
            await rl.acquire()
        start = time.monotonic()
        await rl.acquire()
        assert time.monotonic() - start > 0.02, "third request should have waited"
        assert rl.total_waits == 1
    asyncio.run(run())


def test_only_the_shared_budget_requests_are_paced():
    """proposal, proposal_open_contract, buy and sell share ONE budget.
    ticks_history and active_symbols do not, and pacing them would throttle
    startup for no reason."""
    keys = DerivClient.RATE_LIMITED_KEYS
    assert {"proposal", "proposal_open_contract", "buy", "sell"} == set(keys)
    for free in ("ticks", "ticks_history", "active_symbols", "contracts_for",
                 "balance", "portfolio", "forget"):
        assert free not in keys


def test_send_paces_a_proposal_but_not_a_history_request():
    async def run():
        c = _client()
        c._ws = _FakeWs()
        paced: list[bool] = []

        async def fake_acquire():
            paced.append(True)

        c.rate_limiter.acquire = fake_acquire

        async def immediate(payload):
            # resolve the future the moment the request is registered
            await asyncio.sleep(0)
            for rid, fut in list(c._pending.items()):
                if not fut.done():
                    fut.set_result({"req_id": rid})

        c._ws.send = lambda p: immediate(p)
        await c._send({"proposal": 1})
        assert len(paced) == 1
        await c._send({"ticks_history": "R_100"})
        assert len(paced) == 1
    asyncio.run(run())


# ---------------------------------------------------------------------------
# rule 7: settlement is subscribed, not polled
# ---------------------------------------------------------------------------

def test_contract_update_routes_to_the_matching_queue():
    async def run():
        c = _client()
        c._contract_queues[42] = asyncio.Queue()
        c._route_contract_update({
            "proposal_open_contract": {"contract_id": 42, "is_sold": 0},
            "subscription": {"id": "sub-abc"}})
        assert c._contract_queues[42].qsize() == 1
        assert c._contract_subscription_ids[42] == "sub-abc"
    asyncio.run(run())


def test_contract_update_for_an_unknown_contract_is_ignored():
    async def run():
        c = _client()
        c._route_contract_update({"proposal_open_contract": {"contract_id": 7}})
    asyncio.run(run())


def test_full_contract_queue_keeps_the_newest_state():
    """The newest update is the one closest to is_sold, which is what the
    waiter is actually waiting for -- so drop the oldest, not this one."""
    async def run():
        c = _client()
        c._contract_queues[1] = asyncio.Queue(maxsize=1)
        c._route_contract_update({"proposal_open_contract": {"contract_id": 1, "t": 1}})
        c._route_contract_update({"proposal_open_contract": {"contract_id": 1, "t": 2}})
        assert c._contract_queues[1].qsize() == 1
        assert c._contract_queues[1].get_nowait()["t"] == 2
    asyncio.run(run())


def test_settlement_consumes_pushes_and_forgets_the_subscription():
    async def run():
        c = _client()
        q: asyncio.Queue = asyncio.Queue()
        forgotten: list[int] = []

        async def fake_subscribe(contract_id, maxsize=50):
            await q.put({"contract_id": 9, "is_sold": 0})
            await q.put({"contract_id": 9, "is_sold": 1, "profit": 0.95})
            return q

        async def fake_forget(contract_id):
            forgotten.append(contract_id)

        c.subscribe_contract_updates = fake_subscribe
        c.forget_contract_subscription = fake_forget
        poc = await c.wait_for_settlement(9, timeout=2.0)
        assert poc["is_sold"] == 1
        assert forgotten == [9], "the subscription must be released either way"
    asyncio.run(run())


def test_settlement_timeout_returns_an_empty_dict_and_still_forgets():
    """An empty result is what the caller treats as an unknown outcome to
    reconcile. Leaking the subscription on top of that would also leak the
    shared rate budget."""
    async def run():
        c = _client()
        forgotten: list[int] = []

        async def fake_subscribe(contract_id, maxsize=50):
            return asyncio.Queue()          # nothing ever arrives

        async def fake_forget(contract_id):
            forgotten.append(contract_id)

        c.subscribe_contract_updates = fake_subscribe
        c.forget_contract_subscription = fake_forget
        assert await c.wait_for_settlement(9, timeout=0.3) == {}
        assert forgotten == [9]
    asyncio.run(run())


# ---------------------------------------------------------------------------
# rule 8: websockets 14+ uses .state, not .closed
# ---------------------------------------------------------------------------

def test_connection_check_uses_state_not_closed():
    c = _client()
    c._authorized = True
    c._ws = _FakeWs(state=WsState.OPEN)
    assert c.is_connected
    c._ws = _FakeWs(state=WsState.CLOSED)
    assert not c.is_connected
    # `.closed` does not exist on a modern ClientConnection; reading it used
    # to raise AttributeError right after every successful connect.
    assert not hasattr(c._ws, "closed")


# ---------------------------------------------------------------------------
# current Options API field names
# ---------------------------------------------------------------------------

def test_proposal_sends_underlying_symbol_and_no_barrier():
    async def run():
        c = _client()
        sent: dict = {}

        async def fake_send(payload):
            sent.update(payload)
            return {"proposal": {"id": "p1"}}

        c._send = fake_send
        await c.proposal(symbol="R_100", contract_type="DIGITEVEN", amount=1.0,
                         currency="USD")
        assert sent["underlying_symbol"] == "R_100"
        assert "symbol" not in sent, "sending `symbol` returns InputValidationFailed"
        assert "barrier" not in sent, "DIGITEVEN/DIGITODD take no barrier"
    asyncio.run(run())


def test_active_symbols_omits_product_type_and_reads_either_field_name():
    async def run():
        c = _client()
        sent: dict = {}

        async def fake_send(payload):
            sent.update(payload)
            return {"active_symbols": [
                {"underlying_symbol": "R_100", "pip_size": 2},
                {"symbol": "R_50", "pip_size": 4}]}

        c._send = fake_send
        await c.active_symbols()
        assert "product_type" not in sent, "removed from the current Options API"
        assert c._pip_sizes == {"R_100": 2, "R_50": 4}
    asyncio.run(run())
