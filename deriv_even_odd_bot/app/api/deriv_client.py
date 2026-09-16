"""
Deriv WebSocket API client (spec Sections 34, 35, 59).

Isolated from strategy by mandate (Section 34): this module knows about
sockets, request correlation, rate limits and reconnection. It knows nothing
about digits, models or trading decisions, and nothing here may import from
app/models, app/ensemble or app/execution.

=====================================================================
EIGHT RULES CARRIED FORWARD FROM REAL PRODUCTION FAILURES. Do not
"simplify" any of them away -- each looks like redundant ceremony and
each was written after an outage. Rules 5-8 were ported from the Astra
Rise/Fall bot's connection layer, which has run this exact flow live.
=====================================================================

1. ONE READER, ALWAYS. Exactly one coroutine may read the socket. Every
   request/response pair is correlated by req_id through a pending-futures
   table that ONLY the reader resolves. Any code path that awaits a
   response while also being the thing responsible for reading it will
   deadlock until timeout. The reader NEVER awaits a handler: ticks and
   contract updates are pushed onto queues with put_nowait, and pending
   futures are resolved with set_result. Astra's log of this failure was
   100% of trade attempts failing with "no response" / 1011 keepalive
   timeouts, because the pump awaited a handler that called proposal().

2. RECV-PUMP DEATH MUST NEVER BE A SILENT PERMANENT OUTAGE. Observed in
   production on both bots: the pump crashed on `ConnectionClosedError: no
   close frame received or sent` (keepalive ping timeout), logged, and
   returned. Nothing else reconnected the TICK path -- reconnection only
   ran from the request/response path, which a pure tick consumer never
   touches. The process stayed up, looked healthy in Railway, and silently
   stopped receiving ticks forever. `_supervise_pump()` watches the pump
   task and reconnects whenever it exits, for any reason. ONE long-lived
   supervisor for the client's whole life -- never re-spawned per
   reconnect, or two supervisors race to read the same socket.

3. THE PUMP MUST START BEFORE RESUBSCRIBING, NOT AFTER. This is the subtle
   one, and the first fix for (2) got it wrong. Resubscribing goes through
   _send(), which awaits a future only the pump can resolve (rule 1). If
   the new pump has not started yet, EVERY resubscribe after EVERY
   reconnect is guaranteed to hang for the full request timeout -- not
   occasionally, deterministically, for every symbol. Confirmed in a live
   log: clean re-auth immediately followed by a burst of resubscribe
   timeouts spaced exactly request_timeout apart. `_reconnect()` therefore
   starts the pump on the new socket BEFORE the resubscribe loop.

4. NEVER BLINDLY RETRY A BUY (Section 35). A retried buy whose first
   attempt actually succeeded opens two contracts. Buys carry a caller-
   supplied idempotency key, are attempted exactly once, and on ambiguous
   failure the caller must RECONCILE via portfolio rather than retry.

5. AUTH VIA THE REST OTP EXCHANGE, NOT THE LEGACY AUTHORIZE MESSAGE.
   Connect-then-send-authorize has produced handshake-time 401s on this
   account. The Options API exchanges the long-lived token for a
   pre-authenticated WebSocket URL instead:
     GET  {base}/trading/v1/options/accounts        -> pick demo or real
     POST {base}/trading/v1/options/accounts/{id}/otp -> {"data":{"url": ...}}
   The OTP is single-use and valid for 120 SECONDS, so the socket is opened
   immediately after the exchange and a fresh OTP is minted on every
   reconnect. `auth_mode="legacy"` keeps the old flow for app_id 1089.

6. PROPOSAL, PROPOSAL_OPEN_CONTRACT, BUY AND SELL SHARE ONE RATE BUDGET.
   Deriv counts all four against a single 360/minute per-connection budget
   -- not 360 each. This engine requests a proposal on nearly every tick
   per symbol, so at two symbols and ~2 ticks/second the budget is gone in
   well under a minute. Astra observed dozens of concurrent "You have
   reached the rate limit for proposal" errors inside the same second.
   `_RateLimiter` waits BEFORE sending rather than eating the rejection,
   per Deriv's own pace-your-bursts guidance.

7. SETTLEMENT IS SUBSCRIBED, NOT POLLED. The old implementation polled
   proposal_open_contract every second for up to 120 seconds: up to 120
   requests against the rule-6 budget for ONE position. Subscribing costs
   one request total and every update after that is pushed.

8. THE CONNECTION CHECK USES `.state`, NOT `.closed`. websockets 14+
   removed the boolean `.closed` property; reading it raises AttributeError
   on every call, which crashed Astra immediately after every successful
   connect. requirements.txt pins `websockets>=12.0` with no upper bound,
   so any current environment hits this.

ONE THING DELIBERATELY NOT PORTED FROM ASTRA: its client parses `quote`
into a float and derives the digit itself. This engine keeps the quote as
TEXT and extracts the digit in app/digits/extraction.py, because once
json.loads has made "1234.50" a float the trailing zero is gone and the
last digit is wrong. See that module's tests.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field

import websockets
from websockets.protocol import State as WsState

logger = logging.getLogger(__name__)

AUTH_OTP = "otp"
AUTH_LEGACY = "legacy"


class DerivAPIError(RuntimeError):
    def __init__(self, code: str, message: str, echo: dict | None = None):
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message
        self.echo = echo or {}


class DerivAuthError(DerivAPIError):
    def __init__(self, message: str):
        super().__init__("AuthFailed", message)


class BuyAmbiguousError(DerivAPIError):
    """A buy whose outcome could not be determined. MUST be reconciled
    against the portfolio, never retried (rule 4)."""


@dataclass
class TickMessage:
    symbol: str
    epoch: float
    quote_raw: str          # preserved as text -- see app/digits/extraction.py
    pip_size: int | float
    received_at: float = field(default_factory=time.time)


class _RateLimiter:
    """Sliding-window limiter for Deriv's shared per-connection budget (rule 6).

    Waits before sending rather than firing and handling the rejection after
    the fact. A rejected proposal still consumed a request, still cost a
    round trip, and still leaves the caller with no quote -- so pacing is
    strictly cheaper than recovering.
    """

    def __init__(self, max_per_window: int = 300, window_seconds: float = 60.0):
        self.max_per_window = max_per_window
        self.window_seconds = window_seconds
        self._timestamps: list[float] = []
        self._lock = asyncio.Lock()
        self.total_waits = 0
        self.total_wait_seconds = 0.0

    async def acquire(self) -> None:
        async with self._lock:
            waited = 0.0
            while True:
                now = time.monotonic()
                cutoff = now - self.window_seconds
                self._timestamps = [t for t in self._timestamps if t > cutoff]
                if len(self._timestamps) < self.max_per_window:
                    self._timestamps.append(now)
                    if waited:
                        self.total_waits += 1
                        self.total_wait_seconds += waited
                    return
                sleep_for = max(self._timestamps[0] - cutoff, 0.05)
                waited += sleep_for
                await asyncio.sleep(sleep_for)

    @property
    def used_in_window(self) -> int:
        cutoff = time.monotonic() - self.window_seconds
        return len([t for t in self._timestamps if t > cutoff])


class DerivClient:
    #: Requests sharing Deriv's single 360/min budget (rule 6). Capped at
    #: 300 below to leave headroom for the buy and settlement traffic that
    #: shares the same budget outside the proposal path.
    RATE_LIMITED_KEYS = frozenset({"proposal", "proposal_open_contract", "buy", "sell"})

    def __init__(self, app_id: str, api_token: str, ws_url: str,
                 request_timeout: float = 15.0, *,
                 api_base_url: str = "https://api.derivws.com",
                 auth_mode: str = AUTH_OTP,
                 account_id: str | None = None,
                 use_real_account: bool = False,
                 max_requests_per_minute: int = 300):
        self.app_id = app_id
        self.api_token = api_token
        self.ws_url = ws_url            # legacy flow, and OTP fallback
        self.api_base_url = api_base_url.rstrip("/")
        self.auth_mode = auth_mode
        self.account_id = account_id or None   # auto-resolved on first connect
        self.use_real_account = use_real_account
        self.request_timeout = request_timeout

        self._ws = None
        self._req_id = 0
        self._pending: dict[int, asyncio.Future] = {}
        self._tick_queues: dict[str, asyncio.Queue] = {}
        self._subscription_ids: dict[str, str] = {}
        self._contract_queues: dict[int, asyncio.Queue] = {}
        self._contract_subscription_ids: dict[int, str] = {}
        self._pip_sizes: dict[str, int | float] = {}
        self._pump_task: asyncio.Task | None = None
        self._supervisor_task: asyncio.Task | None = None
        self._connect_lock = asyncio.Lock()
        self._closed = False
        self._authorized = False
        self.last_message_at: float = 0.0
        self.rate_limiter = _RateLimiter(max_per_window=max_requests_per_minute)

    # ---- connection ------------------------------------------------------

    async def connect(self) -> None:
        async with self._connect_lock:
            if self.auth_mode == AUTH_OTP and not self.account_id:
                self.account_id = await self._resolve_account_id()
                logger.info("resolved Deriv account %s (wanted %s)", self.account_id,
                            "real" if self.use_real_account else "demo")
            await self._open_socket()
            # Rule 3: reader first, always.
            self._pump_task = asyncio.create_task(self._recv_pump(), name="deriv-pump")
            if self.auth_mode == AUTH_LEGACY:
                await self._authorize()
            # Rule 2: ONE long-lived supervisor for the client's whole life.
            self._supervisor_task = asyncio.create_task(
                self._supervise_pump(), name="deriv-pump-supervisor")
            logger.info("connected to Deriv (auth=%s)", self.auth_mode)

    async def _open_socket(self) -> None:
        """Opens a new socket, replacing self._ws.

        Does not touch the pump or resubscribe anything -- connect() and
        _reconnect() own that, because rule 3 makes the ORDER of those steps
        load-bearing.
        """
        if self.auth_mode == AUTH_OTP:
            # Rule 5: the OTP is single-use and expires in 120s, so it is
            # minted here, immediately before the socket opens, on every
            # connection including reconnects. Caching it across reconnects
            # would fail exactly when it matters -- during a long outage.
            url = await self._exchange_otp(self.account_id)
        else:
            url = f"{self.ws_url}?app_id={self.app_id}"
        self._ws = await websockets.connect(
            url, ping_interval=20, ping_timeout=60, close_timeout=5,
            max_size=4 * 1024 * 1024)
        # The OTP URL arrives pre-authenticated; the legacy flow must still
        # send an authorize message before it is true.
        self._authorized = self.auth_mode == AUTH_OTP
        self.last_message_at = time.time()

    # ---- REST auth (rule 5) ----------------------------------------------

    def _auth_headers(self) -> dict:
        return {"Deriv-App-ID": self.app_id,
                "Authorization": f"Bearer {self.api_token}"}

    @staticmethod
    def _http_client():
        try:
            import httpx
        except ImportError as exc:  # pragma: no cover - import guard
            raise DerivAuthError(
                "auth_mode='otp' requires httpx: pip install httpx, or set "
                "DERIV_AUTH_MODE=legacy") from exc
        return httpx.AsyncClient(timeout=15.0)

    async def _resolve_account_id(self) -> str:
        """Fetch the token's Options accounts and pick demo or real.

        Field matching checks BOTH `type` and `account_type` because the
        response has used each, and compares case-insensitively. If no
        account of the wanted kind exists, falls back to the first returned
        and logs loudly -- better to trade on a visible wrong account than
        to fail to start for a reason nobody can see.
        """
        if not self.api_token:
            raise DerivAuthError("DERIV_API_TOKEN is not set")
        url = f"{self.api_base_url}/trading/v1/options/accounts"
        async with self._http_client() as client:
            resp = await client.get(url, headers=self._auth_headers())
        if resp.status_code != 200:
            # Token diagnostics, never the token itself. A trailing newline
            # pasted into a Railway variable is indistinguishable from a
            # revoked token without these.
            raise DerivAuthError(
                f"fetching accounts failed: HTTP {resp.status_code} "
                f"{resp.text[:300]} (token_len={len(self.api_token)}, "
                f"token_has_surrounding_whitespace="
                f"{self.api_token != self.api_token.strip()}, "
                f"app_id={self.app_id!r})")
        body = resp.json()
        accounts = (body.get("data") or body.get("accounts")
                    or (body if isinstance(body, list) else []))
        if not accounts:
            raise DerivAuthError(f"no Options accounts for this token: {body}")

        wanted = "real" if self.use_real_account else "demo"
        for acc in accounts:
            kind = str(acc.get("type") or acc.get("account_type") or "").lower()
            if kind == wanted:
                account_id = acc.get("account_id") or acc.get("id")
                if account_id:
                    return account_id

        first = accounts[0]
        account_id = first.get("account_id") or first.get("id")
        if not account_id:
            raise DerivAuthError(f"account entry had no id field: {first}")
        logger.warning("no %r account found; falling back to %s", wanted, account_id)
        return account_id

    async def _exchange_otp(self, account_id: str) -> str:
        if not self.api_token:
            raise DerivAuthError("DERIV_API_TOKEN is not set")
        url = f"{self.api_base_url}/trading/v1/options/accounts/{account_id}/otp"
        async with self._http_client() as client:
            resp = await client.post(url, headers=self._auth_headers())
        if resp.status_code != 200:
            raise DerivAuthError(
                f"OTP exchange failed: HTTP {resp.status_code} {resp.text[:300]}")
        body = resp.json()
        data = body.get("data", body)
        auth_url = data.get("url") or data.get("websocket_url") or data.get("ws_url")
        if auth_url:
            return auth_url
        # Older deployments return a bare OTP rather than a ready URL.
        otp = data.get("otp") or data.get("token")
        if not otp:
            raise DerivAuthError(f"OTP response had no usable URL or token: {body}")
        return f"{self.ws_url}?app_id={self.app_id}&otp={otp}"

    async def _authorize(self) -> dict:
        """Legacy flow only."""
        if not self.api_token:
            raise DerivAuthError("DERIV_API_TOKEN is not set")
        resp = await self._send({"authorize": self.api_token})
        self._authorized = True
        return resp.get("authorize", {})

    async def close(self) -> None:
        self._closed = True
        for t in (self._pump_task, self._supervisor_task):
            if t:
                t.cancel()
        if self._ws is not None:
            await self._ws.close()

    # ---- reader ----------------------------------------------------------

    async def _recv_pump(self) -> None:
        """THE ONLY coroutine that reads the socket (rule 1).

        Never awaits a handler. Everything it does with a message is
        synchronous: put_nowait onto a queue, or set_result on a future.
        """
        try:
            async for raw in self._ws:
                self.last_message_at = time.time()
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    logger.warning("undecodable frame dropped")
                    continue
                self._route(msg)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error("recv pump crashed: %s", exc)
        finally:
            # Fail every in-flight request rather than leaving callers to
            # discover the dead socket via timeout, one slow request at a time.
            for fut in list(self._pending.values()):
                if not fut.done():
                    fut.set_exception(DerivAPIError("Disconnected", "socket closed"))
            self._pending.clear()

    def _route(self, msg: dict) -> None:
        msg_type = msg.get("msg_type")
        req_id = msg.get("req_id")

        if req_id is not None and req_id in self._pending:
            fut = self._pending.pop(req_id)
            if not fut.done():
                if "error" in msg:
                    err = msg["error"]
                    fut.set_exception(DerivAPIError(
                        err.get("code", "Unknown"), err.get("message", ""),
                        msg.get("echo_req")))
                else:
                    fut.set_result(msg)
            # A subscribe call's own response also carries the FIRST payload
            # and the subscription id. Routing it here too is why no opening
            # tick or contract state is ever lost.
            if msg_type == "tick":
                self._route_tick(msg)
            elif msg_type == "proposal_open_contract":
                self._route_contract_update(msg)
            return

        if msg_type == "tick":
            self._route_tick(msg)
        elif msg_type == "proposal_open_contract":
            self._route_contract_update(msg)
        else:
            logger.debug("unrouted message: %s", msg_type)

    def _route_tick(self, msg: dict) -> None:
        tick = msg.get("tick") or {}
        symbol = tick.get("symbol") or tick.get("underlying_symbol")
        if not symbol or symbol not in self._tick_queues:
            return
        q = self._tick_queues[symbol]
        if q.full():
            # Drop the OLDEST tick. The reader must never block, and a stale
            # tick is worth less than the one arriving now.
            try:
                q.get_nowait()
            except asyncio.QueueEmpty:
                pass
        q.put_nowait(TickMessage(
            symbol=symbol,
            epoch=float(tick.get("epoch", 0)),
            # Preserved as TEXT -- see the module docstring's closing note.
            quote_raw=self._raw_quote_text(tick),
            pip_size=self._pip_sizes.get(symbol, tick.get("pip_size", 2))))
        if tick.get("id"):
            self._subscription_ids[symbol] = tick["id"]

    def _route_contract_update(self, msg: dict) -> None:
        poc = msg.get("proposal_open_contract") or {}
        contract_id = poc.get("contract_id")
        if contract_id is None:
            return
        q = self._contract_queues.get(contract_id)
        if q is not None:
            try:
                q.put_nowait(poc)
            except asyncio.QueueFull:
                # Drop the OLDEST update, not this one: the newest state is
                # the one closest to is_sold, which is what the waiter wants.
                try:
                    q.get_nowait()
                except asyncio.QueueEmpty:
                    pass
                q.put_nowait(poc)
        sub_id = (msg.get("subscription") or {}).get("id") or poc.get("id")
        if sub_id:
            self._contract_subscription_ids[contract_id] = sub_id

    @staticmethod
    def _raw_quote_text(tick: dict) -> str:
        q = tick.get("quote")
        return q if isinstance(q, str) else repr(q)

    async def _supervise_pump(self) -> None:
        """Rule 2. Reconnects whenever the pump exits, with backoff."""
        backoff = 1.0
        while not self._closed:
            try:
                if self._pump_task:
                    await self._pump_task
            except asyncio.CancelledError:
                raise
            except Exception:
                pass
            if self._closed:
                return
            logger.warning("pump exited -- reconnecting")
            while not self._closed:
                await asyncio.sleep(backoff)
                try:
                    await self._reconnect()
                    backoff = 1.0
                    break
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    backoff = min(backoff * 2, 30.0)
                    logger.error("reconnect failed (%s); next attempt in %.1fs",
                                 exc, backoff)

    async def ensure_connected(self) -> None:
        """Reconnects a dead socket from the request path.

        The supervisor (rule 2) handles the tick path, but it may be mid-
        backoff when a request arrives. Without this, every request during
        that window fails instead of waiting for a socket that is about to
        exist anyway.

        THE OPEN CHECK IS OUTSIDE THE LOCK ON PURPOSE. _reconnect() holds
        _connect_lock while resubscribing, and resubscribing goes through
        _send() -> ensure_connected(). asyncio.Lock is NOT reentrant, so if
        this method acquired the lock before checking, the reconnect path
        would deadlock against itself on the first resubscribe of every
        reconnect.
        """
        if self._ws is not None and self._ws.state is WsState.OPEN:
            return
        await self._reconnect()

    async def _reconnect(self) -> None:
        async with self._connect_lock:
            if self._ws is not None and self._ws.state is WsState.OPEN:
                return   # someone else reconnected while this caller waited
            await self._open_socket()
            # RULE 3, THE WHOLE POINT: start the reader on the new socket
            # BEFORE any resubscribe, because resubscribes await responses
            # only the reader can deliver.
            self._pump_task = asyncio.create_task(self._recv_pump(), name="deriv-pump")
            if self.auth_mode == AUTH_LEGACY:
                await self._authorize()
            for symbol in list(self._tick_queues.keys()):
                self._subscription_ids.pop(symbol, None)
                try:
                    await self._subscribe_ticks(symbol)
                except Exception as exc:
                    logger.error("failed to resubscribe %s: %s", symbol, exc)
            # Contract subscriptions are deliberately NOT resubscribed:
            # wait_for_settlement() is bounded and already treats silence as
            # an unknown outcome the caller reconciles, so there is no
            # permanent-outage risk there the way there is for ticks.

    # ---- request/response ------------------------------------------------

    async def _send(self, payload: dict) -> dict:
        # Rule 6: pace before sending, not after being rejected.
        if self.RATE_LIMITED_KEYS.intersection(payload.keys()):
            await self.rate_limiter.acquire()
        await self.ensure_connected()
        if self._ws is None or self._ws.state is not WsState.OPEN:
            raise DerivAPIError("Disconnected", "socket is not open")
        self._req_id += 1
        req_id = self._req_id
        payload = {**payload, "req_id": req_id}
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        self._pending[req_id] = fut
        try:
            await self._ws.send(json.dumps(payload))
            return await asyncio.wait_for(fut, timeout=self.request_timeout)
        except asyncio.TimeoutError:
            raise DerivAPIError("Timeout", f"no response within {self.request_timeout}s")
        finally:
            self._pending.pop(req_id, None)

    # ---- discovery (Section 1) -------------------------------------------

    async def active_symbols(self) -> list[dict]:
        """Note: `product_type` was REMOVED from this request in Deriv's
        current Options API -- sending it returns a validation error. The
        response field was renamed `symbol` -> `underlying_symbol`; both are
        read here so the client works against either generation."""
        resp = await self._send({"active_symbols": "brief"})
        symbols = resp.get("active_symbols", [])
        for s in symbols:
            code = s.get("underlying_symbol") or s.get("symbol")
            if code and s.get("pip_size") is not None:
                self._pip_sizes[code] = s["pip_size"]
        return symbols

    async def contracts_for(self, symbol: str, currency: str = "USD") -> dict:
        return await self._send({"contracts_for": symbol, "currency": currency})

    async def verify_even_odd_available(self, symbol: str,
                                        currency: str = "USD") -> tuple[bool, str]:
        """Section 1: refuse to trade a contract that is not actually
        offered. Verified against the live API rather than assumed, because
        an assumption here surfaces as a buy rejection mid-session."""
        try:
            resp = await self.contracts_for(symbol, currency)
        except DerivAPIError as exc:
            return False, f"contracts_for failed: {exc}"
        available = resp.get("contracts_for", {}).get("available", [])
        types = {c.get("contract_type") for c in available}
        missing = {"DIGITEVEN", "DIGITODD"} - types
        if missing:
            return False, f"{symbol}: missing contract types {sorted(missing)}"
        return True, f"{symbol}: DIGITEVEN and DIGITODD available"

    async def get_pip_size(self, symbol: str) -> int | float:
        if symbol not in self._pip_sizes:
            await self.active_symbols()
        if symbol not in self._pip_sizes:
            raise DerivAPIError("NoPipSize", f"could not determine pip_size for {symbol}")
        return self._pip_sizes[symbol]

    async def balance(self) -> float:
        resp = await self._send({"balance": 1})
        return float(resp.get("balance", {}).get("balance", 0.0))

    # ---- ticks -----------------------------------------------------------

    async def subscribe_ticks(self, symbol: str, maxsize: int = 1000) -> asyncio.Queue:
        if symbol not in self._tick_queues:
            self._tick_queues[symbol] = asyncio.Queue(maxsize=maxsize)
        await self._subscribe_ticks(symbol)
        return self._tick_queues[symbol]

    async def _subscribe_ticks(self, symbol: str) -> None:
        resp = await self._send({"ticks": symbol, "subscribe": 1})
        sub = (resp.get("subscription") or {}).get("id")
        if sub:
            self._subscription_ids[symbol] = sub

    async def tick_history(self, symbol: str, count: int = 5000) -> list[TickMessage]:
        """Historical ticks for cold start (Section 48)."""
        resp = await self._send({
            "ticks_history": symbol, "count": min(count, 5000),
            "end": "latest", "style": "ticks", "adjust_start_time": 1,
        })
        hist = resp.get("history", {})
        prices = hist.get("prices", [])
        times = hist.get("times", [])
        pip = self._pip_sizes.get(symbol, 2)
        return [TickMessage(symbol, float(t), p if isinstance(p, str) else repr(p), pip)
                for p, t in zip(prices, times)]

    # ---- trading ---------------------------------------------------------

    async def proposal(self, *, symbol: str, contract_type: str, amount: float,
                       currency: str, duration: int = 1,
                       duration_unit: str = "t") -> dict:
        """Counts against the shared rate budget (rule 6).

        The request field was renamed `symbol` -> `underlying_symbol` in the
        current Options API; sending `symbol` returns InputValidationFailed:
        Properties not allowed: symbol. DIGITEVEN/DIGITODD take no barrier,
        and Deriv REJECTS a barrier field for them rather than ignoring it,
        so none is sent.
        """
        resp = await self._send({
            "proposal": 1, "amount": round(float(amount), 2), "basis": "stake",
            "contract_type": contract_type, "currency": currency,
            "duration": duration, "duration_unit": duration_unit,
            "underlying_symbol": symbol,
        })
        return resp.get("proposal", {})

    async def buy(self, proposal_id: str, price: float, *, idempotency_key: str) -> dict:
        """Rule 4: EXACTLY ONE attempt. No retry, ever.

        On an ambiguous failure (timeout, disconnect) the outcome is
        genuinely unknown -- the buy may have been accepted. The caller must
        reconcile via portfolio(); retrying opens a second contract.
        """
        try:
            resp = await self._send({"buy": proposal_id,
                                     "price": round(float(price), 2)})
        except DerivAPIError as exc:
            if exc.code in ("Timeout", "Disconnected"):
                raise BuyAmbiguousError(
                    "BuyAmbiguous",
                    f"buy outcome unknown for {idempotency_key} ({exc.code}) -- "
                    f"reconcile via portfolio, DO NOT retry") from exc
            raise
        return resp.get("buy", {})

    async def portfolio(self) -> list[dict]:
        resp = await self._send({"portfolio": 1})
        return resp.get("portfolio", {}).get("contracts", [])

    async def proposal_open_contract(self, contract_id: int) -> dict:
        resp = await self._send({"proposal_open_contract": 1,
                                 "contract_id": contract_id})
        return resp.get("proposal_open_contract", {})

    # ---- settlement (rule 7) ---------------------------------------------

    async def subscribe_contract_updates(self, contract_id: int,
                                         maxsize: int = 50) -> asyncio.Queue:
        if contract_id not in self._contract_queues:
            self._contract_queues[contract_id] = asyncio.Queue(maxsize=maxsize)
        resp = await self._send({"proposal_open_contract": 1,
                                 "contract_id": contract_id, "subscribe": 1})
        # _route already handled this response via its req_id branch, but the
        # queue may have been created after that ran. Defensive second push
        # so the very first state is never lost.
        poc = resp.get("proposal_open_contract")
        if poc:
            try:
                self._contract_queues[contract_id].put_nowait(poc)
            except asyncio.QueueFull:
                pass
            sub_id = (resp.get("subscription") or {}).get("id") or poc.get("id")
            if sub_id:
                self._contract_subscription_ids[contract_id] = sub_id
        return self._contract_queues[contract_id]

    async def forget_contract_subscription(self, contract_id: int) -> None:
        sub_id = self._contract_subscription_ids.pop(contract_id, None)
        self._contract_queues.pop(contract_id, None)
        if sub_id:
            try:
                await self._send({"forget": sub_id})
            except DerivAPIError:
                pass   # already gone; not worth failing a settlement over

    async def wait_for_settlement(self, contract_id: int,
                                  timeout: float = 120.0) -> dict:
        """Rule 7: subscribe once, consume pushes. Bounded -- an unbounded
        wait on a contract that never reports would wedge the executor, and
        the empty dict returned on timeout is what the caller treats as an
        unknown outcome to reconcile.
        """
        queue = await self.subscribe_contract_updates(contract_id)
        deadline = time.monotonic() + timeout
        contract: dict = {}
        try:
            while time.monotonic() < deadline:
                remaining = deadline - time.monotonic()
                try:
                    contract = await asyncio.wait_for(
                        queue.get(), timeout=max(remaining, 0.1))
                except asyncio.TimeoutError:
                    break
                if contract.get("is_sold") or contract.get("is_expired"):
                    break
        finally:
            await self.forget_contract_subscription(contract_id)
        return contract

    # ---- health ----------------------------------------------------------

    @property
    def is_connected(self) -> bool:
        return (self._ws is not None and self._ws.state is WsState.OPEN
                and self._authorized)

    def seconds_since_last_message(self) -> float:
        return time.time() - self.last_message_at if self.last_message_at else float("inf")

    def rate_limit_status(self) -> dict:
        """Surfaced in the dashboard: if the engine is pacing constantly,
        the proposal cadence is the bottleneck, not the models."""
        return {
            "used_in_window": self.rate_limiter.used_in_window,
            "budget": self.rate_limiter.max_per_window,
            "waits": self.rate_limiter.total_waits,
            "total_wait_seconds": round(self.rate_limiter.total_wait_seconds, 2),
        }
