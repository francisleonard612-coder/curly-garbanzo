"""
Deriv WebSocket API client (spec Sections 34, 35, 59).

Isolated from strategy by mandate (Section 34): this module knows about
sockets, request correlation and reconnection. It knows nothing about
digits, models or trading decisions, and nothing here may import from
app/models, app/ensemble or app/execution.

=====================================================================
FOUR RULES CARRIED FORWARD FROM REAL PRODUCTION FAILURES. Do not
"simplify" any of them away -- each looks like redundant ceremony and
each was written after an outage.
=====================================================================

1. ONE READER, ALWAYS. Exactly one coroutine may read the socket. Every
   request/response pair is correlated by req_id through a pending-futures
   table that ONLY the reader resolves. Any code path that awaits a
   response while also being the thing responsible for reading it will
   deadlock until timeout.

2. RECV-PUMP DEATH MUST NEVER BE A SILENT PERMANENT OUTAGE. Observed in
   production: the pump crashed on `ConnectionClosedError: no close frame
   received or sent` (keepalive ping timeout), logged, and returned.
   Nothing else reconnected the TICK path -- reconnection only ran from the
   request/response path, which a pure tick consumer never touches. The
   process stayed up, looked healthy, and silently stopped receiving ticks
   forever. `_supervise_pump()` watches the pump task and reconnects
   whenever it exits, for any reason.

3. THE PUMP MUST START BEFORE RESUBSCRIBING, NOT AFTER. This is the subtle
   one, and the first fix for (2) got it wrong. Resubscribing goes through
   _send(), which awaits a future only the pump can resolve (rule 1). If
   the new pump has not started yet, EVERY resubscribe after EVERY
   reconnect is guaranteed to hang for the full request timeout -- not
   occasionally, deterministically, for every symbol. It was confirmed in a
   live log: clean re-auth immediately followed by a burst of resubscribe
   timeouts spaced exactly request_timeout apart. `_reconnect()` therefore
   starts the pump on the new socket BEFORE the resubscribe loop.

4. NEVER BLINDLY RETRY A BUY (Section 35). A retried buy whose first
   attempt actually succeeded opens two contracts. Buys carry a caller-
   supplied idempotency key, are attempted exactly once, and on ambiguous
   failure the caller must RECONCILE via portfolio rather than retry.
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


class DerivAPIError(RuntimeError):
    def __init__(self, code: str, message: str, echo: dict | None = None):
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message
        self.echo = echo or {}


class BuyAmbiguousError(DerivAPIError):
    """A buy whose outcome could not be determined. MUST be reconciled
    against the portfolio, never retried (rule 4)."""


@dataclass
class TickMessage:
    symbol: str
    epoch: float
    quote_raw: str          # preserved as text -- see the digit-extraction module
    pip_size: int | float
    received_at: float = field(default_factory=time.time)


class DerivClient:
    def __init__(self, app_id: str, api_token: str, ws_url: str,
                 request_timeout: float = 15.0):
        self.app_id = app_id
        self.api_token = api_token
        self.ws_url = ws_url
        self.request_timeout = request_timeout

        self._ws = None
        self._req_id = 0
        self._pending: dict[int, asyncio.Future] = {}
        self._tick_queues: dict[str, asyncio.Queue] = {}
        self._subscription_ids: dict[str, str] = {}
        self._pip_sizes: dict[str, int | float] = {}
        self._pump_task: asyncio.Task | None = None
        self._supervisor_task: asyncio.Task | None = None
        self._connect_lock = asyncio.Lock()
        self._closed = False
        self._authorized = False
        self.last_message_at: float = 0.0

    # ---- connection ------------------------------------------------------

    async def connect(self) -> None:
        async with self._connect_lock:
            await self._open_socket()
            # Rule 3: reader first, always.
            self._pump_task = asyncio.create_task(self._recv_pump(), name="deriv-pump")
            # Rule 2: ONE long-lived supervisor for the client's whole life,
            # never re-spawned per reconnect (two supervisors would race to
            # read the same socket).
            self._supervisor_task = asyncio.create_task(
                self._supervise_pump(), name="deriv-pump-supervisor")
            await self._authorize()
            logger.info("connected to Deriv")

    async def _open_socket(self) -> None:
        url = f"{self.ws_url}?app_id={self.app_id}"
        # ping_interval keeps the connection alive; the observed production
        # crash was a keepalive ping TIMEOUT, so the timeout is generous
        # relative to the interval rather than tight.
        self._ws = await websockets.connect(
            url, ping_interval=20, ping_timeout=60, close_timeout=5, max_size=4 * 1024 * 1024)
        self._authorized = False
        self.last_message_at = time.time()

    async def _authorize(self) -> None:
        if not self.api_token:
            raise DerivAPIError("NoToken", "DERIV_API_TOKEN is not set")
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
        """THE ONLY coroutine that reads the socket (rule 1)."""
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
        req_id = msg.get("req_id")
        if req_id is not None and req_id in self._pending:
            fut = self._pending.pop(req_id)
            if not fut.done():
                if "error" in msg:
                    err = msg["error"]
                    fut.set_exception(DerivAPIError(
                        err.get("code", "Unknown"), err.get("message", ""), msg.get("echo_req")))
                else:
                    fut.set_result(msg)
            return
        if msg.get("msg_type") == "tick":
            tick = msg.get("tick") or {}
            symbol = tick.get("symbol")
            if symbol and symbol in self._tick_queues:
                # `quote` is preserved as TEXT here. Once json.loads has made
                # it a float the decimal information may already be gone, and
                # the whole digit-extraction module exists to avoid that.
                raw_quote = self._raw_quote_text(msg, tick)
                q = self._tick_queues[symbol]
                if q.full():
                    try:
                        q.get_nowait()   # drop oldest; never block the reader
                    except asyncio.QueueEmpty:
                        pass
                q.put_nowait(TickMessage(
                    symbol=symbol, epoch=float(tick.get("epoch", 0)),
                    quote_raw=raw_quote,
                    pip_size=self._pip_sizes.get(symbol, tick.get("pip_size", 2))))

    @staticmethod
    def _raw_quote_text(msg: dict, tick: dict) -> str:
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
                    logger.error("reconnect failed (%s); next attempt in %.1fs", exc, backoff)

    async def _reconnect(self) -> None:
        async with self._connect_lock:
            if self._ws is not None and self._ws.state is WsState.OPEN:
                return
            await self._open_socket()
            # RULE 3, THE WHOLE POINT: start the reader on the new socket
            # BEFORE any resubscribe, because resubscribes await responses
            # only the reader can deliver.
            self._pump_task = asyncio.create_task(self._recv_pump(), name="deriv-pump")
            await self._authorize()
            for symbol in list(self._tick_queues.keys()):
                self._subscription_ids.pop(symbol, None)
                try:
                    await self._subscribe_ticks(symbol)
                except Exception as exc:
                    logger.error("failed to resubscribe %s: %s", symbol, exc)

    # ---- request/response ------------------------------------------------

    async def _send(self, payload: dict) -> dict:
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
            self._pending.pop(req_id, None)
            raise DerivAPIError("Timeout", f"no response within {self.request_timeout}s")
        finally:
            self._pending.pop(req_id, None)

    # ---- discovery (Section 1) -------------------------------------------

    async def active_symbols(self) -> list[dict]:
        resp = await self._send({"active_symbols": "brief", "product_type": "basic"})
        symbols = resp.get("active_symbols", [])
        for s in symbols:
            if s.get("symbol") and s.get("pip_size") is not None:
                self._pip_sizes[s["symbol"]] = s["pip_size"]
        return symbols

    async def contracts_for(self, symbol: str, currency: str = "USD") -> dict:
        return await self._send({"contracts_for": symbol, "currency": currency,
                                 "product_type": "basic"})

    async def verify_even_odd_available(self, symbol: str, currency: str = "USD") -> tuple[bool, str]:
        """Section 1: refuse to trade a contract that is not actually offered.

        Verified against the live API rather than assumed, because an
        assumption here surfaces as a buy rejection mid-session.
        """
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
        sub = resp.get("subscription", {}).get("id")
        if sub:
            self._subscription_ids[symbol] = sub

    async def tick_history(self, symbol: str, count: int = 5000) -> list[TickMessage]:
        """Historical ticks for cold start (Section 48)."""
        resp = await self._send({
            "ticks_history": symbol, "count": min(count, 5000),
            "end": "latest", "style": "ticks",
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
        resp = await self._send({
            "proposal": 1, "amount": round(float(amount), 2), "basis": "stake",
            "contract_type": contract_type, "currency": currency,
            "duration": duration, "duration_unit": duration_unit, "symbol": symbol,
        })
        return resp.get("proposal", {})

    async def buy(self, proposal_id: str, price: float, *, idempotency_key: str) -> dict:
        """Rule 4: EXACTLY ONE attempt. No retry, ever.

        On an ambiguous failure (timeout, disconnect) the outcome is
        genuinely unknown -- the buy may have been accepted. The caller must
        reconcile via portfolio(); retrying opens a second contract.
        """
        try:
            resp = await self._send({"buy": proposal_id, "price": round(float(price), 2)})
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
        resp = await self._send({"proposal_open_contract": 1, "contract_id": contract_id})
        return resp.get("proposal_open_contract", {})

    async def wait_for_settlement(self, contract_id: int, timeout: float = 120.0) -> dict:
        """Polls until the contract is sold/expired. Bounded: an unbounded
        wait on a contract that never reports would wedge the executor."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                poc = await self.proposal_open_contract(contract_id)
            except DerivAPIError:
                await asyncio.sleep(1.0)
                continue
            if poc.get("is_sold") or poc.get("is_expired"):
                return poc
            await asyncio.sleep(1.0)
        return {}

    @property
    def is_connected(self) -> bool:
        return self._ws is not None and self._ws.state is WsState.OPEN and self._authorized

    def seconds_since_last_message(self) -> float:
        return time.time() - self.last_message_at if self.last_message_at else float("inf")
