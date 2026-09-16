"""
DECIMAL-SAFE DIGIT EXTRACTION (spec Section 5).

Everything downstream -- every distribution, every model, every trade --
rests on this one function being right. If digit extraction is subtly
wrong, every statistical test in this repo faithfully measures an artifact
of float representation instead of the market, and does so with complete
confidence. That failure is silent: the numbers look fine.

WHY `int(str(price)[-1])` IS WRONG, concretely:

    >>> 0.1 + 0.2
    0.30000000000000004
    >>> str(1234.30)
    '1234.3'          # trailing zero GONE -- last digit reads as 3, not 0
    >>> repr(2.675)
    '2.675'
    >>> f"{2.675:.2f}"
    '2.67'            # banker's/representation rounding, not 2.68

Three distinct failure modes there:
  1. Binary floats cannot represent most decimal fractions exactly.
  2. Python's str() strips trailing zeros, so a price ending in 0 silently
     reports the digit before it. Since a stripped trailing zero is always
     an EVEN digit being replaced by an arbitrary one, this biases the
     parity series directly -- the exact quantity this bot trades.
  3. Rounding a float that is already slightly off can round the wrong way.

Deriv sends quotes as JSON numbers, but ALSO publishes the instrument's
decimal precision (`pip_size` on the tick/active_symbols payload). The
correct procedure is therefore:

  - take the quote's ORIGINAL STRING form wherever the transport preserves
    it (we ask the JSON parser to keep it -- see parse_quote_raw below),
  - construct a Decimal from that string (never from the float),
  - quantize to the instrument's known pip_size decimal places,
  - read the final character of the fixed-point representation.

Only when the raw string is genuinely unavailable do we fall back to
Decimal(repr(float)), which is the closest shortest-roundtrip decimal --
still far safer than str-slicing, but recorded as a lower-confidence path
so diagnostics can count how often it happens.

SPEC SECTION 59 (fail closed): extraction that cannot be performed
confidently raises DigitExtractionError rather than guessing. A guessed
digit is worse than no digit -- it enters the distribution as real data.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP


class DigitExtractionError(ValueError):
    """Raised when a digit cannot be extracted with confidence.

    Callers must treat this as a fail-closed condition (spec Section 59):
    drop the tick and, if it recurs, halt trading. Never substitute a
    default digit -- that silently poisons every distribution downstream.
    """


@dataclass(frozen=True)
class ExtractedDigit:
    raw_quote: str          # exactly what the wire carried, unparsed
    normalized_quote: str   # fixed-point, padded to `precision` decimals
    precision: int          # decimal places implied by the instrument pip_size
    digit: int              # 0-9
    parity: int             # 0 = EVEN, 1 = ODD
    exact: bool             # True if derived from the raw string (preferred path)

    @property
    def is_even(self) -> bool:
        return self.parity == 0


def precision_from_pip_size(pip_size) -> int:
    """Decimal places implied by Deriv's pip_size.

    Deriv expresses pip_size either as an integer count of decimal places
    (e.g. 2) or as the pip value itself (e.g. 0.01). Both appear in the
    wild across active_symbols and tick payloads, so both are handled --
    guessing wrong here shifts every digit by a place.
    """
    if pip_size is None:
        raise DigitExtractionError("pip_size is required to extract digits safely")
    if isinstance(pip_size, int) and not isinstance(pip_size, bool):
        # Lower bound is 1, not 0: as a pip VALUE, 0 is invalid outright; as
        # a decimal-place COUNT, 0 would mean integer-quoted prices, where a
        # "last decimal digit" doesn't exist and digit contracts aren't
        # offered. Either way a 0 here means a missing or malformed field,
        # and defaulting it would silently extract the wrong digit position
        # for every tick.
        if 1 <= pip_size <= 12:
            return pip_size
        raise DigitExtractionError(f"implausible pip_size as decimal count: {pip_size}")
    try:
        d = Decimal(str(pip_size))
    except InvalidOperation as exc:
        raise DigitExtractionError(f"uninterpretable pip_size: {pip_size!r}") from exc
    if d <= 0:
        raise DigitExtractionError(f"non-positive pip_size: {pip_size!r}")
    if d >= 1:
        # e.g. 2 delivered as "2" or 2.0 -> a decimal-place count
        if d == d.to_integral_value() and d <= 12:
            return int(d)
        raise DigitExtractionError(f"implausible pip_size: {pip_size!r}")
    exponent = -d.as_tuple().exponent
    return int(exponent)


def extract(raw_quote, pip_size, *, allow_float_fallback: bool = True) -> ExtractedDigit:
    """Extracts the final decimal digit of `raw_quote` at the instrument's
    own precision.

    `raw_quote` should be the ORIGINAL string from the wire when available
    (see app/api/deriv_client.py, which preserves it). A float is accepted
    but flagged `exact=False`, because by then the decimal information may
    already have been destroyed by the JSON float parse.
    """
    precision = precision_from_pip_size(pip_size)

    if isinstance(raw_quote, Decimal):
        dec, exact, raw_str = raw_quote, True, str(raw_quote)
    elif isinstance(raw_quote, str):
        raw_str = raw_quote.strip()
        if not raw_str:
            raise DigitExtractionError("empty quote string")
        try:
            dec, exact = Decimal(raw_str), True
        except InvalidOperation as exc:
            raise DigitExtractionError(f"unparseable quote string: {raw_quote!r}") from exc
    elif isinstance(raw_quote, int) and not isinstance(raw_quote, bool):
        dec, exact, raw_str = Decimal(raw_quote), True, str(raw_quote)
    elif isinstance(raw_quote, float):
        if not allow_float_fallback:
            raise DigitExtractionError(
                "float quote rejected: raw string required for exact extraction")
        if raw_quote != raw_quote or raw_quote in (float("inf"), float("-inf")):
            raise DigitExtractionError(f"non-finite quote: {raw_quote!r}")
        # repr() gives the shortest string that round-trips to this exact
        # float -- the best available reconstruction once the string is gone.
        dec, exact, raw_str = Decimal(repr(raw_quote)), False, repr(raw_quote)
    else:
        raise DigitExtractionError(f"unsupported quote type: {type(raw_quote).__name__}")

    if not dec.is_finite():
        raise DigitExtractionError(f"non-finite quote: {raw_quote!r}")

    # Quantize to the instrument's precision. ROUND_HALF_UP is the ordinary
    # arithmetic convention; Python's default ROUND_HALF_EVEN would map
    # x.xx5 to an even digit more often than an odd one, which is a direct
    # parity bias in a bot that trades parity.
    quantum = Decimal(1).scaleb(-precision)
    try:
        q = dec.quantize(quantum, rounding=ROUND_HALF_UP)
    except InvalidOperation as exc:
        raise DigitExtractionError(f"cannot quantize {raw_quote!r} to {precision} dp") from exc

    # Fixed-point text, trailing zeros preserved. This is the whole point:
    # format with an explicit precision rather than str(), so a price ending
    # in 0 reports 0.
    normalized = f"{q:.{precision}f}"
    last_char = normalized[-1]
    if not last_char.isdigit():
        raise DigitExtractionError(
            f"normalized quote {normalized!r} does not end in a digit "
            f"(precision={precision})")
    digit = int(last_char)
    return ExtractedDigit(
        raw_quote=raw_str,
        normalized_quote=normalized,
        precision=precision,
        digit=digit,
        parity=digit % 2,
        exact=exact,
    )


def parity_of(digit: int) -> int:
    if not isinstance(digit, int) or not 0 <= digit <= 9:
        raise DigitExtractionError(f"not a decimal digit: {digit!r}")
    return digit % 2


EVEN_DIGITS = (0, 2, 4, 6, 8)
ODD_DIGITS = (1, 3, 5, 7, 9)
