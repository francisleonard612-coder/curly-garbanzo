"""
Spec Section 5/58: digit extraction tests.

These are the highest-stakes tests in the repo. Every distribution, every
statistical test and every trade decision is computed on the output of
extract(); a subtle bug here produces a bot that confidently measures
float-representation artifacts and calls them market structure.
"""
import pytest

from app.digits.extraction import (
    DigitExtractionError,
    extract,
    precision_from_pip_size,
)


class TestPrecision:
    def test_pip_size_as_decimal_value(self):
        assert precision_from_pip_size(0.01) == 2
        assert precision_from_pip_size("0.001") == 3
        assert precision_from_pip_size(0.00001) == 5

    def test_pip_size_as_decimal_place_count(self):
        assert precision_from_pip_size(2) == 2
        assert precision_from_pip_size(4) == 4

    def test_rejects_garbage(self):
        for bad in (None, 0, -1, "abc", 99):
            with pytest.raises(DigitExtractionError):
                precision_from_pip_size(bad)


class TestTrailingZero:
    """The single most dangerous case, and the reason str-slicing is banned.

    str(1234.30) == '1234.3'. Slicing the last character yields 3 (ODD)
    for a price whose true final digit is 0 (EVEN). Because a stripped
    trailing zero ALWAYS replaces an even digit, this is not random noise --
    it is a systematic parity bias in the exact series being traded.
    """

    def test_trailing_zero_preserved_from_string(self):
        r = extract("1234.30", 0.01)
        assert r.digit == 0
        assert r.parity == 0
        assert r.normalized_quote == "1234.30"
        assert r.exact is True

    def test_naive_str_slicing_would_have_been_wrong(self):
        assert str(1234.30)[-1] == "3"          # the bug
        assert extract(1234.30, 0.01).digit == 0  # the fix

    def test_trailing_zero_from_float_fallback(self):
        r = extract(1234.30, 0.01)
        assert r.digit == 0
        assert r.exact is False  # flagged as the lower-confidence path


class TestFloatRepresentation:
    def test_classic_float_error_does_not_leak_into_digit(self):
        # 0.1 + 0.2 == 0.30000000000000004
        r = extract(0.1 + 0.2, 0.01)
        assert r.normalized_quote == "0.30"
        assert r.digit == 0

    def test_string_path_is_exact_where_float_path_is_not(self):
        assert extract("0.30", 0.01).exact is True
        assert extract(0.30, 0.01).exact is False
        assert extract("0.30", 0.01).digit == extract(0.30, 0.01).digit

    def test_can_require_exact_extraction(self):
        with pytest.raises(DigitExtractionError):
            extract(1234.30, 0.01, allow_float_fallback=False)


class TestRounding:
    def test_half_up_not_bankers_rounding(self):
        """Python's default ROUND_HALF_EVEN maps .xx5 toward an even digit
        more often than odd -- a parity bias in a parity-trading bot."""
        assert extract("2.675", 0.01).digit == 8   # 2.68, not 2.67
        assert extract("2.685", 0.01).digit == 9   # 2.69, not 2.68

    def test_padding_shorter_prices(self):
        r = extract("100.5", 0.01)
        assert r.normalized_quote == "100.50"
        assert r.digit == 0

    def test_integer_quote(self):
        r = extract(100, 0.01)
        assert r.normalized_quote == "100.00"
        assert r.digit == 0


class TestParity:
    @pytest.mark.parametrize("quote,expected_digit,expected_parity", [
        ("1234.50", 0, 0), ("1234.51", 1, 1), ("1234.52", 2, 0),
        ("1234.53", 3, 1), ("1234.54", 4, 0), ("1234.55", 5, 1),
        ("1234.56", 6, 0), ("1234.57", 7, 1), ("1234.58", 8, 0),
        ("1234.59", 9, 1),
    ])
    def test_all_ten_digits(self, quote, expected_digit, expected_parity):
        r = extract(quote, 0.01)
        assert r.digit == expected_digit
        assert r.parity == expected_parity
        assert r.is_even == (expected_parity == 0)


class TestPrecisionSensitivity:
    def test_same_quote_different_precision_gives_different_digit(self):
        """A wrong pip_size shifts every digit by a place -- which is why
        pip_size is required rather than inferred from the quote text."""
        assert extract("1234.567", 0.001).digit == 7
        assert extract("1234.567", 0.01).digit == 7  # 1234.57 -> 7
        assert extract("1234.561", 0.01).digit == 6  # 1234.56 -> 6

    def test_five_decimal_instrument(self):
        r = extract("0.66742", 0.00001)
        assert r.digit == 2
        assert r.normalized_quote == "0.66742"


class TestFailClosed:
    @pytest.mark.parametrize("bad", ["", "   ", "abc", None, [], {}])
    def test_bad_input_raises_rather_than_guessing(self, bad):
        with pytest.raises(DigitExtractionError):
            extract(bad, 0.01)

    def test_non_finite_rejected(self):
        for bad in (float("nan"), float("inf"), float("-inf")):
            with pytest.raises(DigitExtractionError):
                extract(bad, 0.01)
