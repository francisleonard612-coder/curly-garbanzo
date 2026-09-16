"""
Spec Sections 7/13/40/43: statistical engine tests.

The critical property under test is ASYMMETRY OF ERROR. False negatives
here cost opportunity; false positives cost money, because a false positive
is the bot deciding a CSPRNG is predictable. So the tests below check both
directions, but the noise tests use many independent replications: a
randomness detector that fires on 1 run in 20 is useless when the bot
evaluates thousands of times a day.
"""
import math
import random

from app.statistics.distribution import (
    benjamini_hochberg,
    chi2_sf,
    chi_square_gof,
    g_test,
    hellinger_distance,
    jensen_shannon_divergence,
    kl_divergence,
    sidak_alpha,
    total_variation_distance,
    wilson_interval,
)
from app.statistics.information import (
    digit_transition_mi,
    mutual_information,
    normalized_digit_entropy,
    parity_entropy,
)
from app.statistics.randomness import RandomnessMonitor, runs_test


def uniform_digits(n, seed=0):
    rng = random.Random(seed)
    return [rng.randrange(10) for _ in range(n)]


class TestChiSquareDistribution:
    def test_known_survival_values(self):
        # Reference values from standard chi-square tables.
        assert abs(chi2_sf(3.84, 1) - 0.05) < 0.001
        assert abs(chi2_sf(16.919, 9) - 0.05) < 0.001
        assert abs(chi2_sf(21.666, 9) - 0.01) < 0.001
        assert abs(chi2_sf(0.0, 5) - 1.0) < 1e-12


class TestGoodnessOfFit:
    def test_uniform_data_is_not_flagged(self):
        counts = [100] * 10
        r = chi_square_gof(counts)
        assert r.statistic == 0.0
        assert r.p_value > 0.99
        assert r.total_variation_distance == 0.0

    def test_strongly_biased_data_is_flagged(self):
        counts = [300] + [77] * 9   # digit 0 heavily over-represented
        r = chi_square_gof(counts)
        assert r.p_value < 1e-10
        assert r.total_variation_distance > 0.1

    def test_g_test_agrees_with_chi_square_on_clear_cases(self):
        counts = [300] + [77] * 9
        assert g_test(counts).p_value < 1e-10
        assert g_test([100] * 10).p_value > 0.99

    def test_non_uniform_reference_is_respected(self):
        """Section 7: the reference must not be hard-coded to uniform."""
        counts = [200] + [88] * 9
        ref = [0.2] + [0.0888] * 9
        assert chi_square_gof(counts, ref).p_value > 0.05
        assert chi_square_gof(counts).p_value < 1e-5

    def test_small_sample_flagged_insufficient(self):
        assert chi_square_gof([2] * 10).sufficient_sample is False
        assert chi_square_gof([50] * 10).sufficient_sample is True


class TestDivergences:
    def test_identical_distributions_are_zero(self):
        p = [0.1] * 10
        assert abs(kl_divergence(p, p)) < 1e-12
        assert abs(jensen_shannon_divergence(p, p)) < 1e-12
        assert total_variation_distance(p, p) < 1e-12
        assert hellinger_distance(p, p) < 1e-12

    def test_jsd_is_symmetric_and_bounded(self):
        p = [0.5, 0.5] + [0.0] * 8
        q = [0.0] * 8 + [0.5, 0.5]
        a = jensen_shannon_divergence(p, q)
        b = jensen_shannon_divergence(q, p)
        assert abs(a - b) < 1e-12
        assert 0 <= a <= math.log(2) + 1e-9

    def test_tvd_bounded(self):
        p = [1.0] + [0.0] * 9
        q = [0.0] * 9 + [1.0]
        assert abs(total_variation_distance(p, q) - 1.0) < 1e-9


class TestIntervals:
    def test_wilson_contains_truth_and_narrows_with_n(self):
        narrow = wilson_interval(5000, 10000)
        wide = wilson_interval(50, 100)
        assert narrow.lower < 0.5 < narrow.upper
        assert narrow.width < wide.width

    def test_wilson_handles_extremes_without_degenerating(self):
        assert wilson_interval(0, 100).lower == 0.0
        assert wilson_interval(0, 100).upper > 0.0     # not a zero-width lie
        assert wilson_interval(100, 100).upper == 1.0
        assert wilson_interval(100, 100).lower < 1.0


class TestMultipleTesting:
    def test_bh_rejects_nothing_when_all_null(self):
        rng = random.Random(1)
        ps = [rng.random() for _ in range(100)]
        assert sum(benjamini_hochberg(ps, 0.05)) <= 5

    def test_bh_finds_strong_signals(self):
        ps = [1e-10] * 5 + [0.5] * 95
        keep = benjamini_hochberg(ps, 0.05)
        assert sum(keep) >= 5
        assert all(keep[:5])

    def test_sidak_is_stricter_for_more_tests(self):
        assert sidak_alpha(0.05, 20) < sidak_alpha(0.05, 1)
        assert abs(sidak_alpha(0.05, 1) - 0.05) < 1e-12


class TestRunsTest:
    def test_perfect_alternation_is_detected(self):
        """The case frequency tests are blind to: exactly 50% even, and
        perfectly predictable."""
        seq = [i % 2 for i in range(400)]
        r = runs_test(seq)
        assert r.p_value < 1e-10
        assert abs(sum(seq) / len(seq) - 0.5) < 0.01   # marginal rate is fair

    def test_long_blocks_detected(self):
        seq = ([0] * 20 + [1] * 20) * 10
        assert runs_test(seq).p_value < 1e-5

    def test_random_sequence_not_flagged(self):
        rng = random.Random(42)
        seq = [rng.randrange(2) for _ in range(2000)]
        assert runs_test(seq).p_value > 0.01


class TestMutualInformationBias:
    def test_raw_mi_is_biased_upward_on_independent_data(self):
        """The trap from the module docstring, demonstrated: independent
        data yields clearly positive raw MI."""
        digits = uniform_digits(1000, seed=3)
        mi = digit_transition_mi(digits)
        assert mi.raw > 0.01               # the illusion is real and large
        assert mi.raw > mi.expected_bias * 0.5
        assert mi.p_value > 0.01           # but not significant
        assert mi.corrected < mi.raw       # and correction shrinks it

    def test_bias_shrinks_as_sample_grows(self):
        small = digit_transition_mi(uniform_digits(500, seed=4))
        large = digit_transition_mi(uniform_digits(20000, seed=4))
        assert large.raw < small.raw

    def test_genuine_dependence_is_still_detected(self):
        rng = random.Random(5)
        digits = [rng.randrange(10)]
        for _ in range(5000):
            digits.append((digits[-1] + 1) % 10 if rng.random() < 0.8 else rng.randrange(10))
        mi = digit_transition_mi(digits)
        assert mi.p_value < 1e-10
        assert mi.corrected > 0.3

    def test_independent_pairs_are_not_significant(self):
        rng = random.Random(6)
        pairs = [(rng.randrange(2), rng.randrange(2)) for _ in range(5000)]
        assert mutual_information(pairs, 2, 2).p_value > 0.01


class TestEntropy:
    def test_uniform_window_is_near_max_entropy(self):
        assert normalized_digit_entropy(uniform_digits(20000, seed=7)) > 0.99

    def test_constant_stream_is_zero_entropy(self):
        assert normalized_digit_entropy([4] * 500) == 0.0

    def test_fair_parity_is_one_bit(self):
        h = parity_entropy(uniform_digits(20000, seed=8), base=2)
        assert abs(h - 1.0) < 0.01


class TestRandomnessMonitorVeto:
    """The most important tests in the repo: the veto must hold on random
    data across many independent histories, not just on average."""

    def test_csprng_like_stream_is_never_tradeable(self):
        monitor = RandomnessMonitor(alpha=0.01, min_samples=2000)
        for seed in range(25):
            digits = uniform_digits(5000, seed=seed)
            v = monitor.evaluate(digits)
            assert v.tradeable is False, f"seed {seed}: falsely declared tradeable\n{v.summary()}"

    def test_insufficient_sample_is_not_tradeable(self):
        monitor = RandomnessMonitor(min_samples=2000)
        v = monitor.evaluate(uniform_digits(100, seed=1))
        assert v.tradeable is False
        assert v.sufficient_sample is False
        assert "insufficient sample" in v.summary()

    def test_significant_but_economically_irrelevant_is_not_tradeable(self):
        """Section 40's core lesson: a real but tiny bias still loses money.
        51% even is genuinely non-random at this n, and still below the
        ~51.28% break-even."""
        rng = random.Random(11)
        digits = []
        for _ in range(200000):
            if rng.random() < 0.51:
                digits.append(rng.choice([0, 2, 4, 6, 8]))
            else:
                digits.append(rng.choice([1, 3, 5, 7, 9]))
        v = RandomnessMonitor(alpha=0.01, min_samples=2000).evaluate(digits)
        assert v.any_significant is True          # detects the bias
        assert v.economically_relevant is False   # but refuses to trade it
        assert v.tradeable is False

    def test_large_genuine_edge_is_allowed_through(self):
        """The detector must not be SO strict that a real, large,
        economically decisive edge is missed -- otherwise it is just an
        expensive `return False`."""
        rng = random.Random(12)
        digits = []
        for _ in range(50000):
            if rng.random() < 0.60:
                digits.append(rng.choice([0, 2, 4, 6, 8]))
            else:
                digits.append(rng.choice([1, 3, 5, 7, 9]))
        v = RandomnessMonitor(alpha=0.01, min_samples=2000).evaluate(digits)
        assert v.any_significant is True
        assert v.economically_relevant is True
        assert v.tradeable is True

    def test_perfectly_alternating_parity_is_caught_as_dependence(self):
        digits = []
        for i in range(5000):
            digits.append(random.Random(i).choice([0, 2, 4, 6, 8] if i % 2 == 0
                                                  else [1, 3, 5, 7, 9]))
        v = RandomnessMonitor(alpha=0.01, min_samples=2000).evaluate(digits)
        assert v.any_significant is True
