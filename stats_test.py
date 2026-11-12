import pytest
import math
from stats_in_python import (
    calculate_average,
    get_max_and_min,
    calculate_variance,
    calculate_std_dev,
    calculate_median,
)

# ─────────────────────────────────────────────
# Shared fixtures
# ─────────────────────────────────────────────

@pytest.fixture
def simple_list():
    return [1, 2, 3, 4, 5]

@pytest.fixture
def float_list():
    return [1.5, 2.5, 3.5, 4.5]

@pytest.fixture
def negative_list():
    return [-3, -1, 0, 2, 4]

@pytest.fixture
def single_element():
    return [42.0]

@pytest.fixture
def repeated_values():
    return [7, 7, 7, 7]

@pytest.fixture
def even_length_list():
    return [1, 3, 5, 7]

@pytest.fixture
def odd_length_list():
    return [10, 20, 30]

@pytest.fixture
def unsorted_list():
    return [5, 1, 4, 2, 3]


# ─────────────────────────────────────────────
# calculate_average
# ─────────────────────────────────────────────

class TestCalculateAverage:

    def test_average_simple(self, simple_list):
        """Average of [1,2,3,4,5] should be 3.0"""
        assert calculate_average(simple_list) == pytest.approx(3.0)

    def test_average_floats(self, float_list):
        """Average of [1.5, 2.5, 3.5, 4.5] should be 3.0"""
        assert calculate_average(float_list) == pytest.approx(3.0)

    def test_average_negatives(self, negative_list):
        """Average of [-3,-1,0,2,4] should be 0.4"""
        assert calculate_average(negative_list) == pytest.approx(0.4)

    def test_average_single_element(self, single_element):
        """Average of a one-element list equals that element"""
        assert calculate_average(single_element) == pytest.approx(42.0)

    def test_average_repeated_values(self, repeated_values):
        """Average of identical values equals that value"""
        assert calculate_average(repeated_values) == pytest.approx(7.0)

    def test_average_returns_float(self, simple_list):
        """Return type should be float"""
        assert isinstance(calculate_average(simple_list), float)


# ─────────────────────────────────────────────
# get_max_and_min
# ─────────────────────────────────────────────

class TestGetMaxAndMin:

    def test_max_min_simple(self, simple_list):
        """Max=5, min=1 for [1,2,3,4,5]"""
        result = get_max_and_min(simple_list)
        assert result == (5, 1)

    def test_max_min_negatives(self, negative_list):
        """Max=4, min=-3 for [-3,-1,0,2,4]"""
        result = get_max_and_min(negative_list)
        assert result == (4, -3)

    def test_max_min_floats(self, float_list):
        """Max=4.5, min=1.5 for float list"""
        result = get_max_and_min(float_list)
        assert result == pytest.approx((4.5, 1.5))

    def test_max_min_single_element(self, single_element):
        """Max and min are the same for a single-element list"""
        result = get_max_and_min(single_element)
        assert result == pytest.approx((42.0, 42.0))

    def test_max_min_repeated_values(self, repeated_values):
        """Max and min are equal when all values are identical"""
        result = get_max_and_min(repeated_values)
        assert result == (7, 7)

    def test_returns_tuple(self, simple_list):
        """Return type must be a tuple"""
        assert isinstance(get_max_and_min(simple_list), tuple)

    def test_max_is_first(self, simple_list):
        """First element of tuple must be the maximum"""
        result = get_max_and_min(simple_list)
        assert result[0] >= result[1], "Maximum should be the first element"

    def test_min_is_second(self, simple_list):
        """Second element of tuple must be the minimum"""
        result = get_max_and_min(simple_list)
        assert result[1] <= result[0], "Minimum should be the second element"


# ─────────────────────────────────────────────
# calculate_variance
# ─────────────────────────────────────────────

class TestCalculateVariance:

    def test_variance_simple(self, simple_list):
        """Variance of [1,2,3,4,5] = 2.5 (sample variance, n-1)"""
        assert calculate_variance(simple_list) == pytest.approx(2.5)

    def test_variance_repeated_values(self, repeated_values):
        """Variance of identical values is 0"""
        assert calculate_variance(repeated_values) == pytest.approx(0.0)

    def test_variance_single_element(self, single_element):
        """Variance of a single-element list is 0"""
        assert calculate_variance(single_element) == pytest.approx(0.0)

    def test_variance_negatives(self, negative_list):
        """Variance of [-3,-1,0,2,4] = 7.8 (sample variance, n-1: mean=0.4)"""
        assert calculate_variance(negative_list) == pytest.approx(7.8)

    def test_variance_non_negative(self, float_list):
        """Variance must always be >= 0"""
        assert calculate_variance(float_list) >= 0

    def test_variance_returns_float(self, simple_list):
        """Return type should be float"""
        assert isinstance(calculate_variance(simple_list), float)


# ─────────────────────────────────────────────
# calculate_std_dev
# ─────────────────────────────────────────────

class TestCalculateStdDev:

    def test_std_dev_simple(self, simple_list):
        """Std dev of [1,2,3,4,5] = sqrt(2.5) ≈ 1.5811 (sample std dev, n-1)"""
        assert calculate_std_dev(simple_list) == pytest.approx(math.sqrt(2.5))

    def test_std_dev_repeated_values(self, repeated_values):
        """Std dev of identical values is 0"""
        assert calculate_std_dev(repeated_values) == pytest.approx(0.0)

    def test_std_dev_single_element(self, single_element):
        """Std dev of a single-element list is 0"""
        assert calculate_std_dev(single_element) == pytest.approx(0.0)

    def test_std_dev_equals_sqrt_variance(self, float_list):
        """Std dev must equal the square root of the variance"""
        assert calculate_std_dev(float_list) == pytest.approx(
            math.sqrt(calculate_variance(float_list))
        )

    def test_std_dev_non_negative(self, negative_list):
        """Std dev must always be >= 0"""
        assert calculate_std_dev(negative_list) >= 0

    def test_std_dev_returns_float(self, simple_list):
        """Return type should be float"""
        assert isinstance(calculate_std_dev(simple_list), float)


# ─────────────────────────────────────────────
# calculate_median
# ─────────────────────────────────────────────

class TestCalculateMedian:

    def test_median_odd_length(self, odd_length_list):
        """Median of [10,20,30] = 20.0 (middle element)"""
        assert calculate_median(odd_length_list) == pytest.approx(20.0)

    def test_median_even_length(self, even_length_list):
        """Median of [1,3,5,7] = 4.0 (average of two middle elements)"""
        assert calculate_median(even_length_list) == pytest.approx(4.0)

    def test_median_unsorted_input(self, unsorted_list):
        """Median should sort the list first — [5,1,4,2,3] → median = 3.0"""
        assert calculate_median(unsorted_list) == pytest.approx(3.0)

    def test_median_single_element(self, single_element):
        """Median of a single-element list equals that element"""
        assert calculate_median(single_element) == pytest.approx(42.0)

    def test_median_repeated_values(self, repeated_values):
        """Median of identical values equals that value"""
        assert calculate_median(repeated_values) == pytest.approx(7.0)

    def test_median_negatives(self, negative_list):
        """Median of [-3,-1,0,2,4] = 0.0 (middle element after sorting)"""
        assert calculate_median(negative_list) == pytest.approx(0.0)

    def test_median_returns_float(self, simple_list):
        """Return type should be float"""
        assert isinstance(calculate_median(simple_list), float)