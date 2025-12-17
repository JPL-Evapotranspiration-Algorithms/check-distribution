import pytest
import numpy as np
from datetime import date
from check_distribution import check_distribution, BlankOutputError


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_array(self):
        """Test with empty array."""
        image = np.array([])
        # Empty array causes ZeroDivisionError in nan_proportion calculation
        with pytest.raises((BlankOutputError, ValueError, ZeroDivisionError)):
            check_distribution(image, "empty_array")

    def test_single_element_array(self, caplog):
        """Test with single element array."""
        image = np.array([42.0])
        check_distribution(image, "single_element")
        assert "single_element" in caplog.text

    def test_single_nan_element(self):
        """Test with single NaN element."""
        image = np.array([np.nan])
        with pytest.raises(BlankOutputError):
            check_distribution(image, "single_nan")

    def test_very_large_array(self, caplog):
        """Test with very large array."""
        image = np.random.rand(1000, 1000)
        check_distribution(image, "large_array")
        assert "large_array" in caplog.text

    def test_exactly_10_unique_values(self, caplog):
        """Test boundary case with exactly 10 unique values."""
        image = np.array([i % 10 for i in range(100)])
        check_distribution(image, "ten_values")
        assert "ten_values" in caplog.text
        # With exactly 10 unique values, should use the detailed path
        assert "min:" in caplog.text

    def test_exactly_9_unique_values(self, caplog):
        """Test boundary case with exactly 9 unique values."""
        image = np.array([i % 9 for i in range(100)])
        check_distribution(image, "nine_values")
        assert "nine_values" in caplog.text
        # With 9 unique values, should use the simple path
        assert "unique values" in caplog.text

    def test_infinity_values(self, caplog):
        """Test with infinity values."""
        image = np.array([1.0, 2.0, np.inf, 4.0, -np.inf])
        check_distribution(image, "infinity_values")
        assert "infinity_values" in caplog.text

    def test_very_small_values(self, caplog):
        """Test with very small positive values near zero."""
        image = np.array([1e-100, 1e-99, 1e-98, 1e-97, 1e-96])
        check_distribution(image, "tiny_values")
        assert "tiny_values" in caplog.text

    def test_very_large_positive_values(self, caplog):
        """Test with very large positive values."""
        image = np.array([1e100, 1e101, 1e102, 1e103, 1e104])
        check_distribution(image, "huge_values")
        assert "huge_values" in caplog.text

    def test_very_large_negative_values(self, caplog):
        """Test with very large negative values."""
        image = np.array([-1e100, -1e101, -1e102, -1e103, -1e104])
        check_distribution(image, "huge_negative_values")
        assert "huge_negative_values" in caplog.text

    def test_mixed_int_types(self, caplog):
        """Test with different integer dtypes."""
        for dtype in [np.int8, np.int16, np.int32, np.int64]:
            image = np.array([1, 2, 3, 4, 5], dtype=dtype)
            check_distribution(image, f"int_{dtype.__name__}")
            assert f"int_{dtype.__name__}" in caplog.text
            caplog.clear()

    def test_mixed_float_types(self, caplog):
        """Test with different float dtypes."""
        for dtype in [np.float32, np.float64]:
            image = np.array([1.5, 2.5, 3.5, 4.5, 5.5], dtype=dtype)
            check_distribution(image, f"float_{dtype.__name__}")
            assert f"float_{dtype.__name__}" in caplog.text
            caplog.clear()

    def test_boolean_array(self, caplog):
        """Test with boolean array."""
        image = np.array([True, False, True, False, True])
        check_distribution(image, "boolean_array")
        assert "boolean_array" in caplog.text

    def test_complex_numbers(self, caplog):
        """Test behavior with complex numbers (may not be fully supported)."""
        image = np.array([1+2j, 3+4j, 5+6j])
        # This may raise an error or handle it unexpectedly
        try:
            check_distribution(image, "complex_numbers")
        except (TypeError, ValueError):
            pass  # Expected for complex numbers

    def test_50_percent_nan(self, caplog):
        """Test exactly 50% NaN boundary."""
        image = np.array([1.0, 2.0, np.nan, np.nan])
        check_distribution(image, "fifty_percent_nan")
        assert "fifty_percent_nan" in caplog.text

    def test_51_percent_nan(self, caplog):
        """Test just over 50% NaN (should trigger yellow warning)."""
        image = np.array([1.0] * 49 + [np.nan] * 51)
        check_distribution(image, "majority_nan", allow_blank=True)
        assert "majority_nan" in caplog.text

    def test_nan_in_integer_context(self, caplog):
        """Test that integer arrays don't have NaN proportion calculated."""
        image = np.array([1, 2, 3, 4, 5], dtype=np.int32)
        check_distribution(image, "integer_no_nan")
        assert "integer_no_nan" in caplog.text

    def test_all_same_nonzero_value(self, caplog):
        """Test array with all same non-zero value."""
        image = np.full((50,), 42.0)
        check_distribution(image, "constant_42")
        assert "constant_42" in caplog.text
        assert "unique values" in caplog.text

    def test_variable_name_with_special_chars(self, caplog):
        """Test variable name with special characters."""
        image = np.random.rand(10, 10)
        check_distribution(image, "variable_with_特殊字符_and_émojis")
        assert "variable_with_特殊字符_and_émojis" in caplog.text

    def test_target_with_special_chars(self, caplog):
        """Test target location with special characters."""
        image = np.random.rand(10, 10)
        check_distribution(image, "test_var", target="Location_特殊_字符")
        assert "test_var" in caplog.text
        assert "Location_特殊_字符" in caplog.text

    def test_future_date(self, caplog):
        """Test with future date."""
        image = np.random.rand(10, 10)
        future_date = date(2030, 12, 31)
        check_distribution(image, "future_data", date_UTC=future_date)
        assert "future_data" in caplog.text
        assert "2030-12-31" in caplog.text

    def test_past_date(self, caplog):
        """Test with historical date."""
        image = np.random.rand(10, 10)
        past_date = date(1990, 1, 1)
        check_distribution(image, "historical_data", date_UTC=past_date)
        assert "historical_data" in caplog.text
        assert "1990-01-01" in caplog.text

    def test_leap_year_date(self, caplog):
        """Test with leap year date."""
        image = np.random.rand(10, 10)
        leap_date = date(2024, 2, 29)
        check_distribution(image, "leap_year_data", date_UTC=leap_date)
        assert "leap_year_data" in caplog.text
        assert "2024-02-29" in caplog.text

    def test_alternating_values(self, caplog):
        """Test with alternating pattern."""
        image = np.array([1, 2] * 50)
        check_distribution(image, "alternating")
        assert "alternating" in caplog.text
        assert "unique values" in caplog.text

    def test_ascending_sequence(self, caplog):
        """Test with ascending sequence."""
        image = np.arange(100, dtype=float)
        check_distribution(image, "ascending")
        assert "ascending" in caplog.text
        assert "min:" in caplog.text

    def test_descending_sequence(self, caplog):
        """Test with descending sequence."""
        image = np.arange(100, 0, -1, dtype=float)
        check_distribution(image, "descending")
        assert "descending" in caplog.text
        assert "min:" in caplog.text
