import pytest
import numpy as np
from check_distribution import check_distribution


class TestCoverageCompletion:
    """Tests to achieve complete code coverage for uncovered branches."""

    def test_maximum_less_than_or_equal_zero(self, caplog):
        """Test the red maximum path when max <= 0."""
        # Create array with many unique values (>= 10) where max is 0 or negative
        image = np.linspace(-100, 0, 20)  # 20 unique values from -100 to 0
        check_distribution(image, "negative_to_zero")
        assert "negative_to_zero" in caplog.text
        assert "max:" in caplog.text

    def test_100_percent_nan_with_allow_blank(self, caplog):
        """Test 100% NaN with allow_blank=True (attempting line 59).
        
        Note: Line 59 (elif nan_proportion == 1) is actually dead code because:
        - If nan_proportion == 1, then nan_proportion > 0.5 is also True
        - So it will always take the first branch (line 57), never line 59
        
        This test documents this issue but cannot actually cover line 59.
        """
        # Create array with >= 10 potential unique values but all NaN
        image = np.full(100, np.nan)
        check_distribution(image, "all_nan_allowed", allow_blank=True)
        assert "all_nan_allowed" in caplog.text
        # Should show nan percentage (takes line 57, not line 59)
        assert "nan:" in caplog.text

    def test_all_zeros_with_many_unique_unreachable(self, caplog):
        """Document that 'all zeros' path (lines 77-78) is logically unreachable.
        
        The condition requires:
        1. len(unique) >= 10 (to enter the else branch)
        2. np.all(image == 0) (all values are zero)
        
        These conditions are mutually exclusive because if all values are 0,
        then unique will be [0.] (only 1 unique value), not >= 10 unique values.
        
        This test documents this unreachable code path.
        """
        # Attempting to trigger: would need >= 10 unique values AND all zeros
        # This is impossible, so we just document it
        pass

    def test_very_small_negative_to_zero_range(self, caplog):
        """Test with values ranging from negative to exactly zero."""
        image = np.array([-0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.0, 0.0, 0.0, 0.0,
                         -0.15, -0.25, -0.35, -0.45])
        check_distribution(image, "small_negative_range")
        assert "small_negative_range" in caplog.text

    def test_exactly_zero_maximum(self, caplog):
        """Test array where maximum is exactly 0."""
        # Create >= 10 unique negative values and zeros
        image = np.array([-10.0, -9.0, -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0,
                         0.0, 0.0, 0.0])
        check_distribution(image, "max_is_zero")
        assert "max_is_zero" in caplog.text
        assert "max:" in caplog.text
