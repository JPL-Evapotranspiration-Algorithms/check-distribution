import pytest
import check_distribution


class TestModuleStructure:
    """Test suite for module structure and exports."""

    def test_check_distribution_function_exists(self):
        """Test that check_distribution function is exported."""
        assert hasattr(check_distribution, 'check_distribution')
        assert callable(check_distribution.check_distribution)

    def test_blank_output_error_exists(self):
        """Test that BlankOutputError is exported."""
        assert hasattr(check_distribution, 'BlankOutputError')
        assert issubclass(check_distribution.BlankOutputError, Exception)

    def test_version_exists(self):
        """Test that __version__ is exported."""
        assert hasattr(check_distribution, '__version__')

    def test_author_exists(self):
        """Test that __author__ is exported."""
        assert hasattr(check_distribution, '__author__')
        assert isinstance(check_distribution.__author__, str)

    def test_author_value(self):
        """Test that __author__ has expected value."""
        assert check_distribution.__author__ == "Gregory H. Halverson"
