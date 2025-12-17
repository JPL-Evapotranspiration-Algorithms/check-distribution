import pytest
from check_distribution import __version__


class TestVersion:
    """Test suite for version information."""

    def test_version_exists(self):
        """Test that __version__ attribute exists."""
        assert __version__ is not None

    def test_version_is_string(self):
        """Test that __version__ is a string."""
        assert isinstance(__version__, str)

    def test_version_not_empty(self):
        """Test that __version__ is not empty."""
        assert len(__version__) > 0

    def test_version_format(self):
        """Test that __version__ follows semantic versioning pattern."""
        # Should contain at least one dot (e.g., "1.0" or "1.0.0")
        assert "." in __version__
