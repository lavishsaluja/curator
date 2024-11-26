"""Tests for curator utility functions."""

import os
import pytest
from bespokelabs.curator import clear_cache

def test_clear_cache(tmp_path):
    """Test clear_cache function."""
    cache_dir = str(tmp_path)
    os.environ["CURATOR_CACHE_DIR"] = cache_dir

    # Create mock cache files
    metadata_db_path = os.path.join(cache_dir, "metadata.db")
    os.makedirs(cache_dir, exist_ok=True)
    with open(metadata_db_path, "w") as f:
        f.write("test")

    # Create some mock cache directories and files
    test_dir = os.path.join(cache_dir, "test_dir")
    os.makedirs(test_dir)
    with open(os.path.join(test_dir, "test.arrow"), "w") as f:
        f.write("test")

    # Clear cache
    clear_cache(working_dir=cache_dir)

    # Verify cache is cleared
    assert not os.path.exists(metadata_db_path)
    assert not os.path.exists(test_dir)
    assert len(os.listdir(cache_dir)) == 0

def test_clear_cache_nonexistent_dir():
    """Test clear_cache with non-existent directory."""
    nonexistent_dir = "/tmp/nonexistent_dir_12345"
    clear_cache(working_dir=nonexistent_dir)  # Should not raise error

def test_clear_cache_readonly_file(tmp_path):
    """Test clear_cache with read-only files."""
    cache_dir = str(tmp_path)
    os.environ["CURATOR_CACHE_DIR"] = cache_dir

    # Create read-only file
    readonly_file = os.path.join(cache_dir, "readonly.txt")
    os.makedirs(cache_dir, exist_ok=True)
    with open(readonly_file, "w") as f:
        f.write("test")
    os.chmod(readonly_file, 0o444)

    # Clear cache should handle read-only files
    clear_cache(working_dir=cache_dir)
    assert not os.path.exists(readonly_file)
