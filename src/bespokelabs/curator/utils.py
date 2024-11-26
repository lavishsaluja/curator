"""Utility functions for the curator package."""

import os
import shutil
import logging

logger = logging.getLogger(__name__)

_CURATOR_DEFAULT_CACHE_DIR = "~/.cache/curator"

def clear_cache(working_dir: str | None = None) -> None:
    """Clear all cached data for curator.

    Args:
        working_dir (Optional[str]): Specific working directory to clear.
            If None, uses CURATOR_CACHE_DIR environment variable or default cache directory.
    """
    if working_dir is None:
        cache_dir = os.environ.get(
            "CURATOR_CACHE_DIR",
            os.path.expanduser(_CURATOR_DEFAULT_CACHE_DIR),
        )
    else:
        cache_dir = working_dir

    if not os.path.exists(cache_dir):
        logger.info(f"Cache directory {cache_dir} does not exist, nothing to clear")
        return

    # Clear metadata DB
    metadata_db_path = os.path.join(cache_dir, "metadata.db")
    if os.path.exists(metadata_db_path):
        try:
            os.remove(metadata_db_path)
            logger.info(f"Successfully removed metadata database at {metadata_db_path}")
        except Exception as e:
            logger.warning(f"Error while removing metadata.db: {e}")

    # Clear all cached files and run directories
    for item in os.listdir(cache_dir):
        item_path = os.path.join(cache_dir, item)
        try:
            if os.path.isfile(item_path):
                os.remove(item_path)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)
            logger.info(f"Successfully removed cache item: {item_path}")
        except Exception as e:
            logger.warning(f"Error while removing {item_path}: {e}")
