import os
from typing import Optional
import shutil

import pytest
from datasets import Dataset
from pydantic import BaseModel

from bespokelabs.curator import Prompter


class MockResponseFormat(BaseModel):
    """Mock response format for testing."""

    message: str
    confidence: Optional[float] = None


@pytest.fixture
def prompter() -> Prompter:
    """Create a Prompter instance for testing.

    Returns:
        PromptCaller: A configured prompt caller instance.
    """

    def prompt_func(row):
        return {
            "user_prompt": f"Context: {row['context']} Answer this question: {row['question']}",
            "system_prompt": "You are a helpful assistant.",
        }

    return Prompter(
        model_name="gpt-4o-mini",
        prompt_func=prompt_func,
        response_format=MockResponseFormat,
    )


@pytest.mark.test
def test_completions(prompter: Prompter, tmp_path):
    """Test that completions processes a dataset correctly.

    Args:
        prompter: Fixture providing a configured Prompter instance.
        tmp_path: Pytest fixture providing temporary directory.
    """
    # Create a simple test dataset
    test_data = {
        "context": ["Test context 1", "Test context 2"],
        "question": ["What is 1+1?", "What is 2+2?"],
    }
    dataset = Dataset.from_dict(test_data)

    # Set up temporary cache directory
    os.environ["BELLA_CACHE_DIR"] = str(tmp_path)

    result_dataset = prompter(dataset)
    result_dataset = result_dataset.to_huggingface()

    # Assertions
    assert len(result_dataset) == len(dataset)
    assert "message" in result_dataset.column_names
    assert "confidence" in result_dataset.column_names


@pytest.mark.test
def test_single_completion_batch(prompter: Prompter):
    """Test that a single completion works with batch=True.

    Args:
        prompter: Fixture providing a configured Prompter instance.
    """

    # Create a prompter with batch=True
    def simple_prompt_func():
        return [
            {
                "role": "user",
                "content": "Write a test message",
            },
            {
                "role": "system",
                "content": "You are a helpful assistant.",
            },
        ]

    batch_prompter = Prompter(
        model_name="gpt-4o-mini",
        prompt_func=simple_prompt_func,
        response_format=MockResponseFormat,
        batch=True,
    )

    # Get single completion
    result = batch_prompter()

    # Assertions
    assert isinstance(result, MockResponseFormat)
    assert hasattr(result, "message")
    assert hasattr(result, "confidence")


@pytest.mark.test
def test_single_completion_no_batch(prompter: Prompter):
    """Test that a single completion works without batch parameter.

    Args:
        prompter: Fixture providing a configured Prompter instance.
    """

    # Create a prompter without batch parameter
    def simple_prompt_func():
        return [
            {
                "role": "user",
                "content": "Write a test message",
            },
            {
                "role": "system",
                "content": "You are a helpful assistant.",
            },
        ]

    non_batch_prompter = Prompter(
        model_name="gpt-4o-mini",
        prompt_func=simple_prompt_func,
        response_format=MockResponseFormat,
    )

    # Get single completion
    result = non_batch_prompter()

    # Assertions
    assert isinstance(result, MockResponseFormat)
    assert hasattr(result, "message")
    assert hasattr(result, "confidence")


@pytest.mark.test
def test_clear_cache(prompter: Prompter, tmp_path):
    """Test that clear_cache removes all cached data."""
    # Set up temporary cache directory
    cache_dir = str(tmp_path)
    os.environ["CURATOR_CACHE_DIR"] = cache_dir

    # Create mock cache files and directories
    metadata_db_path = os.path.join(cache_dir, "metadata.db")
    os.makedirs(cache_dir, exist_ok=True)

    # Create a mock metadata.db file
    with open(metadata_db_path, 'wb') as f:
        f.write(b'mock db content')

    # Create mock cache directories and files
    mock_cache_dir = os.path.join(cache_dir, "mock_cache")
    os.makedirs(mock_cache_dir)
    with open(os.path.join(mock_cache_dir, "cache.arrow"), 'w') as f:
        f.write("mock cache data")

    # Create a read-only file to test permission errors
    readonly_file = os.path.join(cache_dir, "readonly.txt")
    with open(readonly_file, 'w') as f:
        f.write("readonly content")
    os.chmod(readonly_file, 0o444)

    # Verify cache exists
    assert os.path.exists(metadata_db_path)
    assert os.path.exists(mock_cache_dir)
    assert os.path.exists(readonly_file)

    # Clear cache
    prompter.clear_cache(working_dir=cache_dir)

    # Verify cache is cleared (even with readonly file)
    assert not os.path.exists(metadata_db_path)
    assert not os.path.exists(mock_cache_dir)
    assert not os.path.exists(readonly_file)

    # Test clearing non-existent directory (should not raise error)
    non_existent_dir = os.path.join(tmp_path, "non_existent")
    prompter.clear_cache(working_dir=non_existent_dir)

    # Test using environment variable
    os.environ["CURATOR_CACHE_DIR"] = str(tmp_path)
    os.makedirs(os.path.join(str(tmp_path), "env_var_test"))
    prompter.clear_cache()  # Should use CURATOR_CACHE_DIR
    assert len(os.listdir(str(tmp_path))) == 0
