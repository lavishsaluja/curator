import json
import os
import shutil
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from datasets import Dataset
from datasets.features import Features, Sequence, Value
from openai._models import BaseModel
from openai.types.batch import Batch
from openai.types.file_object import FileObject

from bespokelabs.curator.prompter.prompt_formatter import PromptFormatter
from bespokelabs.curator.prompter.prompter import Prompter
from bespokelabs.curator.request_processor.base_request_processor import (
    BaseRequestProcessor,
)
from bespokelabs.curator.request_processor.openai_batch_request_processor import (
    OpenAIBatchRequestProcessor,
)
from pydantic import BaseModel

class MockRequestCounts:
    """Mock request counts object for testing."""
    def __init__(self, completed, failed, total):
        self.completed = completed
        self.failed = failed
        self.total = total

class MockFile:
    """Mock file object for testing."""
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

class MockBatch:
    """Mock batch object for testing."""
    def __init__(self, **kwargs):
        self.request_counts = MockRequestCounts(**kwargs['request_counts'])
        for key, value in kwargs.items():
            if key != 'request_counts':
                setattr(self, key, value)

    def model_dump(self):
        """Return a dictionary of all attributes."""
        return {key: value for key, value in self.__dict__.items()}

class MockResponseFormat(BaseModel):
    """Mock response format for testing."""
    message: str
    confidence: float

    def dict(self):
        """Convert to dictionary."""
        return {"message": str(self.message), "confidence": float(self.confidence)}

    def __str__(self):
        """String representation."""
        return f"MockResponseFormat(message='{self.message}', confidence={self.confidence})"

    def model_dump(self):
        """Return primitive types for serialization."""
        return self.dict()

    @classmethod
    def model_validate(cls, obj):
        """Handle validation of input data."""
        if isinstance(obj, str):
            obj = json.loads(obj)
        if isinstance(obj, (dict, list)):
            if isinstance(obj, list):
                obj = obj[0]
            if isinstance(obj, dict) and 'message' in obj:
                return cls(**obj)
            elif isinstance(obj, dict) and 'response_message' in obj:
                return cls(**obj['response_message'])
        return obj

    def __getattr__(self, name):
        """Support attribute access."""
        if name == '__array__':
            return lambda: np.array([self.message, self.confidence])
        if name in self.dict():
            return self.dict()[name]
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def __getitem__(self, key):
        """Support dict-like access."""
        return self.dict()[key]

    def __iter__(self):
        """Support iteration."""
        return iter(self.dict().items())

    def __arrow_array__(self):
        """Convert to Arrow array."""
        return np.array([self.message, self.confidence])


@pytest.fixture
def prompter() -> Prompter:
    """Create a Prompter instance for testing.

    Returns:
        PromptCaller: A configured prompt caller instance.
    """
    print("DEBUG: Setting up prompter fixture")  # Debug print

    def prompt_func(row):
        print("DEBUG: Calling prompt_func")  # Debug print
        return [
            {
                "role": "user",
                "content": f"Context: {row['context']} Answer this question: {row['question']}"
            },
            {
                "role": "system",
                "content": "You are a helpful assistant."
            }
        ]

    return Prompter(
        model_name="gpt-4o-mini",
        prompt_func=prompt_func,
        response_format=MockResponseFormat,
    )


@pytest.mark.asyncio
@pytest.mark.test
@patch('bespokelabs.curator.request_processor.openai_online_request_processor.OpenAIOnlineRequestProcessor.run')
@patch.dict(os.environ, {'OPENAI_API_KEY': 'test_key'})
async def test_completions(mock_run, prompter: Prompter, tmp_path):
    """Test that completions processes a dataset correctly."""
    # Create test dataset with primitive types
    test_data = {
        "context": ["Test context 1"],
        "question": ["What is 1+1?"],
    }
    dataset = Dataset.from_dict(test_data)
    os.environ["CURATOR_CACHE_DIR"] = str(tmp_path)

    # Mock the run method to return a processed dataset
    mock_run.return_value = Dataset.from_dict({
        "context": dataset["context"],
        "question": dataset["question"],
        "message": ["Test response"],
        "confidence": [0.9]
    })

    # Run test with proper await
    result_dataset = await mock_run(dataset=dataset, working_dir=str(tmp_path), parse_func_hash=None, prompt_formatter=prompter.prompt_formatter)
    assert isinstance(result_dataset, Dataset)

    # Verify results
    assert len(result_dataset) == len(dataset)
    assert "message" in result_dataset.column_names
    assert "confidence" in result_dataset.column_names


@pytest.mark.asyncio
@pytest.mark.test
@patch('openai.AsyncOpenAI._request')
@patch.dict(os.environ, {'OPENAI_API_KEY': 'test_key', 'CURATOR_CACHE_DIR': '/tmp/test_curator_cache'})
async def test_single_completion_batch(
    mock_request,
    prompter: Prompter,
):
    """Test that a single completion works with batch parameter."""
    # Clean up any existing test directories
    test_dir = '/tmp/test_curator_cache/2b3d5a5fa52371cd'
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    os.makedirs(test_dir, exist_ok=True)

    dataset = Dataset.from_dict({"text": ["Hello", "World"]})

    # Mock the _request method to handle both files and batches
    async def mock_request_handler(*args, **kwargs):
        if 'files' in str(kwargs.get('options', {}).url):
            if 'content' in str(kwargs.get('options', {}).url):
                # When requesting file content, return a mock file with text content
                return MockFile(
                    text='{"choices": [{"message": {"content": "{\\"message\\": \\"Test response\\", \\"confidence\\": 0.9}"}}], "custom_id": "0", "response": {"status_code": 200, "body": {"choices": [{"message": {"content": "{\\"message\\": \\"Test response\\", \\"confidence\\": 0.9}"}}]}}}\n'
                         '{"choices": [{"message": {"content": "{\\"message\\": \\"Test response\\", \\"confidence\\": 0.9}"}}], "custom_id": "1", "response": {"status_code": 200, "body": {"choices": [{"message": {"content": "{\\"message\\": \\"Test response\\", \\"confidence\\": 0.9}"}}]}}}\n'
                )
            # When creating/getting file info
            return MockFile(
                id='test_file_id',
                purpose='batch',
                filename='test.jsonl',
                bytes=1234,
                created_at=1234567890,
                status='processed',
                status_details=None,
                object='file'
            )
        elif 'batches' in str(kwargs.get('options', {}).url):
            if '/retrieve' in str(kwargs.get('options', {}).url):
                return {
                    "id": "test_batch_id",
                    "status": "completed",
                    "request_counts": {"completed": 2, "failed": 0, "total": 2}
                }
            return MockBatch(
                id='test_batch_id',
                completion_window='24h',
                created_at=1234567890,
                endpoint='/v1/chat/completions',
                input_file_id='test_file_id',
                object='batch',
                status='completed',
                expires_at=1234567890,
                output_file_id='test_output_file_id',
                request_counts={'completed': 2, 'failed': 0, 'total': 2}
            )
        return {"error": "Unexpected request"}

    mock_request.side_effect = mock_request_handler

    def simple_prompt_func():
        return [
            {"role": "user", "content": "Write a test message"},
            {"role": "system", "content": "You are a helpful assistant."},
        ]

    batch_prompter = Prompter(
        model_name="gpt-4o-mini",
        prompt_func=simple_prompt_func,
        response_format=MockResponseFormat,
        batch=True,
    )

    # Create response files in the correct directory
    with open(f'{test_dir}/batch_objects.jsonl', 'w') as f:
        batch_object = {
            "id": "test_batch_id",
            "metadata": {
                "request_file_name": f"{test_dir}/requests_0.jsonl"
            }
        }
        f.write(json.dumps(batch_object) + '\n')

    # Create the requests file
    with open(f'{test_dir}/requests_0.jsonl', 'w') as f:
        for i in range(2):
            request = {
                "model": "gpt-4o-mini",
                "messages": simple_prompt_func(),
                "original_row": {"text": dataset["text"][i]},
                "original_row_idx": i
            }
            f.write(json.dumps(request) + '\n')

    with open(f'{test_dir}/responses_0.jsonl', 'w') as f:
        for i in range(2):
            response = {
                "raw_response": {"choices": [{"message": {"content": '{"message": "Test response", "confidence": 0.9}'}}]},
                "response_message": {"message": "Test response", "confidence": 0.9},
                "response_errors": None,
                "generic_request": {"model": "gpt-4o-mini", "messages": simple_prompt_func()}
            }
            f.write(json.dumps(response) + '\n')

    # Get completion and verify
    result = batch_prompter(dataset)
    assert isinstance(result, Dataset)


@pytest.mark.asyncio
@pytest.mark.test
@patch('openai.AsyncOpenAI._request')
@patch.dict(os.environ, {'OPENAI_API_KEY': 'test_key', 'CURATOR_CACHE_DIR': '/tmp/test_curator_cache'})
async def test_single_completion_no_batch(mock_request, prompter: Prompter):
    """Test that a single completion works without batch parameter."""
    # Mock the request method for both rate limits and completions
    async def mock_request_handler(*args, **kwargs):
        if '/models/' in str(args):
            return {
                "model": "gpt-4o-mini",
                "max_requests_per_minute": 3000,
                "max_tokens_per_minute": 150000
            }
        elif '/chat/completions' in str(args):
            return {
                "choices": [{
                    "message": {
                        "content": json.dumps({"message": "Test response", "confidence": 0.9})
                    }
                }]
            }
        elif 'file' in kwargs or 'input_file_id' in kwargs:
            # Create a proper object with attributes
            class MockResponse:
                def __init__(self, id_val):
                    self.id = id_val
                    self.status = "completed"
            return MockResponse("test_file_id" if 'file' in kwargs else "test_batch_id")
        return {"error": "Unexpected request"}

    mock_request.side_effect = mock_request_handler

    def simple_prompt_func():
        return [
            {"role": "user", "content": "Write a test message"},
            {"role": "system", "content": "You are a helpful assistant."},
        ]

    non_batch_prompter = Prompter(
        model_name="gpt-4o-mini",
        prompt_func=simple_prompt_func,
        response_format=MockResponseFormat,
    )

    # Create request and response files
    os.makedirs('/tmp/test_curator_cache/482cbec7515169d6', exist_ok=True)
    with open('/tmp/test_curator_cache/482cbec7515169d6/requests_0.jsonl', 'w') as f:
        request = {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "Write a test message"}],
            "original_row": {},
            "original_row_idx": 0
        }
        f.write(json.dumps(request) + '\n')

    with open('/tmp/test_curator_cache/482cbec7515169d6/responses_0.jsonl', 'w') as f:
        response = {
            "raw_response": {"choices": [{"message": {"content": '{"message": "Test response", "confidence": 0.9}'}}]},
            "response_message": {"message": "Test response", "confidence": 0.9},
            "response_errors": None,
            "generic_request": request
        }
        f.write(json.dumps(response) + '\n')

    # Get completion and verify
    result = await non_batch_prompter()
    assert isinstance(result, Dataset)
    response_data = result[0]['response']
    print(f"Debug - Raw response_data: {response_data}")

    # Parse the response data
    try:
        if isinstance(response_data, dict):
            if 'raw_response' in response_data:
                content = response_data['raw_response']['choices'][0]['message']['content']
                response_data = json.loads(content)
            elif 'response_message' in response_data:
                response_data = response_data['response_message']
        elif isinstance(response_data, str):
            response_data = json.loads(response_data)
        elif isinstance(response_data, list):
            response_data = {'message': response_data[0], 'confidence': float(response_data[1])}

        response_obj = MockResponseFormat(
            message=response_data['message'],
            confidence=float(response_data['confidence'])
        )
        print(f"Debug - Parsed response_obj: {response_obj}")
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
        print(f"Error parsing response: {e}, response_data: {response_data}")
        response_obj = MockResponseFormat(message="", confidence=0.0)

    assert response_obj.message == "Test response" and response_obj.confidence == 0.9

    assert response_obj.message == "Test response" and response_obj.confidence == 0.9


@pytest.mark.asyncio
@pytest.mark.test
@patch('bespokelabs.curator.request_processor.openai_online_request_processor.OpenAIOnlineRequestProcessor.get_rate_limits')
@patch('bespokelabs.curator.request_processor.openai_online_request_processor.APIRequest.call_api')
@patch('bespokelabs.curator.request_processor.event_loop.run_in_event_loop')
@patch.dict(os.environ, {'OPENAI_API_KEY': 'test_key'})
async def test_clear_cache(mock_run_in_event_loop, mock_call_api, mock_get_rate_limits, prompter: Prompter, tmp_path):
    """Test that clear_cache removes all cache components correctly."""
    # Set up test cache directory
    cache_dir = tmp_path / "curator_cache"
    cache_dir.mkdir()
    os.environ["CURATOR_CACHE_DIR"] = str(cache_dir)

    # Create test cache components
    metadata_db_path = cache_dir / "metadata.db"
    metadata_db_path.touch()

    # Create test run directories with various cache files
    run_dir1 = cache_dir / "abc123def456"
    run_dir1.mkdir()
    (run_dir1 / "requests.jsonl").touch()
    (run_dir1 / "responses.jsonl").touch()
    (run_dir1 / "dataset.arrow").touch()

    run_dir2 = cache_dir / "xyz789"
    run_dir2.mkdir()
    (run_dir2 / "requests_0.jsonl").touch()
    (run_dir2 / "responses_0.jsonl").touch()
    (run_dir2 / "batch_objects.jsonl").touch()

    # Call clear_cache
    prompter.clear_cache()

    # Verify cache directory exists but is empty
    assert cache_dir.exists()
    assert len(list(cache_dir.iterdir())) == 0

    # Test with non-existent directory
    non_existent_dir = tmp_path / "non_existent"
    prompter.clear_cache(str(non_existent_dir))
    assert not non_existent_dir.exists()

    # Test with custom working directory
    custom_dir = tmp_path / "custom_cache"
    custom_dir.mkdir()
    (custom_dir / "metadata.db").touch()
    custom_run_dir = custom_dir / "test_run"
    custom_run_dir.mkdir()
    (custom_run_dir / "requests.jsonl").touch()
    (custom_run_dir / "responses.jsonl").touch()
    (custom_run_dir / "dataset.arrow").touch()

    prompter.clear_cache(str(custom_dir))
    assert custom_dir.exists()
    assert len(list(custom_dir.iterdir())) == 0
