# Load environment variables before importing curator
import os
from dotenv import load_dotenv

# Load API key from .env file
load_dotenv()

import time
from bespokelabs import curator
from datasets import Dataset
from bespokelabs.curator.utils import clear_cache
from pydantic import BaseModel, Field
from typing import List

# Constants
CURATOR_CACHE_DIR = "~/.cache/curator"  # Changed to match Curator's default
CACHE_CHECK_DELAY = 2  # seconds to wait after generation before checking cache

def check_cache_status():
    """Check the status of the Curator cache directory and return size and file count."""
    cache_path = os.path.expanduser(CURATOR_CACHE_DIR)
    
    if not os.path.exists(cache_path):
        print(f"Cache directory {CURATOR_CACHE_DIR} does not exist")
        return 0, 0
    
    print("\nCache Contents:")
    print("=" * 50)
    total_size = 0
    file_count = 0
    for dirpath, dirnames, filenames in os.walk(cache_path):
        rel_path = os.path.relpath(dirpath, cache_path)
        if rel_path == ".":
            print(f"\nRoot Directory ({cache_path}):")
        else:
            print(f"\nSubdirectory ({rel_path}):")
            
        for f in filenames:
            fp = os.path.join(dirpath, f)
            size = os.path.getsize(fp)
            total_size += size
            file_count += 1
            
            print(f"\n  File: {f}")
            print(f"     Size: {size / 1024:.2f} KB")
            
            # Show content for text files (skip binary files)
            if f.endswith(('.jsonl', '.txt', '.json', '.db')):
                try:
                    with open(fp, 'r') as file:
                        content = file.read()
                        print("     Content Preview (first 200 chars):")
                        print(f"     {content[:200]}...")
                except UnicodeDecodeError:
                    print("     [Binary file or different encoding]")
    
    print("\n" + "=" * 50)
    return total_size, file_count

def generate_cache_data():
    """Generate some cache data using the poem generation functionality."""
    print("\nStep 1: Generating cache data...")
    
    # Create a dataset object for the topics
    topics = Dataset.from_dict({"topic": [
        "Urban loneliness in a bustling city",
        "Beauty of Bespoke Labs's Curator library"
    ]})
    
    # Create a prompter instance using Curator
    poet = curator.Prompter(
        prompt_func=lambda row: f"Write a poem about {row['topic']}.",
        model_name="gpt-4",
        parse_func=lambda row, response: {
            "topic": row["topic"],
            "poem": response
        },
    )
    
    try:
        # Generate poems using Curator
        poems = poet(topics)
        print("Cache data generation completed.")
        return True
    except Exception as e:
        print(f"Error generating cache data: {str(e)}")
        return False

def test_cache_clearing():
    """Test the cache clearing functionality."""
    print("\n=== Testing Cache Clearing Functionality ===")
    
    # Step 1: Generate cache data
    success = generate_cache_data()
    if not success:
        print("Failed to generate cache data. Test cannot proceed.")
        return False
    
    # Wait a bit to ensure cache is written
    print(f"\nWaiting {CACHE_CHECK_DELAY} seconds for cache to be written...")
    time.sleep(CACHE_CHECK_DELAY)
    
    # Step 2: Check initial cache status
    print("\nStep 2: Checking initial cache status...")
    initial_size, initial_files = check_cache_status()
    print(f"Initial cache status:")
    print(f"- Total size: {initial_size / 1024 / 1024:.2f} MB")
    print(f"- Total files: {initial_files}")
    
    if initial_files == 0:
        print("No cache files found. Test cannot proceed.")
        return False
    
    # Step 3: Clear the cache using Curator's clear_cache function
    print("\nStep 3: Clearing cache...")
    try:
        clear_cache()
        print("Cache cleared successfully.")
    except Exception as e:
        print(f"Error clearing cache: {str(e)}")
        return False
    
    # Step 4: Verify cache is cleared
    print("\nStep 4: Verifying cache is cleared...")
    final_size, final_files = check_cache_status()
    print(f"Final cache status:")
    print(f"- Total size: {final_size / 1024 / 1024:.2f} MB")
    print(f"- Total files: {final_files}")
    
    # Step 5: Report results
    print("\n=== Test Results ===")
    if final_files == 0 and final_size == 0:
        print("✅ SUCCESS: Cache was properly cleared")
        print(f"- Initial cache size: {initial_size / 1024 / 1024:.2f} MB ({initial_files} files)")
        print(f"- Final cache size: 0.00 MB (0 files)")
        return True
    else:
        print("❌ FAILURE: Cache was not properly cleared")
        print(f"- {final_files} files remaining")
        print(f"- {final_size / 1024 / 1024:.2f} MB remaining")
        return False

if __name__ == "__main__":
    success = test_cache_clearing()
    exit(0 if success else 1)  # Exit with appropriate status code
