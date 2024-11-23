import asyncio
from time import sleep
import nest_asyncio

# Apply nest_asyncio at module level to avoid multiple applications
nest_asyncio.apply()

def run_in_event_loop(coroutine):
    """
    Run a coroutine in the current event loop or create a new one if there isn't one.
    """
    try:
        loop = asyncio.get_running_loop()
        return loop.run_until_complete(coroutine)
    except RuntimeError:
        # If no event loop is running, asyncio will
        # return a RuntimeError (https://docs.python.org/3/library/asyncio-eventloop.html#asyncio.get_running_loop).
        # In that case, we can just use asyncio.run.
        return asyncio.run(coroutine)
