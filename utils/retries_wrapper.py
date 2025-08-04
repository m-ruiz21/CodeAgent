import asyncio
from typing import Callable, Awaitable
import time
import httpx

async def retries_wrapper(
    fun: Callable[[], Awaitable],
    retries: int,
    desc: str,
):
    """
    Retries a function call with exponential backoff
    
    :param fun: The function to call.
    :param retries: Number of retries.
    :param desc: Description for logging.

    :return: The result of the function call, or None if all retries fail.
    """
    delay = 0.0
    last_exception = None
    
    for attempt in range(retries):
        try:
            # Create a fresh coroutine for each attempt
            return await fun()
        except Exception as e:
            last_exception = e
            print(f"Attempt {attempt + 1}/{retries} failed for {desc}: {str(e)}")
            
            # Check if it's an HTTP rate limit error
            if hasattr(e, 'response') and hasattr(e.response, 'headers'):
                reset = int(e.response.headers.get("x-rateLimit-reset", time.time()+60))
                delay = max(reset - time.time(), 30)
                print(f"Rate limit exceeded, retrying in {delay} seconds...")
            else:
                # Exponential backoff for other errors
                delay = min(2 ** (5 + attempt), 120)  # Cap at 120 seconds
                print(f"Retrying in {delay} seconds...")
            
            if attempt < retries - 1:  # Don't sleep on the last attempt
                await asyncio.sleep(delay)
                print("Awake to resume operation")
    
    # If we've exhausted all retries, raise the last exception
    print(f"All {retries} attempts failed for {desc}")
    if last_exception:
        raise last_exception
    else:
        raise RuntimeError(f"All {retries} attempts failed for {desc}")
