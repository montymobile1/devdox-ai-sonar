"""Async HTTP client with retry logic and rate limiting."""

import asyncio
import httpx
from typing import Optional, Dict, Any
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
import logging

logger = logging.getLogger(__name__)


class AsyncHTTPClient:
    """Async HTTP client with retries and rate limiting."""

    def __init__(
        self,
        timeout: float = 30.0,
        max_retries: int = 3,
        rate_limit_per_second: Optional[float] = None,
    ):
        self.timeout = httpx.Timeout(timeout)
        self.max_retries = max_retries
        self.rate_limit = rate_limit_per_second
        self._client: Optional[httpx.AsyncClient] = None
        self._semaphore = asyncio.Semaphore(10)  # Max concurrent requests
        self._last_request_time = 0.0

    async def __aenter__(self):
        self._client = httpx.AsyncClient(
            timeout=self.timeout,
            follow_redirects=True,
            limits=httpx.Limits(max_keepalive_connections=20, max_connections=100),
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._client:
            await self._client.aclose()

    async def _rate_limit_delay(self):
        """Enforce rate limiting."""
        if not self.rate_limit:
            return

        now = asyncio.get_event_loop().time()
        time_since_last = now - self._last_request_time
        min_interval = 1.0 / self.rate_limit

        if time_since_last < min_interval:
            await asyncio.sleep(min_interval - time_since_last)

        self._last_request_time = asyncio.get_event_loop().time()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.NetworkError)),
    )
    async def request(
        self,
        method: str,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        json: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> httpx.Response:
        """Make async HTTP request with retry logic."""
        async with self._semaphore:
            await self._rate_limit_delay()

            try:
                response = await self._client.request(
                    method=method, url=url, headers=headers, json=json, **kwargs
                )
                response.raise_for_status()
                return response
            except httpx.HTTPStatusError as e:
                logger.error(f"HTTP {e.response.status_code}: {url}")
                raise
            except httpx.TimeoutException:
                logger.warning(f"Timeout requesting {url}, retrying...")
                raise
            except Exception as e:
                logger.error(f"Request failed: {url}, error: {e}")
                raise

    async def get(self, url: str, **kwargs) -> httpx.Response:
        """Async GET request."""
        return await self.request("GET", url, **kwargs)

    async def post(self, url: str, **kwargs) -> httpx.Response:
        """Async POST request."""
        return await self.request("POST", url, **kwargs)
