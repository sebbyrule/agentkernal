"""Shared HTTP transport for provider adapters.

A provider that is unreachable after retries raises ``ProviderError`` — one of
the few kernel faults that is allowed to propagate out of the loop (design §8.3).
Tests never touch this module: they exercise the pure translation functions
directly, so the suite stays offline (design §15).
"""

from __future__ import annotations

import time
from typing import Any

import httpx

_RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class ProviderError(RuntimeError):
    """A non-recoverable provider/transport fault (config or network)."""


def post_json(
    url: str,
    *,
    headers: dict[str, str],
    payload: dict[str, Any],
    timeout: float = 120.0,
    retries: int = 2,
) -> dict[str, Any]:
    """POST ``payload`` as JSON and return the parsed JSON response.

    Retries transient transport errors and retryable status codes; raises
    ``ProviderError`` once retries are exhausted or on a non-retryable 4xx.
    """
    last_detail = ""
    for attempt in range(retries + 1):
        try:
            resp = httpx.post(url, headers=headers, json=payload, timeout=timeout)
        except httpx.TransportError as exc:
            last_detail = f"transport error: {exc}"
        else:
            if resp.status_code < 400:
                return resp.json()
            # Body may contain provider error detail but never our secrets.
            last_detail = f"HTTP {resp.status_code}: {resp.text[:500]}"
            if resp.status_code not in _RETRYABLE_STATUS:
                raise ProviderError(f"{url} -> {last_detail}")
        if attempt < retries:
            time.sleep(0.5 * (attempt + 1))
    raise ProviderError(f"{url} unreachable after {retries + 1} attempts; {last_detail}")
