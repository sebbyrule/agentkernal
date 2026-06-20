"""Shared HTTP transport for provider adapters.

A provider that is unreachable after retries raises ``ProviderError`` — one of
the few kernel faults that is allowed to propagate out of the loop (design §8.3).
Tests never touch this module: they exercise the pure translation functions
directly, so the suite stays offline (design §15).
"""

from __future__ import annotations

import email.utils
import time
from typing import Any

import httpx

_RETRYABLE_STATUS = {429, 500, 502, 503, 504}

# Cap how long a server-supplied Retry-After can stall us, so a hostile or
# misconfigured header can't park the loop for minutes.
_MAX_RETRY_AFTER = 30.0


class ProviderError(RuntimeError):
    """A non-recoverable provider/transport fault (config or network)."""


def _parse_retry_after(value: str | None) -> float | None:
    """Parse a ``Retry-After`` header (delta-seconds or HTTP-date) to seconds.

    Returns ``None`` when the header is absent or unparseable, and never a
    negative delay (a past date clamps to 0).
    """
    if not value:
        return None
    value = value.strip()
    if value.isdigit():
        return float(value)
    try:
        parsed = email.utils.parsedate_to_datetime(value)
    except (ValueError, TypeError):
        return None
    if parsed is None:
        return None
    delay = parsed.timestamp() - time.time()
    return max(0.0, delay)


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
        retry_after: float | None = None
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
            # Honor Retry-After (429/503) when the server tells us how long to
            # wait, bounded so a bad header can't stall the loop indefinitely.
            retry_after = _parse_retry_after(resp.headers.get("Retry-After"))
        if attempt < retries:
            backoff = 0.5 * (attempt + 1)
            delay = min(retry_after, _MAX_RETRY_AFTER) if retry_after is not None else backoff
            time.sleep(delay)
    raise ProviderError(f"{url} unreachable after {retries + 1} attempts; {last_detail}")
