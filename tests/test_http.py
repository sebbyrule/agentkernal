"""Offline tests for the shared HTTP transport's pure helpers.

No network is touched: only the ``Retry-After`` parser is exercised directly,
keeping with the suite's offline contract (design §15).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from email.utils import format_datetime

from agentkernel.providers._http import _parse_retry_after


def test_retry_after_absent_or_blank():
    assert _parse_retry_after(None) is None
    assert _parse_retry_after("") is None
    assert _parse_retry_after("   ") is None


def test_retry_after_delta_seconds():
    assert _parse_retry_after("0") == 0.0
    assert _parse_retry_after("5") == 5.0
    assert _parse_retry_after(" 12 ") == 12.0


def test_retry_after_garbage_is_none():
    assert _parse_retry_after("soon") is None


def test_retry_after_http_date_in_future():
    future = datetime.now(UTC) + timedelta(seconds=30)
    delay = _parse_retry_after(format_datetime(future, usegmt=True))
    assert delay is not None
    # Allow slack for the clock advancing between formatting and parsing.
    assert 25.0 <= delay <= 31.0


def test_retry_after_past_date_clamps_to_zero():
    past = datetime.now(UTC) - timedelta(seconds=60)
    assert _parse_retry_after(format_datetime(past, usegmt=True)) == 0.0
