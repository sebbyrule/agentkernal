"""Credential-pool tests (design §18.5): key collection from env, rotation, and
the pooled POST that falls over to the next key on a rate limit."""

from __future__ import annotations

import pytest

from agentkernel.providers._http import ProviderError, RateLimitError, post_json_pooled
from agentkernel.providers.credentials import CredentialPool

# --- CredentialPool -----------------------------------------------------------

def test_from_env_comma_and_numbered_and_dedupe():
    env = {"K": "a, b ,a", "K_1": "c", "K_2": "b", "K_3": ""}
    pool = CredentialPool.from_env("K", env=env)
    # order preserved, blanks and duplicates dropped, numbered stops at the gap
    assert len(pool) == 3
    assert pool.current() == "a"


def test_single_key_pool():
    pool = CredentialPool(["only"])
    assert len(pool) == 1
    assert pool.current() == "only"
    assert pool.rotate() is False  # nowhere to go


def test_empty_pool():
    pool = CredentialPool([])
    assert len(pool) == 0
    assert pool.current() is None
    assert pool.rotate() is False


def test_rotate_skips_exhausted():
    pool = CredentialPool(["k1", "k2", "k3"])
    assert pool.current() == "k1"
    pool.mark_exhausted()
    assert pool.rotate() is True and pool.current() == "k2"
    pool.mark_exhausted()
    assert pool.rotate() is True and pool.current() == "k3"
    pool.mark_exhausted()
    assert pool.rotate() is False  # all spent


# --- post_json_pooled ---------------------------------------------------------

def test_pooled_rotates_to_working_key():
    seen_keys = []

    def fake_post(url, *, headers, payload, timeout, retries):
        key = headers["x-api-key"]
        seen_keys.append(key)
        if key == "k1":
            raise RateLimitError("429")
        return {"ok": True, "key": key}

    pool = CredentialPool(["k1", "k2"])
    out = post_json_pooled(
        "u",
        header_for_key=lambda k: {"x-api-key": k},
        payload={},
        pool=pool,
        _post=fake_post,
    )
    assert out == {"ok": True, "key": "k2"}
    assert seen_keys == ["k1", "k2"]  # tried k1, rotated to k2


def test_pooled_raises_when_all_rate_limited():
    def fake_post(url, *, headers, payload, timeout, retries):
        raise RateLimitError("429")

    pool = CredentialPool(["k1", "k2"])
    with pytest.raises(RateLimitError):
        post_json_pooled(
            "u", header_for_key=lambda k: {"x-api-key": k}, payload={},
            pool=pool, _post=fake_post,
        )


def test_pooled_does_not_rotate_on_non_ratelimit_error():
    calls = []

    def fake_post(url, *, headers, payload, timeout, retries):
        calls.append(headers["x-api-key"])
        raise ProviderError("HTTP 400")

    pool = CredentialPool(["k1", "k2"])
    with pytest.raises(ProviderError):
        post_json_pooled(
            "u", header_for_key=lambda k: {"x-api-key": k}, payload={},
            pool=pool, _post=fake_post,
        )
    assert calls == ["k1"]  # a 400 is not retried across keys
