"""Credential pools for providers (design §18.5).

A provider can be given several API keys and rotate to the next one when the
current key is rate-limited or exhausted. Keys still come only from the
environment (design §11): a pool is read from one env var that may hold a
comma-separated list, plus numbered siblings ``<VAR>_1``, ``<VAR>_2``, …

A single key is just a pool of one, so existing single-key setups are unchanged.
"""

from __future__ import annotations

import os


class CredentialPool:
    """An ordered set of API keys with a rotating cursor."""

    def __init__(self, keys: list[str]) -> None:
        # Dedupe, preserving order; drop blanks.
        seen: set[str] = set()
        self._keys: list[str] = []
        for k in keys:
            k = (k or "").strip()
            if k and k not in seen:
                seen.add(k)
                self._keys.append(k)
        self._idx = 0
        self._exhausted: set[int] = set()

    @classmethod
    def from_env(cls, env_var: str, *, env: dict[str, str] | None = None) -> CredentialPool:
        """Collect keys from ``env_var`` (comma-separated) and ``env_var_1..N``."""
        env = os.environ if env is None else env
        keys: list[str] = [p.strip() for p in (env.get(env_var) or "").split(",")]
        i = 1
        while True:
            value = env.get(f"{env_var}_{i}")
            if not value:
                break
            keys.append(value)
            i += 1
        return cls(keys)

    def __len__(self) -> int:
        return len(self._keys)

    def current(self) -> str | None:
        """The active key, or None if the pool is empty."""
        return self._keys[self._idx] if self._keys else None

    def mark_exhausted(self) -> None:
        """Flag the active key as exhausted (rate-limited) for this session."""
        if self._keys:
            self._exhausted.add(self._idx)

    def rotate(self) -> bool:
        """Advance to the next key that isn't exhausted. False if none remain."""
        n = len(self._keys)
        for step in range(1, n):
            j = (self._idx + step) % n
            if j not in self._exhausted:
                self._idx = j
                return True
        return False
