"""Read side of the measured-data database, with a per-release response cache.

The data change only when the loader writes a new release, so every response
is computed once per release and kept as JSON bytes. The current release id is
re-read from the database at most every ``RELEASE_TTL`` seconds; a new id
empties the cache, so a reload goes live within that time without a restart.
"""
from __future__ import annotations

import json
import threading
import time

from watereos_api.measured import db as dbmod
from watereos_api.measured import schema as S

RELEASE_TTL = 60.0
MAX_CACHE_ENTRIES = 256


class NotConfigured(Exception):
    """No database URL is set."""


class Unavailable(Exception):
    """The database is set but cannot be read (down, empty, wrong schema)."""


def dumps(obj) -> bytes:
    return json.dumps(obj, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()


class MeasuredStore:
    def __init__(self, url: str | None):
        self.url = url
        self.target = dbmod.parse_url(url) if url else None
        self._lock = threading.Lock()
        self._release = None
        self._checked = 0.0
        self._cache: dict = {}

    @property
    def configured(self) -> bool:
        return self.target is not None

    def query(self, statement, params=()):
        if not self.configured:
            raise NotConfigured('measured data are not configured on this server')
        try:
            conn = dbmod.connect(self.target, readonly=True)
        except Exception as e:  # noqa: BLE001 - any driver error means "unavailable"
            raise Unavailable(f'cannot open {self.target.describe()}: {type(e).__name__}') from e
        try:
            cur = conn.cursor()
            cur.execute(dbmod.sql(self.target, statement), tuple(params))
            cols = [S.CANONICAL_NAMES.get(d[0].lower(), d[0]) for d in cur.description]
            return cols, cur.fetchall()
        except Exception as e:  # noqa: BLE001
            raise Unavailable(f'query failed on {self.target.describe()}: {type(e).__name__}: {e}') from e
        finally:
            conn.close()

    def release(self) -> dict:
        """The current release record. Raises NotConfigured or Unavailable."""
        now = time.monotonic()
        with self._lock:
            if self._release is not None and now - self._checked < RELEASE_TTL:
                return self._release
        cols, rows = self.query(f'SELECT {", ".join(c for c, _ in S.RELEASE_COLUMNS)} FROM {S.TABLE_RELEASE}')
        if not rows:
            raise Unavailable('the measured-data database holds no release; run the loader')
        rel = dict(zip(cols, rows[0]))
        if rel.get('schema_version') != S.SCHEMA_VERSION:
            raise Unavailable(f'database schema {rel.get("schema_version")} does not match the API '
                              f'({S.SCHEMA_VERSION}); reload it with this version of the loader')
        with self._lock:
            if self._release is None or self._release['release_id'] != rel['release_id']:
                self._cache.clear()
            self._release = rel
            self._checked = now
        return rel

    def cached(self, key, producer):
        """``producer()`` -> bytes, computed once per release and key."""
        rid = self.release()['release_id']
        k = (rid, key)
        with self._lock:
            if k in self._cache:
                return rid, self._cache[k]
        value = producer()
        with self._lock:
            if len(self._cache) >= MAX_CACHE_ENTRIES:
                self._cache.clear()
            self._cache.setdefault(k, value)
            return rid, self._cache[k]


_STORE = None
_STORE_LOCK = threading.Lock()


def get_store() -> MeasuredStore:
    global _STORE
    with _STORE_LOCK:
        if _STORE is None:
            _STORE = MeasuredStore(dbmod.url_from_env())
        return _STORE


def reset_store(url: str | None = None):
    """Forget the current store; the next call re-reads the environment (or uses ``url``)."""
    global _STORE
    with _STORE_LOCK:
        _STORE = MeasuredStore(url) if url else None
