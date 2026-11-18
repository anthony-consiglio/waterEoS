"""Read side of the atlas tables, with a per-release cache of the stored files.

The files change only when the loader writes a new release. The release record
is re-read at most every ``RELEASE_TTL`` seconds; a new release id empties the
cache, so a reload goes live within that time without a restart.

The database is named by ``$WATEREOS_ATLAS_URL``, then ``$WATEREOS_DATA_URL``
and ``$DATABASE_URL`` (in production the atlas shares the measured-data
database).
"""
from __future__ import annotations

import json
import os
import threading
import time

from watereos_api.atlas import schema as A
from watereos_api.measured import db as dbmod
from watereos_api.measured.store import NotConfigured, Unavailable

ENV_VAR = 'WATEREOS_ATLAS_URL'
RELEASE_TTL = 60.0
MAX_CACHE_ENTRIES = 128

__all__ = ['AtlasStore', 'NotConfigured', 'Unavailable', 'get_store', 'reset_store', 'url_from_env']


def url_from_env() -> str | None:
    return os.getenv(ENV_VAR, '').strip() or dbmod.url_from_env()


class AtlasStore:
    def __init__(self, url: str | None):
        self.target = dbmod.parse_url(url) if url else None
        self._lock = threading.Lock()
        self._release = None
        self._checked = 0.0
        self._files: dict = {}

    def _query(self, statement, params=()):
        if self.target is None:
            raise NotConfigured('the phase diagrams are not configured on this server')
        try:
            conn = dbmod.connect(self.target, readonly=True)
        except Exception as e:  # noqa: BLE001 - any driver error means "unavailable"
            raise Unavailable(f'cannot open {self.target.describe()}: {type(e).__name__}') from e
        try:
            cur = conn.cursor()
            cur.execute(dbmod.sql(self.target, statement), tuple(params))
            return cur.fetchall()
        except Exception as e:  # noqa: BLE001
            raise Unavailable(f'the phase diagrams are not loaded in {self.target.describe()} '
                              f'({type(e).__name__}); run python -m watereos_api.atlas.load') from e
        finally:
            conn.close()

    def release(self) -> dict:
        """The current release: release_id, created_at and the manifest (parsed)."""
        now = time.monotonic()
        with self._lock:
            if self._release is not None and now - self._checked < RELEASE_TTL:
                return self._release
        rows = self._query(f'SELECT {", ".join(A.RELEASE_COLUMNS)} FROM {A.TABLE_RELEASE}')
        if not rows:
            raise Unavailable('the atlas tables hold no release; run python -m watereos_api.atlas.load')
        rel = dict(zip(A.RELEASE_COLUMNS, rows[0]))
        if rel['schema_version'] != A.SCHEMA_VERSION:
            raise Unavailable(f'atlas schema {rel["schema_version"]} does not match the API ({A.SCHEMA_VERSION}); '
                              f'reload it with this version of the loader')
        rel['manifest'] = json.loads(rel['manifest'])
        with self._lock:
            if self._release is None or self._release['release_id'] != rel['release_id']:
                self._files.clear()
            self._release = rel
            self._checked = now
        return rel

    def file(self, solute: str, name: str):
        """(sha256, gzip-compressed JSON) of one file, or None when the release has no such file."""
        rid = self.release()['release_id']
        k = (rid, solute, name)
        with self._lock:
            if k in self._files:
                return self._files[k]
        rows = self._query(f'SELECT sha256, body FROM {A.TABLE_FILES} WHERE solute = ? AND name = ?', (solute, name))
        value = (rows[0][0], bytes(rows[0][1])) if rows else None
        with self._lock:
            if len(self._files) >= MAX_CACHE_ENTRIES:
                self._files.clear()
            self._files[k] = value
        return value


_STORE = None
_STORE_LOCK = threading.Lock()


def get_store() -> AtlasStore:
    global _STORE
    with _STORE_LOCK:
        if _STORE is None:
            _STORE = AtlasStore(url_from_env())
        return _STORE


def reset_store(url: str | None = None):
    """Forget the current store; the next call re-reads the environment (or uses ``url``)."""
    global _STORE
    with _STORE_LOCK:
        _STORE = AtlasStore(url) if url else None
