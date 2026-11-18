"""Connections to the measured-data database: SQLite or PostgreSQL.

The database is named by a URL:

* ``sqlite:///relative/path.sqlite``, ``sqlite:////absolute/path.sqlite`` or a
  plain file path ending in ``.sqlite``/``.db``: a SQLite file (local work,
  tests);
* ``postgresql://user:pass@host:port/db`` (or ``postgres://``): PostgreSQL,
  through psycopg 3 (production).

The API reads the URL from ``WATEREOS_DATA_URL``, then ``DATABASE_URL``.
"""
from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path

ENV_VARS = ('WATEREOS_DATA_URL', 'DATABASE_URL')


@dataclass(frozen=True)
class Target:
    kind: str           # 'sqlite' or 'postgres'
    path: str = ''      # SQLite file
    url: str = ''       # PostgreSQL URL

    def describe(self):
        """A printable name that never shows a password."""
        if self.kind == 'sqlite':
            return f'SQLite file {self.path}'
        from urllib.parse import urlsplit
        u = urlsplit(self.url)
        host = u.hostname or '?'
        port = f':{u.port}' if u.port else ''
        return f'PostgreSQL database {u.path.lstrip("/") or "?"} on {host}{port}'


def parse_url(url: str) -> Target:
    url = url.strip()
    if url.startswith(('postgresql://', 'postgres://', 'postgresql+psycopg://')):
        return Target('postgres', url=url.replace('postgresql+psycopg://', 'postgresql://', 1))
    if url.startswith('sqlite:///'):
        return Target('sqlite', path=url[len('sqlite:///'):])
    if url.startswith('sqlite:'):
        raise ValueError(f'use sqlite:///relative/path or sqlite:////absolute/path, not {url!r}')
    if '://' not in url and url.endswith(('.sqlite', '.sqlite3', '.db')):
        return Target('sqlite', path=url)
    raise ValueError('unsupported database URL (expected sqlite:///..., a .sqlite path or postgresql://...)')


def url_from_env() -> str | None:
    for name in ENV_VARS:
        v = os.getenv(name, '').strip()
        if v:
            return v
    return None


def connect(target: Target, readonly: bool = True):
    """A DB-API connection. SQLite opens read-only unless asked otherwise."""
    if target.kind == 'sqlite':
        p = Path(target.path)
        if readonly:
            if not p.is_file():
                raise FileNotFoundError(f'no SQLite database at {p}')
            return sqlite3.connect(f'{p.resolve().as_uri()}?mode=ro', uri=True, check_same_thread=False)
        return sqlite3.connect(str(p))
    import psycopg  # imported lazily: only production needs it
    conn = psycopg.connect(target.url, connect_timeout=10)
    if readonly:
        conn.read_only = True
    return conn


def sql(target: Target, statement: str) -> str:
    """Statements are written with ``?`` placeholders; psycopg wants ``%s``."""
    return statement.replace('?', '%s') if target.kind == 'postgres' else statement
