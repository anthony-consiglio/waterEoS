"""Layout of the atlas tables, shared by the loader and the API.

Two tables, next to the measured-data tables in the same database:

* ``atlas_files``   one row per published file: a figure (``d-T-xv``, its
                    logarithmic twin ``d-T-xv-log``, ...) or the system notes
                    (``info``), as gzip-compressed JSON with the SHA-256 of the
                    uncompressed text;
* ``atlas_release`` one row: the release id, when it was loaded, the schema
                    version and the manifest (groups, views, captions, color
                    key, the solute) as JSON.
"""
from __future__ import annotations

SCHEMA_VERSION = 1

TABLE_FILES = 'atlas_files'
TABLE_RELEASE = 'atlas_release'

FILE_COLUMNS = ['solute', 'name', 'body', 'sha256', 'raw_size']
RELEASE_COLUMNS = ['release_id', 'created_at', 'schema_version', 'manifest']

# Groups and views of the cryo-isochoric atlas left out of the tab: the process
# view (the isochoric chamber) is computed on request by the atlas server and
# has no static figure to publish.
EXCLUDED_GROUPS = {'process'}
EXCLUDED_VIEWS = {'paths'}

# Solute ids and file names: lower-case words joined by hyphens (``nacl``,
# ``d-T-xv-log``, ``proj-pvT``).
NAME_PATTERN = r'^[a-z0-9]+(?:-[a-zA-Z0-9]+)*$'


def ddl(kind: str):
    """CREATE TABLE statements for 'sqlite' or 'postgres'."""
    blob = 'BYTEA' if kind == 'postgres' else 'BLOB'
    return [
        f'CREATE TABLE {TABLE_FILES} (\n  solute TEXT NOT NULL,\n  name TEXT NOT NULL,\n  body {blob} NOT NULL,\n'
        f'  sha256 TEXT NOT NULL,\n  raw_size INTEGER NOT NULL,\n  PRIMARY KEY (solute, name)\n)',
        f'CREATE TABLE {TABLE_RELEASE} (\n  release_id TEXT PRIMARY KEY,\n  created_at TEXT,\n'
        f'  schema_version INTEGER,\n  manifest TEXT\n)',
    ]
