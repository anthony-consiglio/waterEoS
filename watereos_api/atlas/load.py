"""Load a built cryo-isochoric atlas into the database behind the H2O Phase Diagram tab.

Usage (from the repository root)::

    python -m watereos_api.atlas.load --source /path/to/atlas
    python -m watereos_api.atlas.load --source /path/to/atlas --db postgresql://...
    python -m watereos_api.atlas.load --source /path/to/atlas --dry-run

``--source`` is the folder written by ``cryo-isochoric-gallery build nacl --out
<folder>``: ``index.html`` (whose manifest lists the views and their captions)
and ``data/<solute>/*.js``, one figure per file. ``--db`` defaults to
``$WATEREOS_ATLAS_URL``, ``$WATEREOS_DATA_URL``, ``$DATABASE_URL``, then the
local file ``.atlas-data/atlas.sqlite`` (git-ignored). Locally, keep the atlas
in its own SQLite file: the measured-data loader replaces its whole file.

What it does:

1. Reads the manifest and, for every solute built, each figure file and the
   system notes (``info``), leaving out the process view (``paths``), which
   the tab does not have.
2. Checks that each file is the figure it is named for and parses as JSON with
   no NaN or Infinity, that every view of the manifest is there (with its
   logarithmic twin where it has one), and that no local path appears in any
   published text.
3. Replaces the atlas tables, and only those, in one transaction, with every
   file gzip-compressed and a release record (date and a hash of the content):
   visitors see the old atlas or the new one, never a mix.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from watereos_api.atlas import schema as A
from watereos_api.atlas.store import url_from_env
from watereos_api.measured import db as dbmod

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SQLITE = REPO_ROOT / '.atlas-data' / 'atlas.sqlite'

_MANIFEST = re.compile(r'^const M = (\{.*\});\s*$', re.M)
_REGISTER = re.compile(r'^ATLAS\.register\("([^"]+)", ')
_INFO = re.compile(r'^ATLAS\.info\("([^"]+)", ')
_NAME = re.compile(A.NAME_PATTERN)
# Local paths that must never be published (strings only: base64 array data is skipped).
_LEFTOVER = re.compile(r'/Users/|/home/|/private/|~/|\\Users\\|\b[A-Za-z]:\\')


class LoadError(Exception):
    pass


def log(msg):
    print(msg, flush=True)


def _no_constant(name):
    raise ValueError(f'{name} is not JSON')


def read_manifest(source: Path) -> dict:
    index = source / 'index.html'
    if not index.is_file():
        raise LoadError(f'{index} not found: --source is the folder written by cryo-isochoric-gallery build')
    m = _MANIFEST.search(index.read_text(encoding='utf-8'))
    if not m:
        raise LoadError(f'no manifest (const M = ...) in {index}')
    manifest = json.loads(m.group(1), parse_constant=_no_constant)
    for k in ('solutes', 'groups', 'views', 'key', 'quality'):
        if k not in manifest:
            raise LoadError(f'the manifest in {index} has no {k!r}')
    if manifest.get('server'):
        raise LoadError(f'{index} is the shell of a running atlas server, not a static build')
    return manifest


def expected_files(views) -> list[str]:
    names = []
    for v in views:
        names.append(v['id'])
        if v.get('log'):
            names.append(v['id'] + '-log')
    return names + ['info']


def parse_file(path: Path, solute: str) -> tuple[str, str, object]:
    """(name, JSON text, value) of one data file of the atlas."""
    text = path.read_text(encoding='utf-8')
    m = _REGISTER.match(text) or _INFO.match(text)
    if not m:
        raise LoadError(f'{path.name}: not an atlas data file')
    if m.re is _INFO:
        if m.group(1) != solute:
            raise LoadError(f'{path.name}: notes of {m.group(1)!r}, expected {solute!r}')
        name = 'info'
    else:
        sol, _, name = m.group(1).partition('/')
        if sol != solute or not name:
            raise LoadError(f'{path.name}: figure {m.group(1)!r}, expected {solute}/...')
    body = text[m.end():].rstrip()
    if not body.endswith(');'):
        raise LoadError(f'{path.name}: truncated (no closing ");")')
    body = body[:-2]
    try:
        value = json.loads(body, parse_constant=_no_constant)
    except ValueError as e:
        raise LoadError(f'{path.name}: not valid JSON ({e})') from e
    if name == 'info':
        if not isinstance(value, dict) or value.get('id') != solute:
            raise LoadError(f'{path.name}: the notes are not those of {solute!r}')
    elif not (isinstance(value, dict) and isinstance(value.get('data'), list) and isinstance(value.get('layout'), dict)):
        raise LoadError(f'{path.name}: not a Plotly figure')
    return name, body, value


def _strings(v, key=None):
    """Every string of a JSON value, except base64 array data."""
    if isinstance(v, str):
        if key != 'bdata':
            yield v
    elif isinstance(v, dict):
        for k, x in v.items():
            yield from _strings(x, k)
    elif isinstance(v, list):
        for x in v:
            yield from _strings(x)


def check_private(where: str, value, forbid=()):
    for s in _strings(value):
        if _LEFTOVER.search(s) or any(t in s for t in forbid):
            raise LoadError(f'{where}: refusing to publish a local path or a forbidden term in {s[:120]!r}')


def read_atlas(source: Path, forbid=()):
    """(site manifest, [(solute, name, gz, sha256, raw_size)]) of the built atlas."""
    manifest = read_manifest(source)
    views = [v for v in manifest['views'] if v['id'] not in A.EXCLUDED_VIEWS and v['group'] not in A.EXCLUDED_GROUPS]
    groups = [g for g in manifest['groups'] if g['id'] not in A.EXCLUDED_GROUPS]
    want = expected_files(views)
    built = manifest.get('built') or [s['id'] for s in manifest['solutes']]
    if not built:
        raise LoadError('the atlas has no solute')
    files, solutes, builds = [], [], {}
    for s in manifest['solutes']:
        sid = s['id']
        if sid not in built:
            continue
        if not _NAME.match(sid):
            raise LoadError(f'unexpected solute id {sid!r}')
        folder = source / 'data' / sid
        have = {}
        for path in sorted(folder.glob('*.js')):
            stem = path.name[:-3]
            if stem in A.EXCLUDED_VIEWS:
                continue
            if stem not in want:
                log(f'  {sid}: skipping {path.name} (not a view of the manifest)')
                continue
            name, body, value = parse_file(path, sid)
            if name != stem:
                raise LoadError(f'{path.name} holds {name!r}')
            check_private(f'{sid}/{path.name}', value, forbid)
            raw = body.encode('utf-8')
            have[name] = (gzip.compress(raw, compresslevel=9, mtime=0), hashlib.sha256(raw).hexdigest(), len(raw))
        missing = [n for n in want if n not in have]
        if missing:
            raise LoadError(f'{sid}: missing {", ".join(missing)} (build the atlas again)')
        files += [(sid, n, *have[n]) for n in want]
        solutes.append(s)
        version = folder / 'version.txt'
        builds[sid] = version.read_text().strip() if version.is_file() else None
        raw_mb = sum(have[n][2] for n in want) / 1e6
        gz_mb = sum(len(have[n][0]) for n in want) / 1e6
        log(f'  {sid}: {len(want)} files, {raw_mb:.1f} MB, {gz_mb:.1f} MB compressed')
    for k in ('title', 'caption'):
        for v in views:
            check_private(f'manifest view {v["id"]}', v.get(k), forbid)
    site = {
        'solutes': solutes,
        'groups': groups,
        'views': views,
        'key': manifest['key'],
        'quality': manifest['quality'],
        'built_at': manifest.get('stamp'),
        'builds': builds,
    }
    return site, files


def make_release(site, files):
    h = hashlib.sha256(json.dumps(site, sort_keys=True).encode())
    for sid, name, _, sha, _ in files:
        h.update(f'{sid}/{name}:{sha}\n'.encode())
    now = datetime.now(timezone.utc)
    return {
        'release_id': f'{now:%Y-%m-%d}-{h.hexdigest()[:8]}',
        'created_at': now.isoformat(timespec='seconds'),
        'schema_version': A.SCHEMA_VERSION,
        'manifest': json.dumps(site, separators=(',', ':'), ensure_ascii=False),
    }


def write(target, files, release):
    """Replace the atlas tables (only those) in one transaction."""
    t0 = time.time()
    log(f'Writing to {target.describe()}')
    if target.kind == 'sqlite':
        Path(target.path).parent.mkdir(parents=True, exist_ok=True)
    conn = dbmod.connect(target, readonly=False)
    rcols = A.RELEASE_COLUMNS
    try:
        if target.kind == 'sqlite':
            conn.isolation_level = None            # explicit transaction, DDL included
            cur = conn.cursor()
            cur.execute('BEGIN IMMEDIATE')
            try:
                _rewrite(cur, target, files, release, rcols)
                cur.execute('COMMIT')
            except BaseException:
                cur.execute('ROLLBACK')
                raise
        else:
            with conn.transaction():
                _rewrite(conn.cursor(), target, files, release, rcols)
    finally:
        conn.close()
    log(f'Done in {time.time() - t0:.1f} s: release {release["release_id"]}, {len(files)} files')


def _rewrite(cur, target, files, release, rcols):
    for t in (A.TABLE_FILES, A.TABLE_RELEASE):
        cur.execute(f'DROP TABLE IF EXISTS {t}')
    for stmt in A.ddl(target.kind):
        cur.execute(stmt)
    ins = dbmod.sql(target, f'INSERT INTO {A.TABLE_FILES} ({", ".join(A.FILE_COLUMNS)}) VALUES (?, ?, ?, ?, ?)')
    for sid, name, gz, sha, size in files:
        cur.execute(ins, (sid, name, gz, sha, size))
    cur.execute(dbmod.sql(target, f'INSERT INTO {A.TABLE_RELEASE} ({", ".join(rcols)}) '
                                  f'VALUES ({", ".join(["?"] * len(rcols))})'),
                tuple(release[c] for c in rcols))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--source', required=True, type=Path, help='the folder written by cryo-isochoric-gallery build')
    ap.add_argument('--db', help='database URL (default: $WATEREOS_ATLAS_URL, $WATEREOS_DATA_URL, $DATABASE_URL, '
                                 f'then {DEFAULT_SQLITE.relative_to(REPO_ROOT)})')
    ap.add_argument('--dry-run', action='store_true', help='read and check only; write nothing')
    ap.add_argument('--forbid', action='append', default=[], metavar='TERM',
                    help='refuse to load if TERM appears in any published text (repeatable)')
    args = ap.parse_args(argv)
    try:
        t0 = time.time()
        log(f'Reading {args.source}')
        site, files = read_atlas(args.source, forbid=tuple(args.forbid))
        release = make_release(site, files)
        log(f'Read {len(files)} files of {len(site["solutes"])} solute(s) in {time.time() - t0:.1f} s; '
            f'release {release["release_id"]}')
        if args.dry_run:
            log('Dry run: nothing written')
            return 0
        url = args.db or url_from_env() or f'sqlite:///{DEFAULT_SQLITE}'
        write(dbmod.parse_url(url), files, release)
        return 0
    except LoadError as e:
        print(f'error: {e}', file=sys.stderr, flush=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
