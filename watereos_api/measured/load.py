"""Load the curated experimental water database into the measured-data database.

Usage (from the repository root)::

    python -m watereos_api.measured.load --source /path/to/fitting_data
    python -m watereos_api.measured.load --source /path/to/fitting_data --db postgresql://...
    python -m watereos_api.measured.load --source /path/to/fitting_data --dry-run

``--db`` defaults to ``$WATEREOS_DATA_URL``, then ``$DATABASE_URL``, then the
local file ``.measured-data/measured.sqlite`` (git-ignored).

What it does:

1. Reads every table CSV under ``liquid/``, ``phase_equilibrium/``,
   ``amorphous/``, ``kinetic/``, ``dynamics/`` and ``structure/``, plus
   ``sources.csv``. Nothing in the source folder is written; if a file changes
   while it is being read (a build in progress), the read is repeated.
2. Checks every row against ``fitting_data/SCHEMA.md`` (columns, unique ids,
   numbers, substance) and stops on the first table that fails.
3. Removes what is private to the machine the database was built on: the
   local paths in ``via`` become the name of the supplement they point to (or
   nothing, for the database's own transcriptions), and mentions of internal
   files in the notes are reworded. A final scan refuses to load if any local
   path is left in any published text.
4. Adds the absolute standard uncertainty ``u_std`` (k = 1), the rate in K/min,
   the publication year, and a release record (date, counts and a hash of the
   inputs) that the web app shows.
5. Replaces the database contents in one step: a new SQLite file is renamed
   over the old one; PostgreSQL is rewritten in a single transaction, so
   visitors see either the old release or the new one, never a mix.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import math
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from watereos_api.measured import db as dbmod
from watereos_api.measured import schema as S

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SQLITE = REPO_ROOT / '.measured-data' / 'measured.sqlite'

SUBSTANCES = {'H2O', 'D2O'}
U_TYPES = {'absolute', 'relative', 'none', ''}
RATE_TO_K_PER_MIN = {'K/min': 1.0, 'K/s': 60.0, 'K/h': 1.0 / 60.0}


class LoadError(Exception):
    pass


def log(msg):
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------
def _input_files(source: Path):
    files = []
    for grp in S.GROUP_KEYS:
        d = source / grp
        if d.is_dir():
            files += [(grp, p) for p in sorted(d.glob('*.csv'))]
    if not files:
        raise LoadError(f'no table CSVs under {source} (expected liquid/, phase_equilibrium/, ...)')
    src = source / 'sources.csv'
    if not src.is_file():
        raise LoadError(f'{src} not found')
    return files, src


def _stat(p: Path):
    st = p.stat()
    return (st.st_mtime_ns, st.st_size)


def _read_snapshot(source: Path, attempts=4):
    """Read all inputs as bytes. Repeat if any file changes during the read."""
    for attempt in range(1, attempts + 1):
        files, src = _input_files(source)
        paths = [p for _, p in files] + [src]
        before = {p: _stat(p) for p in paths}
        blobs = {p: p.read_bytes() for p in paths}
        after = {p: _stat(p) for p in paths}
        changed = [p for p in paths if before[p] != after[p]]
        if not changed:
            return files, src, blobs, max(after[p][0] for p in paths)
        log(f'  {len(changed)} file(s) changed while reading (a build in progress?); '
            f'retrying in 5 s ({attempt}/{attempts})')
        time.sleep(5)
    raise LoadError('the source files kept changing while being read; try again when the build has finished')


def _parse_csv(blob: bytes, name: str):
    text = blob.decode('utf-8-sig')
    rd = csv.DictReader(io.StringIO(text, newline=''))
    header = rd.fieldnames or []
    rows = []
    for i, r in enumerate(rd, start=2):
        if None in r or any(v is None for v in r.values()):
            raise LoadError(f'{name}, line {i}: wrong number of fields (file truncated or malformed?)')
        rows.append(r)
    return header, rows


# ---------------------------------------------------------------------------
# Values
# ---------------------------------------------------------------------------
def _num(v, col, rid):
    v = (v or '').strip()
    if not v or v.lower() in ('nan', 'na', 'n/a'):
        return None
    try:
        x = float(v)
    except ValueError as e:
        raise LoadError(f'row {rid}: column {col} is not a number: {v!r}') from e
    return x if math.isfinite(x) else None


def standard_uncertainty(value, u, u_type, u_k):
    """Absolute standard (k = 1) uncertainty of the value, as fitting_data/scripts/fdata.py computes it."""
    if u is None:
        return None
    ua = u * abs(value) if (u_type == 'relative' and value is not None) else u
    if u_type == 'relative' and value is None:
        return None
    return ua / (u_k if u_k else 1.0)


def rate_K_per_min(rate, unit):
    try:
        v = float(rate)
    except (TypeError, ValueError):
        return None
    f = RATE_TO_K_PER_MIN.get((unit or '').strip())
    return v * f if f and math.isfinite(v) else None


def short_cite(citation, year):
    """'Holten & Anisimov (2012)' / 'Holten et al. (2014)' from a full citation."""
    authors = (citation or '').split(' (')[0]
    names = [a.strip() for a in authors.replace(' and ', ', ').split(',') if a.strip()]
    sur = [n.split()[-1] for n in names if n.split()]
    if not sur:
        return str(year or '')
    if len(sur) == 1:
        s = sur[0]
    elif len(sur) == 2:
        s = f'{sur[0]} & {sur[1]}'
    else:
        s = f'{sur[0]} et al.'
    return f'{s} ({year})' if year else s


# ---------------------------------------------------------------------------
# Removing what is private to the build machine
# ---------------------------------------------------------------------------
# Any of these left in a published text means a local path slipped through.
_LEFTOVER = re.compile(
    r'/Users/|/home/|~/|\\Users\\|\breferences/|\bfitting_data/|'
    r'\breading_notes/|\btranscribed/|\bchecks/|\bstubs/|\bdashboard/')

_TEXT_REWRITES = [
    (re.compile(r'\breading_notes/([A-Za-z0-9_\-]+)\.md'), r'the reading note on \1'),
    (re.compile(r'\bchecks/reference_deviations\.csv'), "the database's reference checks"),
    (re.compile(r'\bchecks/summary_by_source\.csv'), "the database's per-source check summary"),
    (re.compile(r'\bchecks/build_report\.json'), "the database's build report"),
    (re.compile(r'\b(?:fitting_data/)?transcribed/_curation/_overrides\.csv'), "the database's curation overrides"),
    (re.compile(r'\b(?:fitting_data/)?transcribed/_curation/[^\s);,]*'), "the database's curation files"),
    (re.compile(r'\b(?:fitting_data/)?transcribed/_logs/[^\s);,]*'), "the database's transcription log"),
    (re.compile(r'\b(?:fitting_data/)?transcribed/[^\s);,]*'), "the database's transcription file"),
    # "(wagner2011equations, Table 5, on disk at project/references/papers/x.pdf)"
    (re.compile(r',? on disk at [^);]*'), ''),
    # "(references/supplements/shilling2006measurements_si2.txt)"
    (re.compile(r'\((?:[\w.-]+/)?references/supplements/[^)]*\)'), "(the authors' supplementary file)"),
    # An absolute path to a file: keep the file name only.
    (re.compile(r'(?:/Users/|/home/)[^)\n;]*?/([^/)\n;]+\.(?:pdf|txt|csv|xlsx?|docx?|mat|dat|md))'), r'\1'),
    # Any other path into the local reference library: keep the last component.
    (re.compile(r'(?:[\w.-]+/)?references/(?:papers|supplements)/(?:[^\s);,/]+/)*([^\s);,/]+)'), r'\1'),
]


def _rewrite_text(s):
    for pat, rep in _TEXT_REWRITES:
        s = pat.sub(rep, s)
    return s


def _cite_for_unlisted_key(key, cites):
    """A short citation for a supplement whose own work is not a source key (e.g. a compilation):
    the one listed source with the same first author and year, else 'Author (year)' from the key."""
    m = re.match(r'([a-z]+)(\d{4})', key)
    if not m:
        return key
    same = [v for k, v in cites.items() if k.startswith(m.group(0))]
    if len(same) == 1:
        return same[0]
    return f'{m.group(1).capitalize()} et al. ({m.group(2)})'


def public_via(via, cites):
    """Turn the local path(s) in ``via`` into the name of the published file they point to.

    ``project/references/holten_2014_supplementary_material/Data tables/rho-X.txt``
        -> ``Holten et al. (2014) supplementary material: Data tables/rho-X.txt``
    ``references/supplements/bollengier2019thermodynamics_si/SM_D_data_eos_scripts/WaterEOS.mat``
        -> ``Bollengier et al. (2019) supplementary material: SM_D_data_eos_scripts/WaterEOS.mat``
    ``fitting_data/transcribed/G1/x.csv`` -> '' (the database's own transcription; ``extraction`` says so)
    """
    out = []
    for part in (via or '').split(';'):
        p = part.strip().replace('\\', '/')
        if not p:
            continue
        if '/transcribed/' in f'/{p}' or p.startswith('transcribed/'):
            continue
        m = re.search(r'holten_2014_supplementary_material/(.*)$', p)
        if m:
            label = f"{cites.get('holten2014equation') or 'Holten, Sengers & Anisimov (2014)'} supplementary material"
            out.append(f'{label}: {m.group(1)}' if m.group(1) else label)
            continue
        m = re.search(r'references/supplements/([^/]+)(?:/(.*))?$', p)
        if m:
            first, rest = m.group(1), (m.group(2) or '')
            key = re.sub(r'_si\d*$', '', re.sub(r'\.[A-Za-z0-9]+$', '', first))
            who = cites.get(key) or _cite_for_unlisted_key(key, cites)
            label = f'{who} supplementary material'
            out.append(f'{label}: {rest}' if rest else label)
            continue
        if '/' in p:
            # Unknown local layout: keep only the file name, so no directory leaks.
            out.append(p.rsplit('/', 1)[1])
            continue
        out.append(p)
    seen = []
    for o in out:
        if o not in seen:
            seen.append(o)
    return '; '.join(seen)


# ---------------------------------------------------------------------------
# Build the published rows
# ---------------------------------------------------------------------------
def build(source: Path, forbid=()):
    t0 = time.time()
    log(f'Reading {source}')
    files, src_path, blobs, newest_ns = _read_snapshot(source)

    _, src_rows = _parse_csv(blobs[src_path], 'sources.csv')
    sources = {}
    for s in src_rows:
        key = (s.get('source_key') or '').strip()
        if not key:
            continue
        y = (s.get('year') or '').strip()
        year = int(y) if y.isdigit() else None
        sources[key] = {
            'source_key': key,
            'citation': (s.get('citation') or '').strip(),
            'year': year,
            'doi': (s.get('doi') or '').strip(),
            'used_in_fits': (s.get('used_in_fits') or '').strip(),
        }
        sources[key]['cite_short'] = short_cite(sources[key]['citation'], year)
    cites = {k: v['cite_short'] for k, v in sources.items()}
    # Holten 2014's supplement is keyed by its own source key in the catalog.
    for k, v in sources.items():
        if k.startswith('holten2014'):
            cites.setdefault('holten2014equation', v['cite_short'])

    rows, ids, missing_src = [], set(), set()
    per_table = {}
    for grp, path in files:
        name = f'{grp}/{path.name}'
        header, raw = _parse_csv(blobs[path], name)
        if not raw:
            log(f'  {name:42s}      0 rows (empty, skipped)')
            continue
        missing = [c for c in S.SOURCE_COLUMNS if c not in header]
        if missing:
            raise LoadError(f'{name}: missing columns {missing} (has the schema changed? update '
                            'watereos_api/measured/schema.py)')
        extra = [c for c in header if c not in S.SOURCE_COLUMNS]
        if extra:
            log(f'  warning: {name} has columns not in the published schema, left out: {extra}')
        label = f'{grp}/{path.stem}'
        n0 = len(rows)
        for r in raw:
            rid = r['id'].strip()
            if not rid:
                raise LoadError(f'{name}: a row has no id')
            if rid in ids:
                raise LoadError(f'{name}: duplicate id {rid}')
            ids.add(rid)
            if r['substance'] not in SUBSTANCES:
                raise LoadError(f'row {rid}: substance {r["substance"]!r} is not H2O or D2O')
            if r['u_type'] not in U_TYPES:
                raise LoadError(f'row {rid}: u_type {r["u_type"]!r} not in {sorted(U_TYPES)}')
            out = {}
            for c in S.SOURCE_COLUMNS:
                if c == 'table':
                    continue
                v = r[c]
                if c in S.REAL_SOURCE_COLUMNS:
                    out[c] = _num(v, c, rid)
                else:
                    out[c] = _rewrite_text(v.strip())
            out['id'] = rid
            out['via'] = public_via(r['via'], cites)
            out['tbl'] = r['table'].strip()
            out['table_label'] = label
            out['grp'] = grp
            out['u_std'] = standard_uncertainty(out['value'], out['u'], out['u_type'], out['u_k'])
            out['rate_K_min'] = rate_K_per_min(r['rate'], r['rate_unit'])
            s = sources.get(out['source_key'])
            if s is None:
                missing_src.add(out['source_key'])
            out['year'] = s['year'] if s else None
            rows.append(out)
        tables_here = sorted({x['tbl'] for x in rows[n0:]})
        if len(tables_here) != 1:
            raise LoadError(f'{name}: rows name {len(tables_here)} tables {tables_here}; expected one')
        per_table[label] = len(rows) - n0
        log(f'  {name:42s} {len(rows) - n0:6d} rows')

    if missing_src:
        log(f'  warning: {len(missing_src)} source key(s) used by rows are not in sources.csv: '
            f'{sorted(missing_src)[:10]}')

    # Nothing private may remain in any published text.
    extra = re.compile('|'.join(re.escape(t) for t in forbid if t), re.I) if any(forbid) else None
    leaks = []
    text_cols = [c for c in S.ROW_COLUMNS if S.ROW_COLUMN_TYPES[c] == 'TEXT']
    for r in rows:
        for c in text_cols:
            v = r.get(c) or ''
            m = _LEFTOVER.search(v) or (extra.search(v) if extra else None)
            if m:
                leaks.append((r['id'], c, v[max(0, m.start() - 30):m.end() + 40]))
    if leaks:
        detail = '\n'.join(f'    {i} [{c}]: ...{t}...' for i, c, t in leaks[:15])
        raise LoadError(f'{len(leaks)} published text value(s) still contain a local path:\n{detail}\n'
                        'Add a rule in watereos_api/measured/load.py (public_via or _TEXT_REWRITES).')

    used = {}
    tables_of = {}
    extraction_of = {}
    for r in rows:
        k = r['source_key']
        used[k] = used.get(k, 0) + 1
        tables_of.setdefault(k, {}).setdefault(r['table_label'], 0)
        tables_of[k][r['table_label']] += 1
        extraction_of.setdefault(k, set()).add(r['extraction'])
    pub_sources = []
    for k in sorted(used):
        s = sources.get(k, {'source_key': k, 'citation': '', 'year': None, 'doi': '', 'used_in_fits': '',
                            'cite_short': k})
        pub_sources.append({
            **s,
            'n_rows': used[k],
            'tables': '; '.join(f'{t} ({n})' for t, n in sorted(tables_of[k].items())),
            'extraction': '; '.join(sorted(e for e in extraction_of[k] if e)),
        })

    h = hashlib.sha256()
    for p in sorted(blobs, key=str):
        h.update(str(p.relative_to(source)).encode())
        h.update(blobs[p])
    digest = h.hexdigest()
    now = datetime.now(timezone.utc)
    release = {
        'release_id': f'{now:%Y-%m-%d}-{digest[:8]}',
        'created_at': now.isoformat(timespec='seconds'),
        'source_modified_at': datetime.fromtimestamp(newest_ns / 1e9, timezone.utc).isoformat(timespec='seconds'),
        'n_rows': len(rows),
        'n_sources': len(pub_sources),
        'n_tables': len(per_table),
        'inputs_sha256': digest,
        'schema_version': S.SCHEMA_VERSION,
    }
    log(f'Built {len(rows)} rows in {len(per_table)} tables from {len(pub_sources)} sources '
        f'({time.time() - t0:.1f} s); release {release["release_id"]}')
    return rows, pub_sources, release


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------
def _row_tuple(r):
    return tuple(r.get(c) for c in S.ROW_COLUMNS)


def _source_tuple(s):
    return tuple(s.get(c) for c, _ in S.SOURCE_TABLE_COLUMNS)


def _release_tuple(rel):
    return tuple(rel[c] for c, _ in S.RELEASE_COLUMNS)


def write_sqlite(path: Path, rows, sources, release):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f'{path.name}.tmp-{os.getpid()}')
    if tmp.exists():
        tmp.unlink()
    conn = dbmod.connect(dbmod.Target('sqlite', path=str(tmp)), readonly=False)
    try:
        for stmt in S.ddl():
            conn.execute(stmt)
        cols = ', '.join(S.ROW_COLUMNS)
        qs = ', '.join('?' * len(S.ROW_COLUMNS))
        conn.executemany(f'INSERT INTO {S.TABLE_ROWS} ({cols}) VALUES ({qs})', (_row_tuple(r) for r in rows))
        scols = [c for c, _ in S.SOURCE_TABLE_COLUMNS]
        conn.executemany(f'INSERT INTO {S.TABLE_SOURCES} ({", ".join(scols)}) VALUES ({", ".join("?" * len(scols))})',
                         (_source_tuple(s) for s in sources))
        rcols = [c for c, _ in S.RELEASE_COLUMNS]
        conn.execute(f'INSERT INTO {S.TABLE_RELEASE} ({", ".join(rcols)}) VALUES ({", ".join("?" * len(rcols))})',
                     _release_tuple(release))
        conn.commit()
    finally:
        conn.close()
    os.replace(tmp, path)


def write_postgres(target, rows, sources, release):
    conn = dbmod.connect(target, readonly=False)
    try:
        with conn.transaction():
            cur = conn.cursor()
            for t in (S.TABLE_ROWS, S.TABLE_SOURCES, S.TABLE_RELEASE):
                cur.execute(f'DROP TABLE IF EXISTS {t}')
            for stmt in S.ddl():
                cur.execute(stmt)
            log(f'  writing {len(rows)} rows')
            with cur.copy(f'COPY {S.TABLE_ROWS} ({", ".join(S.ROW_COLUMNS)}) FROM STDIN') as cp:
                for r in rows:
                    cp.write_row(_row_tuple(r))
            scols = [c for c, _ in S.SOURCE_TABLE_COLUMNS]
            with cur.copy(f'COPY {S.TABLE_SOURCES} ({", ".join(scols)}) FROM STDIN') as cp:
                for s in sources:
                    cp.write_row(_source_tuple(s))
            rcols = [c for c, _ in S.RELEASE_COLUMNS]
            cur.execute(f'INSERT INTO {S.TABLE_RELEASE} ({", ".join(rcols)}) '
                        f'VALUES ({", ".join(["%s"] * len(rcols))})', _release_tuple(release))
    finally:
        conn.close()


def write(target, rows, sources, release):
    t0 = time.time()
    log(f'Writing to {target.describe()}')
    if target.kind == 'sqlite':
        write_sqlite(Path(target.path), rows, sources, release)
    else:
        write_postgres(target, rows, sources, release)
    log(f'Done in {time.time() - t0:.1f} s: release {release["release_id"]}, {release["n_rows"]} rows, '
        f'{release["n_sources"]} sources')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--source', required=True, type=Path, help='the fitting_data folder')
    ap.add_argument('--db', help='database URL (default: $WATEREOS_DATA_URL, $DATABASE_URL, '
                                 f'then {DEFAULT_SQLITE.relative_to(REPO_ROOT)})')
    ap.add_argument('--dry-run', action='store_true', help='read and check only; write nothing')
    ap.add_argument('--forbid', action='append', default=[], metavar='TERM',
                    help='refuse to load if TERM (e.g. a local folder name) appears in any published text; '
                         'repeatable, case-insensitive')
    a = ap.parse_args(argv)
    try:
        rows, sources, release = build(a.source.resolve(), forbid=a.forbid)
        if a.dry_run:
            log('Dry run: nothing written.')
            return 0
        url = a.db or dbmod.url_from_env() or f'sqlite:///{DEFAULT_SQLITE}'
        write(dbmod.parse_url(url), rows, sources, release)
    except LoadError as e:
        print(f'error: {e}', file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
