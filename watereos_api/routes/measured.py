"""Read-only API for the measured water data (``/api/data/...``).

GET  /data/release         release, tables and properties, citation and notice
GET  /data/points          plotting columns for one table/property, or all rows
GET  /data/rows/{id}       every published column of one row, with its source
GET  /data/sources         the cited studies
GET  /data/download        the whole release as a zip (CSV + sources + README)
GET  /data/lines           melting, homogeneous-nucleation and vapor-pressure lines
POST /data/model-curves    model isobars or isotherms over a property
POST /data/model-surface   a model surface over T and P (3D view)
"""
from __future__ import annotations

import csv
import hashlib
import io
import math
import zipfile
from typing import Literal, Optional

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, ConfigDict, Field, conlist

from watereos_api.measured import curves
from watereos_api.measured import schema as S
from watereos_api.measured.store import NotConfigured, Unavailable, dumps, get_store

router = APIRouter(prefix='/data')

# Columns sent for plotting, filtering and hover. Everything else comes from /rows/{id}.
POINT_COLUMNS = [
    'id', 'tbl', 'property', 'grp', 'T_K', 'P_MPa', 'value', 'unit', 'u_std', 'u_T_K', 'u_P_MPa', 'substance',
    'state', 'state_final', 'ice_phase', 'sample', 'method', 'data_type', 'extraction', 'verification',
    'qc_flag', 'source_key', 'year', 'used_in_fits', 'used_in_fits_level', 'definition', 'rate_K_min',
    'location',
]
CACHE_SECONDS = 300


def _store():
    st = get_store()
    try:
        st.release()
    except NotConfigured as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except Unavailable as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    return st


def _respond(request: Request, body: bytes, media_type='application/json', headers=None):
    # The ETag follows the content, not just the release: a deploy that changes
    # a label or a payload field without a new data release must not leave
    # browsers revalidating their old copy with a 304.
    etag = '"' + hashlib.sha256(body).hexdigest()[:20] + '"'
    h = {'ETag': etag, 'Cache-Control': f'public, max-age={CACHE_SECONDS}'}
    h.update(headers or {})
    if request.headers.get('if-none-match') == etag:
        return Response(status_code=304, headers=h)
    return Response(content=body, media_type=media_type, headers=h)


def _in_order(cols, rows, *keys):
    """Rows sorted by code point on the given columns, whatever the database's collation.

    SQL ORDER BY follows the database's collation: PostgreSQL with en_US.utf8
    puts 'melting-Ih' before 'TMiceI', SQLite (and Python) the other way. Sorting
    here keeps every answer and the download identical on both.
    """
    at = [cols.index(k) for k in keys]
    return sorted(rows, key=lambda r: tuple('' if r[i] is None else r[i] for i in at))


def _cached(st, key, producer):
    try:
        return st.cached(key, producer)
    except Unavailable as e:
        raise HTTPException(status_code=503, detail=str(e)) from e


# ---------------------------------------------------------------------------
# Release and tables
# ---------------------------------------------------------------------------
def _release_payload(st):
    rel = st.release()
    _, rows = st.query(
        f'SELECT tbl, table_label, grp, property, unit, COUNT(*), MIN(value), MAX(value), '
        f'SUM(CASE WHEN value <= 0 THEN 1 ELSE 0 END), '
        f"SUM(CASE WHEN T_K IS NOT NULL AND P_MPa IS NOT NULL THEN 1 ELSE 0 END) "
        f'FROM {S.TABLE_ROWS} GROUP BY tbl, table_label, grp, property, unit')
    _, years = st.query(f'SELECT MIN(year), MAX(year) FROM {S.TABLE_ROWS}')
    tables = {}
    for tbl, label, grp, prop, unit, n, vmin, vmax, nonpos, n_tp in rows:
        t = tables.setdefault(tbl, {'key': tbl, 'label': label, 'group': grp, 'n': 0, 'properties': []})
        t['n'] += n
        log = prop in S.LOG_DEFAULT or bool(
            vmin is not None and vmax is not None and not nonpos and vmin > 0 and math.log10(vmax / vmin) >= 3)
        spec = S.overlay_attr(grp, prop)
        t['properties'].append({
            'key': prop, 'label': S.property_label(prop), 'unit': unit, 'n': n, 'n_tp': n_tp,
            'locus': S.locus_kind(prop), 'log': log,
            'models': [{'key': k, 'label': lab} for k, lab in curves.overlay_models(prop)] if spec else [],
        })
    order = {g: i for i, g in enumerate(S.GROUP_KEYS)}
    tl = sorted(tables.values(), key=lambda t: (order.get(t['group'], 99), t['label']))
    for t in tl:
        t['properties'].sort(key=lambda p: (-p['n'], p['key']))
    rid = rel['release_id']
    return dumps({
        'configured': True,
        'release': rel,
        'groups': [{'key': k, 'label': lab, 'thermodynamic': th} for k, lab, th in S.GROUPS
                   if any(t['group'] == k for t in tl)],
        'tables': tl,
        'years': list(years[0]) if years else [None, None],
        'notice': S.NOTICE,
        'cite': S.CITE_WATEREOS.format(release_id=rid),
    })


@router.get('/release')
def release(request: Request):
    st = get_store()
    try:
        st.release()
    except NotConfigured:
        return JSONResponse({'configured': False, 'detail': 'Measured data are not configured on this server.'})
    except Unavailable as e:
        return JSONResponse({'configured': False, 'detail': str(e)}, status_code=503)
    _, body = _cached(st, 'release', lambda: _release_payload(st))
    return _respond(request, body)


# ---------------------------------------------------------------------------
# Points and rows
# ---------------------------------------------------------------------------
def _points_payload(st, table, prop):
    where, params = [], []
    if table:
        where.append('tbl = ?')
        params.append(table)
    if prop:
        where.append('property = ?')
        params.append(prop)
    clause = f' WHERE {" AND ".join(where)}' if where else ''
    cols, rows = st.query(f'SELECT {", ".join(POINT_COLUMNS)} FROM {S.TABLE_ROWS}{clause}', params)
    rows = _in_order(cols, rows, 'id')
    columns = {c: [r[i] for r in rows] for i, c in enumerate(cols)}
    columns['table'] = columns.pop('tbl')
    columns['group'] = columns.pop('grp')
    return dumps({'release_id': st.release()['release_id'], 'n': len(rows), 'columns': columns})


@router.get('/points')
def points(request: Request,
           table: Optional[str] = Query(None, max_length=80),
           property: Optional[str] = Query(None, max_length=80)):
    st = _store()
    key = ('points', table, property)
    _, body = _cached(st, key, lambda: _points_payload(st, table, property))
    return _respond(request, body)


@router.get('/rows/{row_id}')
def row(row_id: str):
    st = _store()
    cols, rows = st.query(f'SELECT {", ".join(S.ROW_COLUMNS)} FROM {S.TABLE_ROWS} WHERE id = ?', [row_id])
    if not rows:
        raise HTTPException(status_code=404, detail=f'no row {row_id!r}')
    rec = dict(zip(cols, rows[0]))
    rec['table'] = rec.pop('tbl')
    rec['group'] = rec.pop('grp')
    scols, srows = st.query(f'SELECT {", ".join(c for c, _ in S.SOURCE_TABLE_COLUMNS)} FROM {S.TABLE_SOURCES} '
                            f'WHERE source_key = ?', [rec['source_key']])
    return {'row': rec, 'source': dict(zip(scols, srows[0])) if srows else None,
            'property_label': S.property_label(rec['property'])}


def _sources_payload(st):
    cols, rows = st.query(f'SELECT {", ".join(c for c, _ in S.SOURCE_TABLE_COLUMNS)} FROM {S.TABLE_SOURCES}')
    rows = _in_order(cols, rows, 'source_key')
    return dumps({'release_id': st.release()['release_id'], 'sources': [dict(zip(cols, r)) for r in rows]})


@router.get('/sources')
def sources(request: Request):
    st = _store()
    _, body = _cached(st, 'sources', lambda: _sources_payload(st))
    return _respond(request, body)


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------
README_TEMPLATE = """Measured water data, release {release_id}
{rule}

{n_rows} rows from {n_sources} published studies, in {n_tables} tables.
Compiled {created_at} from the waterEoS measured-data database.

{notice}

How to cite
-----------
Cite the original studies (sources.csv: citation and DOI; each row names its
source in source_key and where in the study the value is in "location"), and
cite waterEoS for the compilation:

  {cite}

Files
-----
measured_water_data.csv  one row per value. Units are SI as in the "unit"
                         column; T_K in kelvin (ITS-90), P_MPa in MPa.
                         u is the uncertainty as stated (u_type, u_k);
                         u_std is the absolute standard uncertainty (k = 1).
                         data_type tells measured values from values derived
                         by the authors, smoothed tables, correlations,
                         quoted values and bounds; extraction says whether a
                         value came from a supplied file, a transcribed table
                         or text, or a digitized figure (with its reading
                         error in u_digitization). Kinetic, dynamics and
                         structure tables are not thermodynamic data.
sources.csv              the studies, with citation, DOI and row counts.
"""


def _download_payload(st):
    rel = st.release()
    cols, rows = st.query(f'SELECT {", ".join(S.ROW_COLUMNS)} FROM {S.TABLE_ROWS}')
    rows = _in_order(cols, rows, 'grp', 'tbl', 'id')
    header = ['table' if c == 'tbl' else 'group' if c == 'grp' else c for c in cols]
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator='\n')
    w.writerow(header)
    for r in rows:
        w.writerow(['' if v is None else repr(v) if isinstance(v, float) else v for v in r])
    scols, srows = st.query(f'SELECT {", ".join(c for c, _ in S.SOURCE_TABLE_COLUMNS)} FROM {S.TABLE_SOURCES}')
    srows = _in_order(scols, srows, 'source_key')
    sbuf = io.StringIO()
    sw = csv.writer(sbuf, lineterminator='\n')
    sw.writerow(scols)
    for r in srows:
        sw.writerow(['' if v is None else v for v in r])
    readme = README_TEMPLATE.format(
        rule='=' * (32 + len(rel['release_id'])), notice=S.NOTICE,
        cite=S.CITE_WATEREOS.format(release_id=rel['release_id']), **rel)
    out = io.BytesIO()
    with zipfile.ZipFile(out, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        z.writestr('README.txt', readme)
        z.writestr('measured_water_data.csv', buf.getvalue())
        z.writestr('sources.csv', sbuf.getvalue())
    return out.getvalue()


@router.get('/download')
def download(request: Request):
    st = _store()
    rid, body = _cached(st, 'download', lambda: _download_payload(st))
    name = f'watereos_measured_data_{rid}.zip'
    return _respond(request, body, media_type='application/zip',
                    headers={'Content-Disposition': f'attachment; filename="{name}"'})


# ---------------------------------------------------------------------------
# Reference lines and model overlays
# ---------------------------------------------------------------------------
@router.get('/lines')
def lines():
    return JSONResponse(curves.reference_lines(), headers={'Cache-Control': 'public, max-age=86400'})


class _Strict(BaseModel):
    model_config = ConfigDict(extra='forbid')


class ModelCurvesRequest(_Strict):
    property: str = Field(max_length=80)
    models: conlist(str, min_length=1, max_length=12)
    along: Literal['T', 'P']
    fixed: conlist(float, min_length=1, max_length=16)
    lo: float
    hi: float
    n: int = Field(160, ge=10, le=600)


class ModelSurfaceRequest(_Strict):
    property: str = Field(max_length=80)
    model: str = Field(max_length=80)
    T_range: conlist(float, min_length=2, max_length=2)
    P_range: conlist(float, min_length=2, max_length=2)
    n: int = Field(60, ge=10, le=120)


def _check_overlay(prop, models):
    allowed = dict(curves.overlay_models(prop))
    if not allowed:
        raise HTTPException(status_code=422, detail=f'no model overlay for property {prop!r}')
    bad = [m for m in models if m not in allowed]
    if bad:
        raise HTTPException(status_code=422, detail=f'models without {prop!r}: {bad}')


def _check_range(lo, hi, what, lo_lim, hi_lim):
    if not (math.isfinite(lo) and math.isfinite(hi) and lo < hi):
        raise HTTPException(status_code=422, detail=f'{what} range must be finite and increasing')
    if lo < lo_lim or hi > hi_lim:
        raise HTTPException(status_code=422, detail=f'{what} range must lie within [{lo_lim}, {hi_lim}]')


T_LIMITS = (1.0, 2000.0)
P_LIMITS = (-1000.0, 100000.0)


@router.post('/model-curves')
def model_curves(req: ModelCurvesRequest):
    _check_overlay(req.property, req.models)
    if req.along == 'T':
        _check_range(req.lo, req.hi, 'T', *T_LIMITS)
    else:
        _check_range(req.lo, req.hi, 'P', *P_LIMITS)
    lo, hi = round(req.lo, 6), round(req.hi, 6)
    out = []
    for m in req.models:
        for f in req.fixed:
            out.append(curves.model_curve(m, req.property, req.along, round(float(f), 6), lo, hi, req.n))
    return {'curves': out}


@router.post('/model-surface')
def model_surface(req: ModelSurfaceRequest):
    _check_overlay(req.property, [req.model])
    _check_range(*req.T_range, 'T', *T_LIMITS)
    _check_range(*req.P_range, 'P', *P_LIMITS)
    return curves.model_surface(
        req.model, req.property, round(req.T_range[0], 6), round(req.T_range[1], 6),
        round(req.P_range[0], 6), round(req.P_range[1], 6), req.n)
