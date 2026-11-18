"""Read-only API for the phase diagrams of water + NaCl (``/api/atlas/...``).

GET /atlas/manifest            the release, the solutes, groups, views, captions and color key
GET /atlas/{solute}/{name}     one stored file: a figure (``d-T-xv``, ``d-T-xv-log``, ...) or
                               the system notes (``info``), as Plotly/JSON

The files are stored gzip-compressed and sent as they are to clients that
accept gzip (nearly all), so the server neither computes nor recompresses.
"""
from __future__ import annotations

import gzip
import hashlib
import re

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, Response

from watereos_api.atlas import schema as A
from watereos_api.atlas.store import NotConfigured, Unavailable, get_store
from watereos_api.measured.store import dumps

router = APIRouter(prefix='/atlas')

CACHE_SECONDS = 300
_NAME = re.compile(A.NAME_PATTERN)


def _not_modified(request: Request, etag: str) -> bool:
    return request.headers.get('if-none-match') == etag


@router.get('/manifest')
def manifest(request: Request):
    st = get_store()
    try:
        rel = st.release()
    except NotConfigured:
        return JSONResponse({'configured': False, 'detail': 'The phase diagrams are not configured on this server.'})
    except Unavailable as e:
        return JSONResponse({'configured': False, 'detail': str(e)}, status_code=503)
    body = dumps({'configured': True, 'release_id': rel['release_id'], 'created_at': rel['created_at'],
                  **rel['manifest']})
    etag = '"' + hashlib.sha256(body).hexdigest()[:20] + '"'
    headers = {'ETag': etag, 'Cache-Control': f'public, max-age={CACHE_SECONDS}'}
    if _not_modified(request, etag):
        return Response(status_code=304, headers=headers)
    return Response(content=body, media_type='application/json', headers=headers)


@router.get('/{solute}/{name}')
def atlas_file(solute: str, name: str, request: Request):
    if not (_NAME.match(solute) and _NAME.match(name)) or len(solute) > 64 or len(name) > 64:
        raise HTTPException(status_code=404, detail='no such file')
    st = get_store()
    try:
        found = st.file(solute, name)
    except NotConfigured as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except Unavailable as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    if found is None:
        raise HTTPException(status_code=404, detail=f'no file {solute}/{name}')
    sha, gz = found
    etag = '"' + sha[:20] + '"'
    headers = {'ETag': etag, 'Cache-Control': f'public, max-age={CACHE_SECONDS}', 'Vary': 'Accept-Encoding'}
    if _not_modified(request, etag):
        return Response(status_code=304, headers=headers)
    if 'gzip' in request.headers.get('accept-encoding', '').lower():
        return Response(content=gz, media_type='application/json', headers={**headers, 'Content-Encoding': 'gzip'})
    return Response(content=gzip.decompress(gz), media_type='application/json', headers=headers)
