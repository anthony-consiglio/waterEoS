"""Atlas loader and /api/atlas endpoints (the H2O Phase Diagram tab).

Everything runs on a small, made-up atlas in the layout cryo-isochoric's
``gallery build`` writes (index.html with the manifest, data/<solute>/*.js),
in a temporary folder; the API tests use SQLite.
"""
import gzip
import json
import sqlite3

import pytest
from fastapi.testclient import TestClient

from watereos_api.app import create_app
from watereos_api.atlas import load as L
from watereos_api.atlas import schema as A
from watereos_api.atlas.store import reset_store

MANIFEST = {
    'solutes': [{'id': 'nacl', 'key': 'nacl', 'name': 'Sodium chloride', 'electrolyte': True,
                 'W_fit': 0.22, 'M': 58.443, 'rho': 2.165}],
    'built': ['nacl'], 'server': False, 'quality': 'draft',
    'key': {'fill': ['#a', '#b', '#c'], 'surf': ['#d', '#e', '#f'], 'outside': '#g', 'tie': '#h'},
    'groups': [{'id': 'process', 'title': 'Process', 'short': 'Process'},
               {'id': 'd', 'title': 'Phase diagrams (2-D sections)', 'short': 'Phase diagrams'},
               {'id': 's3', 'title': 'Sections as surfaces', 'short': 'Surfaces'}],
    'views': [{'id': 'paths', 'group': 'process', 'title': 'Isochoric chamber', 'caption': 'c', 'log': None},
              {'id': 'd-T-xv', 'group': 'd', 'title': 'Isothermal · x – v', 'caption': 'c', 'log': 'log v'},
              {'id': 's3-T', 'group': 's3', 'title': 'Isothermal surface  p(x, v)', 'caption': 'c', 'log': None}],
    'stamp': '2026-01-01 00:00',
}


def _figure(name='L', **extra):
    return {'data': [{'type': 'scatter', 'name': name, 'x': [0, 1], 'y': [0.9, 1.0]}],
            'layout': {'meta': {'comp': {'axis': 'x'}}, **extra}}


def _write_atlas(root, figures=None, info=None, manifest=None):
    """A built atlas: index.html and data/nacl/*.js, as cryo-isochoric writes them."""
    m = manifest or MANIFEST
    root.mkdir(parents=True, exist_ok=True)
    (root / 'index.html').write_text(f'<script>\nconst M = {json.dumps(m)};\nconst ATLAS = {{}};\n</script>\n')
    d = root / 'data' / 'nacl'
    d.mkdir(parents=True, exist_ok=True)
    figures = figures if figures is not None else {
        'd-T-xv': _figure(), 'd-T-xv-log': _figure('log'), 's3-T': _figure('surface'), 'paths': _figure('chamber')}
    for name, fig in figures.items():
        body = fig if isinstance(fig, str) else json.dumps(fig)
        (d / f'{name}.js').write_text(f'ATLAS.register("nacl/{name}", {body});\n')
    info = info if info is not None else {'id': 'nacl', 'name': 'Sodium chloride', 'points': []}
    (d / 'info.js').write_text(f'ATLAS.info("nacl", {json.dumps(info)});\n')
    (d / 'version.txt').write_text('9 draft')
    return root


@pytest.fixture()
def atlas(tmp_path):
    return _write_atlas(tmp_path / 'atlas')


def _tables(db):
    with sqlite3.connect(db) as c:
        return {r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------
def test_load_writes_the_views_without_the_process_view(atlas, tmp_path):
    db = tmp_path / 'a.sqlite'
    assert L.main(['--source', str(atlas), '--db', f'sqlite:///{db}']) == 0
    with sqlite3.connect(db) as c:
        rows = {(s, n): (body, sha, size) for s, n, body, sha, size in c.execute(f'SELECT * FROM {A.TABLE_FILES}')}
        rel = dict(zip(A.RELEASE_COLUMNS, c.execute(f'SELECT * FROM {A.TABLE_RELEASE}').fetchone()))
    assert set(rows) == {('nacl', 'd-T-xv'), ('nacl', 'd-T-xv-log'), ('nacl', 's3-T'), ('nacl', 'info')}
    body, sha, size = rows[('nacl', 'd-T-xv')]
    raw = gzip.decompress(body)
    assert json.loads(raw) == _figure() and len(raw) == size
    m = json.loads(rel['manifest'])
    assert [g['id'] for g in m['groups']] == ['d', 's3']
    assert [v['id'] for v in m['views']] == ['d-T-xv', 's3-T']
    assert m['solutes'][0]['id'] == 'nacl' and m['builds'] == {'nacl': '9 draft'}
    assert rel['schema_version'] == A.SCHEMA_VERSION and rel['release_id'].count('-') == 3


def test_missing_view_fails_and_writes_nothing(atlas, tmp_path):
    (atlas / 'data' / 'nacl' / 'd-T-xv-log.js').unlink()
    db = tmp_path / 'a.sqlite'
    assert L.main(['--source', str(atlas), '--db', f'sqlite:///{db}']) == 1
    assert not db.exists()


@pytest.mark.parametrize('bad', [
    {'d-T-xv': _figure('/Users/someone/atlas/d-T-xv')},            # a local path in a trace name
    {'d-T-xv': '{"data": [], "layout": {"x": NaN}}'},               # not JSON
    {'d-T-xv': {'data': 'nope'}},                                   # not a figure
])
def test_refuses_unpublishable_files(tmp_path, bad):
    figures = {'d-T-xv': _figure(), 'd-T-xv-log': _figure(), 's3-T': _figure(), **bad}
    root = _write_atlas(tmp_path / 'atlas', figures=figures)
    with pytest.raises(L.LoadError):
        L.read_atlas(root)


def test_base64_data_is_not_scanned_for_paths(tmp_path):
    fig = _figure()
    fig['data'][0]['x'] = {'dtype': 'f4', 'bdata': 'AAA/home/AAA='}
    root = _write_atlas(tmp_path / 'atlas', figures={'d-T-xv': fig, 'd-T-xv-log': _figure(), 's3-T': _figure()})
    site, files = L.read_atlas(root)
    assert len(files) == 4


def test_forbidden_term_and_misnamed_file(atlas, tmp_path):
    with pytest.raises(L.LoadError, match='forbidden'):
        L.read_atlas(atlas, forbid=('Sodium',))
    (atlas / 'data' / 'nacl' / 's3-T.js').write_text(f'ATLAS.register("nacl/d-T-xv", {json.dumps(_figure())});\n')
    with pytest.raises(L.LoadError):
        L.read_atlas(atlas)


def test_server_shell_is_not_a_build(tmp_path):
    root = _write_atlas(tmp_path / 'atlas', manifest={**MANIFEST, 'server': True})
    with pytest.raises(L.LoadError, match='server'):
        L.read_atlas(root)


def test_reload_replaces_only_the_atlas_tables(atlas, tmp_path):
    db = tmp_path / 'shared.sqlite'
    with sqlite3.connect(db) as c:
        c.execute('CREATE TABLE measured_release (release_id TEXT)')
        c.execute("INSERT INTO measured_release VALUES ('m1')")
    assert L.main(['--source', str(atlas), '--db', f'sqlite:///{db}']) == 0
    _write_atlas(atlas, figures={'d-T-xv': _figure('new'), 'd-T-xv-log': _figure(), 's3-T': _figure()})
    assert L.main(['--source', str(atlas), '--db', f'sqlite:///{db}']) == 0
    assert _tables(db) >= {'measured_release', A.TABLE_FILES, A.TABLE_RELEASE}
    with sqlite3.connect(db) as c:
        assert c.execute('SELECT release_id FROM measured_release').fetchall() == [('m1',)]
        assert c.execute(f'SELECT COUNT(*) FROM {A.TABLE_RELEASE}').fetchone()[0] == 1
        body = c.execute(f"SELECT body FROM {A.TABLE_FILES} WHERE name = 'd-T-xv'").fetchone()[0]
    assert json.loads(gzip.decompress(body))['data'][0]['name'] == 'new'


def test_dry_run_writes_nothing(atlas, tmp_path):
    db = tmp_path / 'a.sqlite'
    assert L.main(['--source', str(atlas), '--db', f'sqlite:///{db}', '--dry-run']) == 0
    assert not db.exists()


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
@pytest.fixture()
def client(atlas, tmp_path):
    db = tmp_path / 'api.sqlite'
    assert L.main(['--source', str(atlas), '--db', f'sqlite:///{db}']) == 0
    reset_store(f'sqlite:///{db}')
    yield TestClient(create_app())
    reset_store()


def test_manifest(client):
    r = client.get('/api/atlas/manifest')
    assert r.status_code == 200
    d = r.json()
    assert d['configured'] is True and d['release_id'] and d['quality'] == 'draft'
    assert [g['id'] for g in d['groups']] == ['d', 's3'] and 'paths' not in [v['id'] for v in d['views']]
    again = client.get('/api/atlas/manifest', headers={'If-None-Match': r.headers['etag']})
    assert again.status_code == 304


def test_file_gzip_identity_and_conditional_get(client):
    r = client.get('/api/atlas/nacl/d-T-xv', headers={'Accept-Encoding': 'gzip'})
    assert r.status_code == 200 and r.headers['content-encoding'] == 'gzip'
    assert r.json() == _figure() and 'Accept-Encoding' in r.headers['vary']
    plain = client.get('/api/atlas/nacl/d-T-xv', headers={'Accept-Encoding': 'identity'})
    assert 'content-encoding' not in plain.headers and plain.json() == _figure()
    assert plain.headers['etag'] == r.headers['etag']
    again = client.get('/api/atlas/nacl/d-T-xv', headers={'If-None-Match': r.headers['etag']})
    assert again.status_code == 304
    assert client.get('/api/atlas/nacl/info').json()['name'] == 'Sodium chloride'


@pytest.mark.parametrize('path', ['/api/atlas/nacl/paths', '/api/atlas/kcl/info', '/api/atlas/nacl/d-T-xv.js',
                                  '/api/atlas/NaCl/info', '/api/atlas/nacl/' + 'x' * 80])
def test_missing_or_malformed_file_is_404(client, path):
    assert client.get(path).status_code == 404


def test_not_configured(monkeypatch):
    for v in ('WATEREOS_ATLAS_URL', 'WATEREOS_DATA_URL', 'DATABASE_URL'):
        monkeypatch.delenv(v, raising=False)
    reset_store()
    try:
        c = TestClient(create_app())
        r = c.get('/api/atlas/manifest')
        assert r.status_code == 200 and r.json()['configured'] is False
        assert c.get('/api/atlas/nacl/info').status_code == 503
    finally:
        reset_store()


def test_database_without_atlas_is_unavailable(tmp_path):
    db = tmp_path / 'empty.sqlite'
    with sqlite3.connect(db) as c:
        c.execute('CREATE TABLE something (x INTEGER)')
    reset_store(f'sqlite:///{db}')
    try:
        c = TestClient(create_app())
        r = c.get('/api/atlas/manifest')
        assert r.status_code == 503 and r.json()['configured'] is False
        assert 'watereos_api.atlas.load' in r.json()['detail']
        assert c.get('/api/atlas/nacl/info').status_code == 503
    finally:
        reset_store()
