"""Measured-data loader and /api/data endpoints.

Everything runs on a small, made-up copy of the fitting_data layout written to
a temporary folder: no measured value from the real database is in this file.
The API tests use SQLite; the PostgreSQL path shares the same SQL and is
exercised by hand (see docs/measured_data.md).
"""
import csv
import io
import zipfile

import pytest
from fastapi.testclient import TestClient

from watereos_api.app import create_app
from watereos_api.measured import db as dbmod
from watereos_api.measured import load as L
from watereos_api.measured import schema as S
from watereos_api.measured.store import reset_store


def _row(**kw):
    r = {c: '' for c in S.SOURCE_COLUMNS}
    r.update({'u_type': 'none', 'substance': 'H2O', 'data_type': 'measured', 'qc_flag': 'ok',
              'extraction': 'machine-readable file', 'verification': 'supplied file'})
    r.update({k: str(v) for k, v in kw.items()})
    return r


def _write(path, rows, header=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=header or S.SOURCE_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow(r)


SOURCES = [
    {'source_key': 'alpha2001made', 'has_note': 'yes', 'year': '2001', 'doi': '10.0000/alpha',
     'citation': 'A. Alpha, B. Beta (2001). A made-up density study. J. Test. 1, 1.',
     'catalog_status': 'obtained', 'pdf': 'references/papers/alpha2001.pdf', 'n_rows': '4', 'tables': 'density (4)',
     'extraction': 'machine-readable file', 'via': '', 'used_in_fits': 'Holten2014'},
    {'source_key': 'gamma1999made', 'has_note': 'no', 'year': '1999', 'doi': '',
     'citation': 'C. Gamma, D. Delta, E. Epsilon (1999). Made-up melting points. J. Test. 2, 2.',
     'catalog_status': 'obtained', 'pdf': '', 'n_rows': '2', 'tables': 'melting (2)',
     'extraction': 'transcribed table', 'via': '', 'used_in_fits': ''},
    {'source_key': 'holten2014equation', 'has_note': 'yes', 'year': '2014', 'doi': '',
     'citation': 'V. Holten, J. V. Sengers, M. A. Anisimov (2014). Equation of state. J. Phys. Chem. Ref. Data.',
     'catalog_status': 'obtained', 'pdf': '', 'n_rows': '0', 'tables': '', 'extraction': '', 'via': '',
     'used_in_fits': ''},
]


@pytest.fixture()
def source(tmp_path):
    fd = tmp_path / 'fitting_data'
    density = [
        _row(id='density:alpha2001made:s:0001', table='density', property='density', unit='kg/m3',
             value='999.0', T_K='280', P_MPa='0.1', u='0.001', u_type='relative', u_k='2', source_key='alpha2001made',
             via='other-project/references/holten_2014_supplementary_material/Data tables/rho-Alpha.txt',
             used_in_fits='Holten2014', used_in_fits_level='Holten2014:point',
             notes='checked against reading_notes/alpha2001made.md'),
        _row(id='density:alpha2001made:s:0002', table='density', property='density', unit='kg/m3',
             value='1050.0', T_K='260', P_MPa='100', u='0.5', u_type='absolute', source_key='alpha2001made',
             via='references/supplements/alpha2001made_si/tables/T2.txt',
             notes='see /Users/someone/Data/papers/alpha2001.pdf for the scan'),
        _row(id='density:alpha2001made:s:0003', table='density', property='density', unit='kg/m3',
             value='1100.0', T_K='250', P_MPa='200', source_key='alpha2001made', data_type='quoted',
             via='fitting_data/transcribed/A/alpha2001made__density.csv',
             notes='value corrected (transcribed/_curation/_overrides.csv)'),
        _row(id='density:alpha2001made:s:0004', table='density', property='density', unit='kg/m3',
             value='1105.0', T_K='280', P_MPa='0.1', substance='D2O', source_key='alpha2001made',
             rate='10', rate_unit='K/s'),
    ]
    melting = [
        _row(id='melting:gamma1999made:m:0001', table='melting', property='melting_point', unit='K',
             value='260', T_K='260', P_MPa='150', u_T_K='0.1', source_key='gamma1999made', ice_phase='Ih',
             extraction='transcribed table', qc_note='see checks/reference_deviations.csv'),
        _row(id='melting:gamma1999made:m:0002', table='melting', property='melting_point', unit='K',
             value='253', T_K='253', P_MPa='200', source_key='gamma1999made', ice_phase='Ih',
             extraction='transcribed table'),
    ]
    _write(fd / 'liquid' / 'density.csv', density)
    _write(fd / 'phase_equilibrium' / 'melting.csv', melting)
    _write(fd / 'structure' / 'fraction.csv', [])
    _write(fd / 'sources.csv', SOURCES, header=list(SOURCES[0]))
    return fd


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------
def test_build_rows_and_release(source):
    rows, sources, release = L.build(source)
    assert len(rows) == 6
    assert release['n_rows'] == 6 and release['n_tables'] == 2 and release['n_sources'] == 2
    assert release['schema_version'] == S.SCHEMA_VERSION
    assert release['release_id'].endswith(release['inputs_sha256'][:8])
    by = {r['id']: r for r in rows}
    r1 = by['density:alpha2001made:s:0001']
    assert r1['tbl'] == 'density' and r1['grp'] == 'liquid' and r1['table_label'] == 'liquid/density'
    assert r1['year'] == 2001
    # relative u = 0.1 %, expanded with k = 2 -> absolute standard uncertainty
    assert r1['u_std'] == pytest.approx(0.001 * 999.0 / 2)
    assert by['density:alpha2001made:s:0002']['u_std'] == pytest.approx(0.5)
    assert by['density:alpha2001made:s:0003']['u_std'] is None
    assert by['density:alpha2001made:s:0004']['rate_K_min'] == pytest.approx(600.0)
    assert {s['source_key'] for s in sources} == {'alpha2001made', 'gamma1999made'}
    alpha = next(s for s in sources if s['source_key'] == 'alpha2001made')
    assert alpha['n_rows'] == 4 and alpha['cite_short'] == 'Alpha & Beta (2001)'
    assert 'pdf' not in alpha and 'catalog_status' not in alpha


def test_private_paths_are_removed(source):
    rows, _, _ = L.build(source)
    by = {r['id']: r for r in rows}
    # The Holten 2014 supplement is named after its source row when sources.csv lists one.
    assert by['density:alpha2001made:s:0001']['via'] == (
        'Holten et al. (2014) supplementary material: Data tables/rho-Alpha.txt')
    assert by['density:alpha2001made:s:0002']['via'] == 'Alpha & Beta (2001) supplementary material: tables/T2.txt'
    assert by['density:alpha2001made:s:0003']['via'] == ''
    assert 'the reading note on alpha2001made' in by['density:alpha2001made:s:0001']['notes']
    assert by['density:alpha2001made:s:0002']['notes'].endswith('see alpha2001.pdf for the scan')
    assert "the database's curation overrides" in by['density:alpha2001made:s:0003']['notes']
    assert "the database's reference checks" in by['melting:gamma1999made:m:0001']['qc_note']
    for r in rows:
        for c in S.ROW_COLUMNS:
            v = r.get(c)
            if isinstance(v, str):
                assert not L._LEFTOVER.search(v), (r['id'], c, v)


def test_unhandled_local_path_stops_the_load(source):
    p = source / 'phase_equilibrium' / 'melting.csv'
    text = p.read_text(encoding='utf-8').replace('ice_phase', 'ice_phase', 1)
    rows = list(csv.DictReader(io.StringIO(text)))
    rows[0]['sample_detail'] = 'sample list in fitting_data/stubs/box3.csv'
    _write(p, rows)
    with pytest.raises(L.LoadError, match='local path'):
        L.build(source)


@pytest.mark.parametrize('breakage, message', [
    ('missing_column', 'missing columns'),
    ('duplicate_id', 'duplicate id'),
    ('bad_number', 'not a number'),
    ('bad_substance', 'substance'),
])
def test_invalid_tables_are_rejected(source, breakage, message):
    p = source / 'liquid' / 'density.csv'
    rows = list(csv.DictReader(io.StringIO(p.read_text(encoding='utf-8'))))
    header = list(S.SOURCE_COLUMNS)
    if breakage == 'missing_column':
        header.remove('u_k')
        rows = [{k: v for k, v in r.items() if k != 'u_k'} for r in rows]
    elif breakage == 'duplicate_id':
        rows[1]['id'] = rows[0]['id']
    elif breakage == 'bad_number':
        rows[0]['T_K'] = 'warm'
    elif breakage == 'bad_substance':
        rows[0]['substance'] = 'HDO'
    _write(p, rows, header=header)
    with pytest.raises(L.LoadError, match=message):
        L.build(source)


def test_truncated_file_is_rejected(source):
    p = source / 'liquid' / 'density.csv'
    p.write_text(p.read_text(encoding='utf-8') + 'density:x:y:9,density,density\n', encoding='utf-8')
    with pytest.raises(L.LoadError, match='wrong number of fields'):
        L.build(source)


def test_sqlite_write_replaces_the_file(source, tmp_path):
    db = tmp_path / 'out' / 'measured.sqlite'
    assert L.main(['--source', str(source), '--db', f'sqlite:///{db}']) == 0
    first = db.stat().st_ino
    assert L.main(['--source', str(source), '--db', f'sqlite:///{db}']) == 0
    assert db.stat().st_ino != first           # a new file renamed over the old one
    assert not list(db.parent.glob('*.tmp-*'))
    conn = dbmod.connect(dbmod.parse_url(str(db)))
    assert conn.execute(f'SELECT COUNT(*) FROM {S.TABLE_ROWS}').fetchone()[0] == 6
    conn.close()


def test_dry_run_writes_nothing(source, tmp_path):
    db = tmp_path / 'never.sqlite'
    assert L.main(['--source', str(source), '--db', f'sqlite:///{db}', '--dry-run']) == 0
    assert not db.exists()


def test_parse_url():
    assert dbmod.parse_url('sqlite:///rel/x.sqlite') == dbmod.Target('sqlite', path='rel/x.sqlite')
    assert dbmod.parse_url('sqlite:////abs/x.sqlite').path == '/abs/x.sqlite'
    assert dbmod.parse_url('local.db').kind == 'sqlite'
    t = dbmod.parse_url('postgres://user:secret@db.example:5432/water')
    assert t.kind == 'postgres' and 'secret' not in t.describe() and 'db.example:5432' in t.describe()
    with pytest.raises(ValueError):
        dbmod.parse_url('mysql://x')


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
@pytest.fixture()
def client(source, tmp_path):
    db = tmp_path / 'api.sqlite'
    assert L.main(['--source', str(source), '--db', f'sqlite:///{db}']) == 0
    reset_store(f'sqlite:///{db}')
    yield TestClient(create_app())
    reset_store()


def test_release(client):
    r = client.get('/api/data/release')
    assert r.status_code == 200
    d = r.json()
    assert d['configured'] is True and d['release']['n_rows'] == 6
    assert [g['key'] for g in d['groups']] == ['liquid', 'phase_equilibrium']
    tables = {t['key']: t for t in d['tables']}
    dens = tables['density']['properties'][0]
    assert dens['key'] == 'density' and dens['unit'] == 'kg/m3' and dens['locus'] is None
    keys = [m['key'] for m in dens['models']]
    assert 'IAPWS95' in keys and 'holten2014' in keys and 'vapor' not in keys
    from watereos.model_registry import MODEL_REGISTRY
    assert not [k for k in keys if getattr(MODEL_REGISTRY[k], 'is_model_water', False)]
    melt = tables['melting']['properties'][0]
    assert melt['locus'] == 'T' and melt['models'] == []
    assert 'cite the original studies' in d['notice'] and d['release']['release_id'] in d['cite']


def test_points_and_conditional_get(client):
    r = client.get('/api/data/points', params={'table': 'density', 'property': 'density'})
    assert r.status_code == 200
    d = r.json()
    assert d['n'] == 4
    cols = d['columns']
    assert set(cols) >= {'id', 'table', 'group', 'T_K', 'P_MPa', 'value', 'u_std', 'substance', 'source_key'}
    assert cols['table'] == ['density'] * 4 and cols['group'] == ['liquid'] * 4
    etag = r.headers['etag']
    again = client.get('/api/data/points', params={'table': 'density', 'property': 'density'},
                       headers={'If-None-Match': etag})
    assert again.status_code == 304
    assert client.get('/api/data/points').json()['n'] == 6


def test_row_and_sources(client):
    r = client.get('/api/data/rows/density:alpha2001made:s:0001')
    assert r.status_code == 200
    d = r.json()
    assert d['row']['table'] == 'density' and d['source']['doi'] == '10.0000/alpha'
    assert d['property_label'].startswith('Density')
    assert client.get('/api/data/rows/nope').status_code == 404
    s = client.get('/api/data/sources').json()['sources']
    assert [x['source_key'] for x in s] == ['alpha2001made', 'gamma1999made']


def test_download(client):
    r = client.get('/api/data/download')
    assert r.status_code == 200 and r.headers['content-type'] == 'application/zip'
    assert 'attachment' in r.headers['content-disposition']
    z = zipfile.ZipFile(io.BytesIO(r.content))
    assert sorted(z.namelist()) == ['README.txt', 'measured_water_data.csv', 'sources.csv']
    rows = list(csv.DictReader(io.StringIO(z.read('measured_water_data.csv').decode())))
    assert len(rows) == 6 and 'table' in rows[0] and 'tbl' not in rows[0]
    readme = z.read('README.txt').decode()
    assert 'attaches no license' in readme and 'cite waterEoS' in readme


def test_lines(client):
    d = client.get('/api/data/lines').json()
    ih = d['melting']['branches'][0]
    assert ih['label'] == 'ice Ih' and ih['T_K'][-1] == pytest.approx(273.16)
    assert ih['P_MPa'][-1] == pytest.approx(611.657e-6, rel=1e-6)
    assert min(d['th']['T_K']) > 170 and max(d['th']['P_MPa']) == pytest.approx(1000)


def test_model_curves(client):
    body = {'property': 'density', 'models': ['IAPWS95'], 'along': 'T', 'fixed': [0.1, 100], 'lo': 250,
            'hi': 300, 'n': 21}
    d = client.post('/api/data/model-curves', json=body).json()
    assert len(d['curves']) == 2
    c0 = d['curves'][0]
    assert c0['fixed'] == 0.1 and len(c0['x']) == 21
    i = c0['x'].index(280.0) if 280.0 in c0['x'] else 12
    assert 990 < c0['y'][i] < 1001                     # liquid water near 280 K, 0.1 MPa
    assert c0['in_range'][0] is True and c0['in_range'][-1] is True  # IAPWS-95 spans 250-300 K


def test_model_curves_compressibility_is_inverted(client):
    body = {'property': 'isothermal_compressibility', 'models': ['IAPWS95'], 'along': 'T', 'fixed': [0.1],
            'lo': 290, 'hi': 300, 'n': 11}
    y = client.post('/api/data/model-curves', json=body).json()['curves'][0]['y']
    assert 4.3e-4 < y[0] < 4.8e-4                      # kappa_T of water near 290 K, in 1/MPa


@pytest.mark.parametrize('body', [
    {'property': 'melting_point', 'models': ['IAPWS95'], 'along': 'T', 'fixed': [0.1], 'lo': 250, 'hi': 300},
    {'property': 'density', 'models': ['vapor'], 'along': 'T', 'fixed': [0.1], 'lo': 250, 'hi': 300},
    {'property': 'density', 'models': ['IAPWS95'], 'along': 'T', 'fixed': [0.1], 'lo': 300, 'hi': 250},
    {'property': 'density', 'models': ['IAPWS95'], 'along': 'T', 'fixed': [0.1], 'lo': 0, 'hi': 9999},
])
def test_model_curves_rejects(client, body):
    assert client.post('/api/data/model-curves', json=body).status_code == 422


def test_model_surface(client):
    body = {'property': 'density', 'model': 'IAPWS95', 'T_range': [260, 300], 'P_range': [0.1, 100], 'n': 10}
    d = client.post('/api/data/model-surface', json=body).json()
    assert len(d['T_K']) == 10 and len(d['z']) == 10 and len(d['z'][0]) == 10
    assert all(v is None or 900 < v < 1100 for row in d['z'] for v in row)


def test_not_configured(monkeypatch):
    for v in dbmod.ENV_VARS:
        monkeypatch.delenv(v, raising=False)
    reset_store()
    try:
        c = TestClient(create_app())
        d = c.get('/api/data/release').json()
        assert d['configured'] is False
        assert c.get('/api/data/points').status_code == 503
    finally:
        reset_store()


def test_empty_database_is_unavailable(tmp_path):
    db = tmp_path / 'empty.sqlite'
    conn = dbmod.connect(dbmod.parse_url(str(db)), readonly=False)
    for stmt in S.ddl():
        conn.execute(stmt)
    conn.commit()
    conn.close()
    reset_store(f'sqlite:///{db}')
    try:
        c = TestClient(create_app())
        r = c.get('/api/data/release')
        assert r.status_code == 503 and r.json()['configured'] is False
        assert 'no release' in r.json()['detail']
    finally:
        reset_store()


# ---------------------------------------------------------------------------
# PostgreSQL compatibility (checked against a live server by hand; these
# guard the two differences from SQLite that matter)
# ---------------------------------------------------------------------------
def test_floats_are_double_precision():
    # PostgreSQL's REAL is single precision and would round the published values.
    ddl = '\n'.join(S.ddl())
    assert ' REAL' not in ddl.replace('DOUBLE PRECISION', '')
    assert S.ROW_COLUMN_TYPES['value'] == 'DOUBLE PRECISION'
    assert S.ROW_COLUMN_TYPES['u_std'] == 'DOUBLE PRECISION'


def test_lower_cased_column_names_are_restored(tmp_path, source):
    # PostgreSQL returns unquoted identifiers in lower case (T_K -> t_k).
    assert S.CANONICAL_NAMES['t_k'] == 'T_K' and S.CANONICAL_NAMES['rate_k_min'] == 'rate_K_min'
    db = tmp_path / 'lc.sqlite'
    assert L.main(['--source', str(source), '--db', f'sqlite:///{db}']) == 0
    from watereos_api.measured.store import MeasuredStore
    cols, _ = MeasuredStore(f'sqlite:///{db}').query(
        f'SELECT t_k, p_mpa, u_p_mpa, rate_k_min FROM {S.TABLE_ROWS}')
    assert cols == ['T_K', 'P_MPa', 'u_P_MPa', 'rate_K_min']


def test_forbidden_terms_stop_the_load(source):
    p = source / 'phase_equilibrium' / 'melting.csv'
    rows = list(csv.DictReader(io.StringIO(p.read_text(encoding='utf-8'))))
    rows[0]['notes'] = 'measured in the Secret-Lab cold room'
    _write(p, rows)
    L.build(source)                                     # nothing forbidden yet
    with pytest.raises(L.LoadError, match='local path'):
        L.build(source, forbid=['secret-lab'])
    assert L.main(['--source', str(source), '--dry-run', '--forbid', 'Secret-Lab']) == 1


def test_order_does_not_depend_on_the_database(source, tmp_path, monkeypatch):
    # PostgreSQL with en_US.utf8 returns rows in a different order from SQLite;
    # simulate that by reversing whatever the database returns.
    from watereos_api.measured.store import MeasuredStore
    db = tmp_path / 'order.sqlite'
    assert L.main(['--source', str(source), '--db', f'sqlite:///{db}']) == 0
    real = MeasuredStore.query

    def reversed_query(self, statement, params=()):
        cols, rows = real(self, statement, params)
        return cols, list(reversed(rows))

    monkeypatch.setattr(MeasuredStore, 'query', reversed_query)
    reset_store(f'sqlite:///{db}')
    try:
        c = TestClient(create_app())
        ids = c.get('/api/data/points').json()['columns']['id']
        assert ids == sorted(ids) and len(ids) == 6
        keys = [s['source_key'] for s in c.get('/api/data/sources').json()['sources']]
        assert keys == sorted(keys)
        z = zipfile.ZipFile(io.BytesIO(c.get('/api/data/download').content))
        rows = list(csv.DictReader(io.StringIO(z.read('measured_water_data.csv').decode())))
        order = [(r['group'], r['table'], r['id']) for r in rows]
        assert order == sorted(order)
        srcs = [r['source_key'] for r in csv.DictReader(io.StringIO(z.read('sources.csv').decode()))]
        assert srcs == sorted(srcs)
    finally:
        reset_store()
