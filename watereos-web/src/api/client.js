const getBase = () => import.meta.env.VITE_API_BASE_URL ?? '';

async function _get(path) {
  const r = await fetch(`${getBase()}${path}`);
  if (!r.ok) throw await _httpError(r);
  return r.json();
}

async function _post(path, body) {
  const r = await fetch(`${getBase()}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!r.ok) throw await _httpError(r);
  return r.json();
}

async function _httpError(r) {
  let detail = `${r.status} ${r.statusText || ''}`.trim();
  try {
    const j = await r.json();
    if (j && typeof j.detail === 'string') detail = j.detail;
  } catch {
    // body not JSON; keep the default detail
  }
  const err = new Error(detail);
  err.status = r.status;
  return err;
}

export const fetchHealth = () => _get('/api/health');
export const fetchMetadata = () => _get('/api/metadata');
export const fetchPoint = (body) => _post('/api/point', body);
export const fetchCurves = (body) => _post('/api/figures/curves', body);
export const fetchSurface2 = (body) => _post('/api/figures/surface2d', body);
export const fetchSurface3 = (body) => _post('/api/figures/surface3d', body);
export const fetchCompare = (body) => _post('/api/figures/compare', body);
export const fetchEosPhase = (body) => _post('/api/figures/eos-phase-diagram', body);
export const fetchH2OPhase = (body) => _post('/api/figures/h2o-phase-diagram', body);

// Measured water data (/api/data/*). These return plain data; the figures are
// built in the browser so that filtering, hover and clicks need no round trip.
export const fetchDataRelease = async () => {
  const r = await fetch(`${getBase()}/api/data/release`);
  // 503 with {configured: false} is a normal answer (no database on this server).
  if (r.status === 503) {
    try {
      const j = await r.json();
      if (j && j.configured === false) return j;
    } catch {
      // fall through to the generic error
    }
  }
  if (!r.ok) throw await _httpError(r);
  return r.json();
};
export const fetchDataPoints = (table, property) => {
  const q = new URLSearchParams();
  if (table) q.set('table', table);
  if (property) q.set('property', property);
  const s = q.toString();
  return _get(`/api/data/points${s ? '?' + s : ''}`);
};
export const fetchDataRow = (id) => _get(`/api/data/rows/${encodeURIComponent(id)}`);
export const fetchDataSources = () => _get('/api/data/sources');
export const fetchDataLines = () => _get('/api/data/lines');
export const fetchModelCurves = (body) => _post('/api/data/model-curves', body);
export const fetchModelSurface = (body) => _post('/api/data/model-surface', body);
export const dataDownloadUrl = () => `${getBase()}/api/data/download`;
