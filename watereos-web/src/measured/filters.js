// Sidebar filters for the measured data. Categorical filters store what is
// *excluded*, so a category that appears in another table is shown by default.

export const CATEGORY_FILTERS = [
  { key: 'substance', label: 'Substance' },
  { key: 'data_type', label: 'Data type' },
  { key: 'extraction', label: 'Extraction' },
  { key: 'state', label: 'State' },
  { key: 'sample', label: 'Sample' },
  { key: 'qc_flag', label: 'QC flag' },
  { key: 'method', label: 'Method', search: true },
];

export const DEFAULT_FILTERS = {
  exclude: {
    substance: ['D2O'],
    data_type: ['quoted', 'correlation'],
    extraction: [],
    state: [],
    sample: [],
    qc_flag: [],
    method: [],
  },
  sources: [], // included source keys; empty = every source
  fits: [], // fit tokens (or 'none'); empty = no restriction
  fitPointOnly: false,
  yearMin: null,
  yearMax: null,
};

function fitsPass(c, i, want, pointOnly) {
  const used = c.used_in_fits[i] || '';
  if (want.has('none') && !used) return true;
  let toks;
  if (pointOnly) {
    toks = (c.used_in_fits_level[i] || '')
      .split(';')
      .filter((s) => s.endsWith(':point'))
      .map((s) => s.split(':')[0]);
  } else toks = used.split(';');
  return toks.some((t) => t && want.has(t));
}

// The subset of `idx` (row indices) that passes every filter. `skip` names
// one filter to ignore (used to count the options of that filter within the
// other ones).
export function applyFilters(c, idx, f, skip = null) {
  const ex = {};
  for (const { key } of CATEGORY_FILTERS) {
    if (key !== skip && f.exclude[key]?.length) ex[key] = new Set(f.exclude[key]);
  }
  const src = skip !== 'sources' && f.sources.length ? new Set(f.sources) : null;
  const fits = skip !== 'fits' && f.fits.length ? new Set(f.fits) : null;
  const yMin = f.yearMin;
  const yMax = f.yearMax;
  const out = [];
  outer: for (const i of idx) {
    for (const k in ex) if (ex[k].has(c[k][i] ?? '')) continue outer;
    if (src && !src.has(c.source_key[i])) continue;
    if (fits && !fitsPass(c, i, fits, f.fitPointOnly)) continue;
    const y = c.year[i];
    if (yMin != null && (y == null || y < yMin)) continue;
    if (yMax != null && (y == null || y > yMax)) continue;
    out.push(i);
  }
  return out;
}

// [[value, count]] of column `key` over the given rows, most frequent first.
export function countBy(c, idx, key) {
  const m = new Map();
  for (const i of idx) {
    const v = c[key][i] ?? '';
    m.set(v, (m.get(v) || 0) + 1);
  }
  return [...m.entries()].sort((a, b) => b[1] - a[1] || String(a[0]).localeCompare(String(b[0])));
}

export function activeFilterCount(f) {
  let n = 0;
  for (const { key } of CATEGORY_FILTERS) {
    const d = DEFAULT_FILTERS.exclude[key];
    const cur = f.exclude[key] || [];
    if (cur.length !== d.length || cur.some((v) => !d.includes(v))) n++;
  }
  if (f.sources.length) n++;
  if (f.fits.length) n++;
  if (f.yearMin != null || f.yearMax != null) n++;
  return n;
}

// Rows per fit token (and 'none') among the given rows.
export function countFits(c, idx, pointOnly) {
  const m = new Map();
  for (const i of idx) {
    const used = c.used_in_fits[i] || '';
    if (!used) {
      m.set('none', (m.get('none') || 0) + 1);
      continue;
    }
    const toks = pointOnly
      ? (c.used_in_fits_level[i] || '')
          .split(';')
          .filter((s) => s.endsWith(':point'))
          .map((s) => s.split(':')[0])
      : used.split(';');
    for (const t of new Set(toks)) if (t) m.set(t, (m.get(t) || 0) + 1);
  }
  return m;
}
