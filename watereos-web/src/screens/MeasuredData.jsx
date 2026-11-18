import { useCallback, useMemo, useState } from 'react';
import { Sidebar } from '../components/Sidebar.jsx';
import { Field } from '../components/Field.jsx';
import { Segmented } from '../components/Segmented.jsx';
import { Checkbox } from '../components/Checkbox.jsx';
import { PlotCard } from '../components/PlotCard.jsx';
import { CheckList } from '../components/CheckList.jsx';
import { useTheme } from '../theme/ThemeContext.jsx';
import {
  useDataLines,
  useDataPoints,
  useDataRelease,
  useDataRow,
  useDataSources,
  useModelCurves,
  useModelSurfaces,
} from '../api/hooks.js';
import { dataDownloadUrl } from '../api/client.js';
import {
  CATEGORY_FILTERS,
  DEFAULT_FILTERS,
  activeFilterCount,
  applyFilters,
  countBy,
  countFits,
} from '../measured/filters.js';
import {
  COLOR_MODES,
  COVERAGE_COLOR_MODES,
  coverageFigure,
  inNoMansLand,
  paddedRange,
  propertyFigure2d,
  propertyFigure3d,
} from '../measured/figures.js';
import {
  FIT_LABELS,
  FIT_TOKENS,
  blankLabel,
  fitsLabel,
  fmtNum,
  unitPretty,
} from '../measured/format.js';
import '../measured/measured.css';

const VIEWS = [
  { value: 'property', label: 'Property' },
  { value: 'coverage', label: 'Coverage' },
  { value: 'rows', label: 'Rows' },
  { value: 'sources', label: 'Sources' },
];
const ISOBARS = [0.1, 10, 50, 100, 150, 200, 300, 400, 500, 700, 1000, 2000];
const ISOTHERMS = [
  180, 200, 220, 240, 253.15, 273.15, 298.15, 323.15, 348.15, 373.15, 423.15, 473.15,
];
const NOT_THERMO = {
  kinetic:
    'Kinetic data: these bound where the liquid or glass can be observed. They are not thermodynamic data.',
  dynamics: 'Transport and relaxation data: not thermodynamic data.',
  structure: 'Structure fractions derived by the authors: not thermodynamic data.',
};

function parseNums(s) {
  return String(s)
    .split(/[,\s;]+/)
    .map(Number)
    .filter((v) => Number.isFinite(v));
}

// Up to five "nice" isobars or isotherms within the range of the data.
function defaultAt(values, candidates) {
  const v = values.filter((x) => x != null && Number.isFinite(x));
  if (!v.length) return [];
  const lo = Math.min(...v);
  const hi = Math.max(...v);
  const inside = candidates.filter((c) => c >= lo - 1e-9 && c <= hi + 1e-9);
  if (inside.length === 0)
    return [...new Set([Number(lo.toPrecision(3)), Number(hi.toPrecision(3))])];
  if (inside.length <= 5) return inside;
  return [0, 1, 2, 3, 4].map((k) => inside[Math.round((k * (inside.length - 1)) / 4)]);
}

const round = (x, d) => Math.round(x * 10 ** d) / 10 ** d;

export default function MeasuredData() {
  const rel = useDataRelease();
  if (rel.error) {
    return (
      <>
        <Sidebar />
        <section className="main">
          <div className="plot-error" role="alert">
            Could not load the measured data: {String(rel.error.message ?? rel.error)}
          </div>
        </section>
      </>
    );
  }
  if (!rel.data) {
    return (
      <>
        <Sidebar />
        <section className="main">
          <div className="screen-loading">Loading measured data…</div>
        </section>
      </>
    );
  }
  if (!rel.data.configured || !rel.data.tables?.length) {
    return (
      <>
        <Sidebar />
        <section className="main">
          <div className="md-cite" role="status">
            <p>
              <b>Measured data are not available on this server.</b>
            </p>
            <p>{rel.data.detail ?? 'The measured-data database holds no rows.'}</p>
          </div>
        </section>
      </>
    );
  }
  return <MeasuredDataScreen release={rel.data} />;
}

function MeasuredDataScreen({ release }) {
  const { theme } = useTheme();
  const tables = release.tables;

  const [view, setView] = useState('property');
  const [tableKey, setTableKey] = useState(
    () => (tables.find((t) => t.key === 'density') ?? tables[0]).key
  );
  const table = tables.find((t) => t.key === tableKey) ?? tables[0];
  const [propKey, setPropKey] = useState(null);
  const prop = table.properties.find((p) => p.key === propKey) ?? table.properties[0];

  const [xVar, setXVar] = useState('T');
  const [dim, setDim] = useState('2d');
  const [colorMode, setColorMode] = useState('other');
  const [errorBars, setErrorBars] = useState(true);
  const [logX, setLogX] = useState(false);
  const [logOverride, setLogOverride] = useState({});
  const logY = logOverride[prop.key] ?? prop.log;
  const [models, setModels] = useState([]);
  const [atText, setAtText] = useState({});
  const [floorLines, setFloorLines] = useState(true);
  const [filters, setFilters] = useState(DEFAULT_FILTERS);
  const [selectedId, setSelectedId] = useState(null);
  const [showCite, setShowCite] = useState(false);

  const [covGroups, setCovGroups] = useState(() =>
    release.groups.filter((g) => g.thermodynamic).map((g) => g.key)
  );
  const [covColor, setCovColor] = useState('table');
  const [covMode, setCovMode] = useState('points');
  const [covNml, setCovNml] = useState(true);
  const [covLogY, setCovLogY] = useState(false);
  const [binT, setBinT] = useState(5);
  const [binP, setBinP] = useState(25);

  const needTable = view === 'property' || view === 'rows';
  const pts = useDataPoints(table.key, prop.key, needTable);
  const all = useDataPoints(null, null, view === 'coverage');
  const srcs = useDataSources(true);
  const lines = useDataLines(true);

  const names = useMemo(() => {
    const cites = {};
    for (const s of srcs.data?.sources ?? []) cites[s.source_key] = s.cite_short || s.source_key;
    const tl = {};
    const props = {};
    for (const t of tables) {
      tl[t.key] = t.label;
      for (const p of t.properties) props[p.key] = p.label;
    }
    const groups = {};
    for (const g of release.groups) groups[g.key] = g.label;
    return { cites, tables: tl, props, groups };
  }, [srcs.data, tables, release.groups]);

  const scope = view === 'coverage' ? all : pts;
  const cols = scope.data?.columns;
  const n = scope.data?.n ?? 0;
  const scopeIdx = useMemo(() => {
    if (!cols) return [];
    if (view !== 'coverage') return Array.from({ length: n }, (_, i) => i);
    const want = new Set(covGroups);
    const out = [];
    for (let i = 0; i < n; i++) if (want.has(cols.group[i])) out.push(i);
    return out;
  }, [cols, n, view, covGroups]);
  const idx = useMemo(
    () => (cols ? applyFilters(cols, scopeIdx, filters) : []),
    [cols, scopeIdx, filters]
  );

  // ---- property view -------------------------------------------------------
  const isLocus = !!prop.locus;
  const effDim = isLocus ? '2d' : dim;
  const otherKey = xVar === 'T' ? 'P_MPa' : 'T_K';
  const xKey = xVar === 'T' ? 'T_K' : 'P_MPa';
  const available = useMemo(() => prop.models ?? [], [prop]);
  const activeModels = useMemo(
    () => models.filter((m) => available.some((a) => a.key === m)),
    [models, available]
  );
  const overlayOK = !isLocus && available.length > 0;
  const modelLabels = useMemo(
    () => Object.fromEntries(available.map((m) => [m.key, m.label])),
    [available]
  );

  const atKey = `${table.key}|${prop.key}|${xVar}`;
  const atDefault = useMemo(
    () =>
      cols && needTable
        ? defaultAt(
            idx.map((i) => cols[otherKey][i]),
            xVar === 'T' ? ISOBARS : ISOTHERMS
          )
        : [],
    [cols, needTable, idx, otherKey, xVar]
  );
  const atStr = atText[atKey] ?? atDefault.join(', ');
  const atVals = useMemo(() => parseNums(atStr).slice(0, 12), [atStr]);

  const ranges = useMemo(() => {
    if (!cols || !needTable) return null;
    const ok = idx.filter((i) => cols.value[i] != null);
    const x = paddedRange(
      ok.map((i) => cols[xKey][i]),
      false,
      0.02
    );
    const T = paddedRange(
      ok.map((i) => cols.T_K[i]),
      false,
      0.02
    );
    const P = paddedRange(
      ok.map((i) => cols.P_MPa[i]),
      false,
      0.02
    );
    const d = xVar === 'T' ? 1 : 2;
    return {
      x: x && [round(x[0], d), round(x[1], d)],
      T: T && [Math.max(1, round(T[0], 1)), round(T[1], 1)],
      P: P && [round(P[0], 2), round(P[1], 2)],
    };
  }, [cols, needTable, idx, xKey, xVar]);

  const curvesBody = useMemo(() => {
    if (
      view !== 'property' ||
      effDim !== '2d' ||
      !overlayOK ||
      !activeModels.length ||
      !atVals.length
    )
      return null;
    if (!ranges?.x || ranges.x[0] >= ranges.x[1]) return null;
    return {
      property: prop.key,
      models: activeModels,
      along: xVar,
      fixed: atVals,
      lo: ranges.x[0],
      hi: ranges.x[1],
      n: 160,
    };
  }, [view, effDim, overlayOK, activeModels, atVals, ranges, prop.key, xVar]);
  const curvesQ = useModelCurves(curvesBody);
  const surfOn = view === 'property' && effDim === '3d' && overlayOK && ranges?.T && ranges?.P;
  const surfQ = useModelSurfaces(surfOn ? prop.key : null, activeModels, ranges?.T, ranges?.P, 50);

  const nonThermo = useMemo(
    () => new Set(release.groups.filter((g) => !g.thermodynamic).map((g) => g.key)),
    [release.groups]
  );

  const figure = useMemo(() => {
    if (!cols) return null;
    if (view === 'property') {
      const common = {
        c: cols,
        idx,
        prop,
        xVar,
        colorMode,
        errorBars,
        theme,
        names,
        modelLabels,
        lines: lines.data,
        uirevision: `${table.key}|${prop.key}|${xVar}|${effDim}`,
      };
      return effDim === '3d'
        ? propertyFigure3d({ ...common, logZ: logY, surfaces: surfQ.data, floorLines })
        : propertyFigure2d({ ...common, logX, logY, curves: curvesQ.data?.curves, at: atVals });
    }
    if (view === 'coverage') {
      return coverageFigure({
        c: cols,
        idx,
        colorMode: covColor,
        mode: covMode,
        theme,
        names,
        lines: lines.data,
        showNml: covNml,
        logY: covLogY,
        binT: Math.max(0.1, binT || 5),
        binP: Math.max(0.1, binP || 25),
        nonThermo,
        uirevision: 'coverage',
      });
    }
    return null;
  }, [
    cols,
    view,
    idx,
    prop,
    xVar,
    colorMode,
    errorBars,
    theme,
    names,
    modelLabels,
    lines.data,
    table.key,
    effDim,
    logY,
    surfQ.data,
    floorLines,
    logX,
    curvesQ.data,
    atVals,
    covColor,
    covMode,
    covNml,
    covLogY,
    binT,
    binP,
    nonThermo,
  ]);

  const onPlotClick = useCallback((e) => {
    const cd = e?.points?.[0]?.customdata;
    if (typeof cd === 'string') setSelectedId(cd);
  }, []);

  // ---- handlers ---------------------------------------------------------------
  const pickTable = (k) => {
    setTableKey(k);
    setPropKey(null);
    setSelectedId(null);
  };
  const setExclude = (key, values, excluded) =>
    setFilters((f) => {
      const cur = new Set(f.exclude[key]);
      for (const v of values) excluded ? cur.add(v) : cur.delete(v);
      return { ...f, exclude: { ...f.exclude, [key]: [...cur] } };
    });
  const setInclude = (key, values, included) =>
    setFilters((f) => {
      const cur = new Set(f[key]);
      for (const v of values) included ? cur.add(v) : cur.delete(v);
      return { ...f, [key]: [...cur] };
    });
  const onlySource = (key) => {
    setFilters((f) => ({ ...f, sources: [key] }));
    const s = srcs.data?.sources?.find((x) => x.source_key === key);
    const labels = (s?.tables ?? '').split(';').map((x) => x.trim().replace(/\s*\(\d+\)$/, ''));
    const t = tables.find((tb) => labels.includes(tb.label));
    if (t && !labels.includes(table.label)) pickTable(t.key);
    setView('property');
  };

  // ---- filter options (counts within the other filters) -------------------------
  const filterItems = useMemo(() => {
    if (!cols) return null;
    const out = {};
    for (const { key } of CATEGORY_FILTERS) {
      const rows = applyFilters(cols, scopeIdx, filters, key);
      out[key] = countBy(cols, rows, key).map(([v, c]) => ({
        value: v,
        label: blankLabel(v),
        count: c,
      }));
      for (const v of filters.exclude[key]) {
        if (!out[key].some((it) => it.value === v))
          out[key].push({ value: v, label: blankLabel(v), count: 0 });
      }
    }
    const srcRows = applyFilters(cols, scopeIdx, filters, 'sources');
    out.sources = countBy(cols, srcRows, 'source_key')
      .map(([v, c]) => ({ value: v, label: names.cites[v] ?? v, title: v, count: c }))
      .sort((a, b) => a.label.localeCompare(b.label));
    const fitRows = applyFilters(cols, scopeIdx, filters, 'fits');
    const fc = countFits(cols, fitRows, filters.fitPointOnly);
    out.fits = [...FIT_TOKENS, 'none']
      .filter((t) => fc.get(t) || filters.fits.includes(t))
      .map((t) => ({
        value: t,
        label: t === 'none' ? 'not used by any' : FIT_LABELS[t],
        count: fc.get(t) || 0,
      }));
    return out;
  }, [cols, scopeIdx, filters, names]);

  const nFilters = activeFilterCount(filters);
  const hasD2O = cols ? idx.some((i) => cols.substance[i] === 'D2O') : false;
  const notThermo = NOT_THERMO[view === 'coverage' ? '' : table.group];

  let caption = null;
  if (figure && view === 'property') {
    const parts = [`${figure.n.toLocaleString()} of ${n.toLocaleString()} rows plotted`];
    if (idx.length - figure.n > 0)
      parts.push(
        `${(idx.length - figure.n).toLocaleString()} filtered rows lack ${isLocus ? 'T or P' : 'a coordinate'}`
      );
    if (hasD2O) parts.push('open markers: D₂O');
    if (activeModels.length)
      parts.push('models describe H₂O; faint where outside their stated range');
    if (isLocus) parts.push('a locus: drawn in the T–P plane');
    if (curvesQ.error) parts.push(`model curves failed: ${curvesQ.error.message}`);
    caption = (
      <div className="md-caption">
        {parts.map((p) => (
          <span key={p}>{p}</span>
        ))}
      </div>
    );
  } else if (figure && view === 'coverage') {
    const nml = lines.data
      ? idx.filter((i) => inNoMansLand(lines.data, cols.T_K[i], cols.P_MPa[i])).length
      : 0;
    caption = (
      <div className="md-caption">
        <span>{figure.n.toLocaleString()} rows with T and P</span>
        <span>{(idx.length - figure.n).toLocaleString()} filtered rows lack T or P</span>
        {lines.data && <span>{nml.toLocaleString()} in no man’s land</span>}
        <span>× marks: kinetic, dynamics and structure rows</span>
      </div>
    );
  }

  return (
    <>
      <Sidebar>
        <Field label="View">
          <div className="md-fill">
            <Segmented options={VIEWS} value={view} onChange={setView} />
          </div>
        </Field>

        {needTable && (
          <>
            <Field label="Table">
              <select
                className="select"
                value={table.key}
                onChange={(e) => pickTable(e.target.value)}
              >
                {release.groups.map((g) => (
                  <optgroup key={g.key} label={g.label}>
                    {tables
                      .filter((t) => t.group === g.key)
                      .map((t) => (
                        <option key={t.key} value={t.key}>
                          {t.label} ({t.n.toLocaleString()})
                        </option>
                      ))}
                  </optgroup>
                ))}
              </select>
            </Field>
            {table.properties.length > 1 && (
              <Field label="Property">
                <select
                  className="select"
                  value={prop.key}
                  onChange={(e) => setPropKey(e.target.value)}
                >
                  {table.properties.map((p) => (
                    <option key={p.key} value={p.key}>
                      {p.label} ({p.n.toLocaleString()})
                    </option>
                  ))}
                </select>
              </Field>
            )}
          </>
        )}

        {view === 'property' && (
          <>
            {!isLocus && (
              <Field label="Horizontal axis">
                <Segmented
                  options={[
                    { value: 'T', label: 'Temperature' },
                    { value: 'P', label: 'Pressure' },
                  ]}
                  value={xVar}
                  onChange={setXVar}
                />
              </Field>
            )}
            <Field label="Plot" hint={isLocus ? 'loci stay 2D' : undefined}>
              <Segmented
                options={[
                  { value: '2d', label: '2D' },
                  { value: '3d', label: '3D' },
                ]}
                value={effDim}
                onChange={(v) => !isLocus && setDim(v)}
              />
            </Field>
            <Field label="Color by">
              <select
                className="select"
                value={colorMode}
                onChange={(e) => setColorMode(e.target.value)}
              >
                {COLOR_MODES.map((m) => (
                  <option key={m.value} value={m.value}>
                    {m.value === 'other'
                      ? isLocus
                        ? 'Source'
                        : xVar === 'T'
                          ? 'Pressure (isobars)'
                          : 'Temperature (isotherms)'
                      : m.label}
                  </option>
                ))}
              </select>
            </Field>
            <div className="md-section">
              <Checkbox
                label="Error bars (standard uncertainty)"
                checked={errorBars}
                onChange={setErrorBars}
              />
              {effDim === '2d' && (
                <Checkbox label="Log horizontal axis" checked={logX} onChange={setLogX} />
              )}
              <Checkbox
                label={effDim === '3d' ? 'Log vertical axis' : 'Log value axis'}
                checked={logY}
                onChange={(v) => setLogOverride((o) => ({ ...o, [prop.key]: v }))}
              />
              {effDim === '3d' && (
                <Checkbox
                  label="Melting and T_H lines on the floor"
                  checked={floorLines}
                  onChange={setFloorLines}
                />
              )}
            </div>
            {overlayOK && (
              <>
                <div className="md-section-head">
                  <span className="section-title">Model overlay</span>
                </div>
                <CheckList
                  items={available.map((m) => ({ value: m.key, label: m.label }))}
                  checked={(v) => activeModels.includes(v)}
                  onToggle={(v, on) =>
                    setModels((ms) => (on ? [...new Set([...ms, v])] : ms.filter((x) => x !== v)))
                  }
                  maxHeight={180}
                />
                {effDim === '2d' && (
                  <Field label={xVar === 'T' ? 'Isobars at (MPa)' : 'Isotherms at (K)'}>
                    <input
                      className="input"
                      value={atStr}
                      onChange={(e) => setAtText((o) => ({ ...o, [atKey]: e.target.value }))}
                      placeholder="e.g. 0.1, 100, 200"
                    />
                  </Field>
                )}
                <div className="md-hint">
                  Equations for real water from the waterEoS registry. Fits to simulated model
                  waters are not offered here.
                </div>
              </>
            )}
          </>
        )}

        {view === 'coverage' && (
          <>
            <Field label="Groups">
              <CheckList
                items={release.groups.map((g) => ({ value: g.key, label: g.label }))}
                checked={(v) => covGroups.includes(v)}
                onToggle={(v, on) =>
                  setCovGroups((gs) => (on ? [...new Set([...gs, v])] : gs.filter((x) => x !== v)))
                }
              />
            </Field>
            <Field label="Show">
              <div className="md-fill">
                <Segmented
                  options={[
                    { value: 'points', label: 'Points' },
                    { value: 'counts', label: 'Counts per bin' },
                  ]}
                  value={covMode}
                  onChange={setCovMode}
                />
              </div>
            </Field>
            {covMode === 'points' ? (
              <Field label="Color by">
                <select
                  className="select"
                  value={covColor}
                  onChange={(e) => setCovColor(e.target.value)}
                >
                  {COVERAGE_COLOR_MODES.map((m) => (
                    <option key={m.value} value={m.value}>
                      {m.label}
                    </option>
                  ))}
                </select>
              </Field>
            ) : (
              <Field label="Bin size: T (K), P (MPa)">
                <div className="input-pair">
                  <input
                    className="input"
                    type="number"
                    min="0.1"
                    step="1"
                    value={binT}
                    onChange={(e) => setBinT(parseFloat(e.target.value))}
                  />
                  <input
                    className="input"
                    type="number"
                    min="0.1"
                    step="5"
                    value={binP}
                    onChange={(e) => setBinP(parseFloat(e.target.value))}
                  />
                </div>
              </Field>
            )}
            <div className="md-section">
              <Checkbox label="Outline no man’s land" checked={covNml} onChange={setCovNml} />
              <Checkbox label="Log pressure axis" checked={covLogY} onChange={setCovLogY} />
            </div>
          </>
        )}

        {view !== 'sources' && filterItems && (
          <>
            <div className="md-section-head">
              <span className="section-title">
                Filters{nFilters ? ` · ${nFilters} changed` : ''}
              </span>
              <button type="button" className="md-link" onClick={() => setFilters(DEFAULT_FILTERS)}>
                reset
              </button>
            </div>
            {CATEGORY_FILTERS.map(({ key, label }) => (
              <details
                key={key}
                className="md-filter-group"
                open={key === 'substance' || key === 'data_type'}
              >
                <summary>
                  {label}
                  {filters.exclude[key].length > 0 && (
                    <span className="md-filter-badge">{filters.exclude[key].length} hidden</span>
                  )}
                </summary>
                <CheckList
                  items={filterItems[key]}
                  checked={(v) => !filters.exclude[key].includes(v)}
                  onToggle={(v, on) => setExclude(key, [v], !on)}
                  onSet={(vs, on) => setExclude(key, vs, !on)}
                  maxHeight={180}
                />
              </details>
            ))}
            <details className="md-filter-group">
              <summary>
                Used in fits
                {filters.fits.length > 0 && (
                  <span className="md-filter-badge">{filters.fits.length} chosen</span>
                )}
              </summary>
              <CheckList
                items={filterItems.fits}
                checked={(v) => filters.fits.includes(v)}
                onToggle={(v, on) => setInclude('fits', [v], on)}
              />
              <Checkbox
                label="Point level only"
                checked={filters.fitPointOnly}
                onChange={(v) => setFilters((f) => ({ ...f, fitPointOnly: v }))}
              />
              <div className="md-hint">
                None ticked: every row. Ticked: rows used by any of them.
              </div>
            </details>
            <details className="md-filter-group">
              <summary>
                Publication year
                {(filters.yearMin != null || filters.yearMax != null) && (
                  <span className="md-filter-badge">set</span>
                )}
              </summary>
              <div className="input-pair">
                <input
                  className="input"
                  type="number"
                  placeholder={String(release.years?.[0] ?? '')}
                  value={filters.yearMin ?? ''}
                  onChange={(e) =>
                    setFilters((f) => ({
                      ...f,
                      yearMin: e.target.value === '' ? null : parseInt(e.target.value, 10),
                    }))
                  }
                />
                <input
                  className="input"
                  type="number"
                  placeholder={String(release.years?.[1] ?? '')}
                  value={filters.yearMax ?? ''}
                  onChange={(e) =>
                    setFilters((f) => ({
                      ...f,
                      yearMax: e.target.value === '' ? null : parseInt(e.target.value, 10),
                    }))
                  }
                />
              </div>
            </details>
            <details className="md-filter-group">
              <summary>
                Sources
                {filters.sources.length > 0 && (
                  <span className="md-filter-badge">{filters.sources.length} chosen</span>
                )}
              </summary>
              <CheckList
                items={filterItems.sources}
                checked={(v) => filters.sources.includes(v)}
                onToggle={(v, on) => setInclude('sources', [v], on)}
                search
                maxHeight={240}
              />
              <div className="md-hint">None ticked: every source.</div>
            </details>
          </>
        )}
      </Sidebar>

      <section className="main md-main">
        <header className="md-header">
          <div>
            <h1 className="md-title">Measured water data</h1>
            <div className="md-release">
              release {release.release.release_id} · {release.release.n_rows.toLocaleString()} rows
              · {release.release.n_sources.toLocaleString()} published studies
            </div>
          </div>
          <div className="md-header-actions">
            <button type="button" className="btn btn-ghost" onClick={() => setShowCite((v) => !v)}>
              {showCite ? 'Hide' : 'How to cite'}
            </button>
            <a className="btn btn-ghost" href={dataDownloadUrl()} download>
              Download all (zip)
            </a>
          </div>
        </header>
        {showCite && (
          <div className="md-cite">
            <p>{release.notice}</p>
            <p>
              Cite the original studies (each point names its study, with DOI) and cite waterEoS for
              the compilation:
            </p>
            <p>
              <code>{release.cite}</code>
            </p>
          </div>
        )}
        {notThermo && <div className="md-banner">{notThermo}</div>}

        <div className="md-body">
          {(view === 'property' || view === 'coverage') && (
            <PlotCard
              bare
              figure={figure}
              loading={scope.isLoading || (view === 'property' && curvesQ.isLoading)}
              error={scope.error || (view === 'property' && effDim === '3d' ? surfQ.error : null)}
              onClick={onPlotClick}
              footer={caption}
            />
          )}
          {view === 'rows' && cols && (
            <RowsTable
              c={cols}
              idx={idx}
              n={n}
              prop={prop}
              names={names}
              selectedId={selectedId}
              onSelect={setSelectedId}
            />
          )}
          {view === 'rows' && !cols && <div className="screen-loading">Loading rows…</div>}
          {view === 'sources' && <SourcesList data={srcs.data} onOnly={onlySource} />}
        </div>

        {selectedId && <DetailDrawer id={selectedId} onClose={() => setSelectedId(null)} />}
      </section>
    </>
  );
}

// ---------------------------------------------------------------------------
// Rows
// ---------------------------------------------------------------------------
const PAGE = 100;

function RowsTable({ c, idx, n, prop, names, selectedId, onSelect }) {
  const [sort, setSort] = useState({ key: 'T_K', dir: 1 });
  const [page, setPage] = useState(0);
  const columns = useMemo(
    () => [
      {
        key: 'source_key',
        label: 'Source',
        get: (i) => names.cites[c.source_key[i]] ?? c.source_key[i],
      },
      { key: 'T_K', label: 'T (K)', num: true },
      { key: 'P_MPa', label: 'P (MPa)', num: true },
      { key: 'value', label: `${prop.label} (${unitPretty(prop.unit)})`, num: true },
      { key: 'u_std', label: 'u (k = 1)', num: true },
      { key: 'substance', label: 'Substance' },
      { key: 'state', label: 'State' },
      { key: 'data_type', label: 'Data type' },
      { key: 'extraction', label: 'Extraction' },
      { key: 'qc_flag', label: 'QC' },
      { key: 'used_in_fits', label: 'Fits', get: (i) => fitsLabel(c.used_in_fits[i]) },
    ],
    [c, names, prop]
  );
  const sorted = useMemo(() => {
    const col = columns.find((x) => x.key === sort.key) ?? columns[1];
    const get = col.get ?? ((i) => c[col.key][i]);
    return [...idx].sort((a, b) => {
      const va = get(a);
      const vb = get(b);
      if (va == null && vb == null) return 0;
      if (va == null) return 1;
      if (vb == null) return -1;
      return (typeof va === 'number' ? va - vb : String(va).localeCompare(String(vb))) * sort.dir;
    });
  }, [idx, c, columns, sort]);
  const pages = Math.max(1, Math.ceil(sorted.length / PAGE));
  const p = Math.min(page, pages - 1);
  const shown = sorted.slice(p * PAGE, (p + 1) * PAGE);
  return (
    <div className="md-table-card">
      <div className="md-table-tools">
        <span>
          {idx.length.toLocaleString()} of {n.toLocaleString()} rows after filters · click a row for
          every column
        </span>
        <span className="md-pager">
          <button
            type="button"
            className="btn btn-ghost"
            disabled={p === 0}
            onClick={() => setPage(p - 1)}
          >
            ‹
          </button>
          <span>
            {p + 1} / {pages}
          </span>
          <button
            type="button"
            className="btn btn-ghost"
            disabled={p >= pages - 1}
            onClick={() => setPage(p + 1)}
          >
            ›
          </button>
        </span>
      </div>
      <div className="md-table-scroll">
        <table className="md-table">
          <thead>
            <tr>
              {columns.map((x) => (
                <th
                  key={x.key}
                  className={x.num ? 'num' : undefined}
                  onClick={() =>
                    setSort((s) => ({ key: x.key, dir: s.key === x.key ? -s.dir : 1 }))
                  }
                >
                  {x.label}
                  {sort.key === x.key ? (sort.dir > 0 ? ' ▲' : ' ▼') : ''}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {shown.map((i) => (
              <tr
                key={c.id[i]}
                className={c.id[i] === selectedId ? 'selected' : undefined}
                onClick={() => onSelect(c.id[i])}
              >
                {columns.map((x) => {
                  const v = x.get ? x.get(i) : c[x.key][i];
                  return (
                    <td key={x.key} className={x.num ? 'num' : undefined}>
                      {x.num ? fmtNum(v) : blankLabel(v) === '(none)' ? '' : String(v)}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Sources
// ---------------------------------------------------------------------------
function SourcesList({ data, onOnly }) {
  const [q, setQ] = useState('');
  const list = useMemo(() => {
    const s = q.trim().toLowerCase();
    const all = data?.sources ?? [];
    return s
      ? all.filter((x) => `${x.citation} ${x.source_key} ${x.doi}`.toLowerCase().includes(s))
      : all;
  }, [data, q]);
  if (!data) return <div className="screen-loading">Loading sources…</div>;
  return (
    <div className="md-table-card">
      <div className="md-table-tools">
        <span>
          {list.length} of {data.sources.length} published studies
        </span>
        <input
          className="input"
          style={{ maxWidth: 280 }}
          type="search"
          placeholder="Search author, title, DOI…"
          value={q}
          onChange={(e) => setQ(e.target.value)}
        />
      </div>
      <div className="md-table-scroll">
        <ul className="md-sources">
          {list.map((s) => (
            <li key={s.source_key}>
              <div className="md-source-cite">
                {s.citation || s.source_key}
                {s.doi && (
                  <>
                    {' '}
                    <a href={`https://doi.org/${s.doi}`} target="_blank" rel="noreferrer">
                      doi:{s.doi}
                    </a>
                  </>
                )}
              </div>
              <div className="md-source-actions">
                <button
                  type="button"
                  className="btn btn-ghost"
                  onClick={() => onOnly(s.source_key)}
                >
                  Plot only this
                </button>
              </div>
              <div className="md-source-meta">
                {s.source_key} · {s.n_rows} rows · {s.tables} · {s.extraction}
              </div>
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// One row, every column
// ---------------------------------------------------------------------------
const SECTIONS = [
  [
    'Value',
    [
      'property',
      'definition',
      'value',
      'unit',
      'u',
      'u_type',
      'u_k',
      'u_std',
      'u_source',
      'u_digitization',
    ],
  ],
  [
    'Conditions',
    [
      'T_K',
      'P_MPa',
      'u_T_K',
      'u_P_MPa',
      'u_digitization_T_K',
      'u_digitization_P_MPa',
      'substance',
      'state',
      'state_final',
      'ice_phase',
      'sample',
      'sample_detail',
      'method',
      'rate',
      'rate_unit',
    ],
  ],
  [
    'Provenance',
    [
      'location',
      'via',
      'extraction',
      'verification',
      'data_type',
      'corrections_applied',
      'value_orig',
      'unit_orig',
      'T_orig',
      'T_unit_orig',
      'T_scale_orig',
      'P_orig',
      'P_unit_orig',
    ],
  ],
  ['Fits and checks', ['used_in_fits', 'used_in_fits_level', 'compared_in', 'qc_flag', 'qc_note']],
  ['Notes', ['notes']],
  ['Identifiers', ['id', 'table', 'table_label', 'group', 'source_key', 'year']],
];

function DetailDrawer({ id, onClose }) {
  const q = useDataRow(id);
  const r = q.data?.row;
  const s = q.data?.source;
  const show = (v) => v !== null && v !== undefined && v !== '';
  return (
    <aside className="md-drawer" aria-label="Row details">
      <div className="md-drawer-head">
        <div>
          <div className="section-title">Measured value</div>
          <div className="md-release">{id}</div>
        </div>
        <button type="button" className="icon-btn" aria-label="close details" onClick={onClose}>
          ✕
        </button>
      </div>
      <div className="md-drawer-body">
        {q.error && <div className="plot-error">{String(q.error.message ?? q.error)}</div>}
        {!r && !q.error && <div className="md-muted">Loading…</div>}
        {r && (
          <>
            <h3>Source</h3>
            <div>
              {s?.citation || r.source_key}
              {s?.doi && (
                <>
                  {' '}
                  <a href={`https://doi.org/${s.doi}`} target="_blank" rel="noreferrer">
                    doi:{s.doi}
                  </a>
                </>
              )}
            </div>
            {SECTIONS.map(([title, keys]) => {
              const items = keys.filter((k) => show(r[k]));
              if (!items.length) return null;
              return (
                <div key={title}>
                  <h3>{title}</h3>
                  <dl className="md-kv">
                    {items.map((k) => (
                      <div key={k} style={{ display: 'contents' }}>
                        <dt>{k}</dt>
                        <dd>
                          {k === 'property'
                            ? q.data.property_label
                            : typeof r[k] === 'number'
                              ? fmtNum(r[k])
                              : k === 'used_in_fits'
                                ? fitsLabel(r[k])
                                : String(r[k])}
                        </dd>
                      </div>
                    ))}
                  </dl>
                </div>
              );
            })}
          </>
        )}
      </div>
    </aside>
  );
}
