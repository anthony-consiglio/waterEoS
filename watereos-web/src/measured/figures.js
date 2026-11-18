// Plotly figures for the measured-data views, built in the browser from the
// columnar /api/data/points payload. Styling follows watereos_api/theming.py
// so these plots match the server-built ones on the other screens.
import { axisTitle, blankLabel, fitsLabel, fmtNum, hoverText, unitPretty } from './format.js';

const FONT_SANS = '"Geist", ui-sans-serif, system-ui, -apple-system, sans-serif';
const FONT_MONO = '"Geist Mono", ui-monospace, "SF Mono", Menlo, monospace';

export const THEMES = {
  dark: {
    bg_elev: '#111114',
    text: '#ECECF0',
    text_muted: '#9398A2',
    text_faint: '#5E626A',
    grid: '#1A1A1E',
    grid_strong: '#25252B',
    border_strong: '#2A2A30',
    nml: 'rgba(91,141,239,0.10)',
  },
  light: {
    bg_elev: '#FFFFFF',
    text: '#0A0A0A',
    text_muted: '#6B7280',
    text_faint: '#9CA3AF',
    grid: '#EEEEF0',
    grid_strong: '#DCDCDF',
    border_strong: '#DEDEE1',
    nml: 'rgba(91,141,239,0.10)',
  },
};

// Categorical colors, readable on both themes.
export const PALETTE = [
  '#5b8def',
  '#ef4444',
  '#84cc16',
  '#f59e0b',
  '#a78bfa',
  '#38bdf8',
  '#ec4899',
  '#14b8a6',
  '#f97316',
  '#22c55e',
  '#eab308',
  '#8b5cf6',
  '#06b6d4',
  '#e11d48',
  '#65a30d',
  '#d946ef',
  '#0ea5e9',
  '#fb7185',
  '#10b981',
  '#c084fc',
  '#facc15',
  '#2dd4bf',
  '#fb923c',
  '#94a3b8',
];

// Continuous scale (blue -> red), also used to color model curves by their
// isobar pressure or isotherm temperature.
export const SCALE = [
  [0, '#3b82f6'],
  [0.25, '#06b6d4'],
  [0.5, '#84cc16'],
  [0.75, '#f59e0b'],
  [1, '#ef4444'],
];
const DASHES = [
  'solid',
  'dash',
  'dot',
  'dashdot',
  'longdash',
  'longdashdot',
  '5px,2px,1px,2px',
  '1px,3px',
];

function hexRgb(h) {
  const v = parseInt(h.slice(1), 16);
  return [(v >> 16) & 255, (v >> 8) & 255, v & 255];
}

export function scaleColor(t) {
  const x = Math.min(1, Math.max(0, Number.isFinite(t) ? t : 0.5));
  for (let k = 1; k < SCALE.length; k++) {
    if (x <= SCALE[k][0]) {
      const [t0, c0] = SCALE[k - 1];
      const [t1, c1] = SCALE[k];
      const f = (x - t0) / (t1 - t0 || 1);
      const a = hexRgb(c0);
      const b = hexRgb(c1);
      return `rgb(${a.map((v, j) => Math.round(v + f * (b[j] - v))).join(',')})`;
    }
  }
  return SCALE[SCALE.length - 1][1];
}

function withAlpha(color, alpha) {
  if (color.startsWith('#')) return `rgba(${hexRgb(color).join(',')},${alpha})`;
  if (color.startsWith('rgb(')) return color.replace('rgb(', 'rgba(').replace(')', `,${alpha})`);
  return color;
}

function axis(c, title, log, range) {
  const a = {
    title: { text: title, font: { color: c.text_muted, size: 12 } },
    gridcolor: c.grid,
    zerolinecolor: c.grid_strong,
    linecolor: c.grid_strong,
    tickfont: { family: FONT_MONO, color: c.text_faint, size: 11 },
    type: log ? 'log' : 'linear',
    exponentformat: 'power',
  };
  if (range) a.range = range;
  return a;
}

function baseLayout(theme, { xtitle, ytitle, logx, logy, xrange, yrange, uirevision }) {
  const c = THEMES[theme] ?? THEMES.dark;
  return {
    paper_bgcolor: c.bg_elev,
    plot_bgcolor: c.bg_elev,
    font: { family: FONT_SANS, color: c.text, size: 12 },
    margin: { l: 72, r: 24, t: 16, b: 52 },
    xaxis: axis(c, xtitle, logx, xrange),
    yaxis: axis(c, ytitle, logy, yrange),
    legend: {
      font: { family: FONT_MONO, size: 11, color: c.text_muted },
      bgcolor: 'rgba(0,0,0,0)',
      itemclick: 'toggle',
      itemdoubleclick: 'toggleothers',
    },
    hoverlabel: {
      bgcolor: c.bg_elev,
      bordercolor: c.border_strong,
      font: { family: FONT_MONO, size: 11, color: c.text },
      align: 'left',
    },
    hovermode: 'closest',
    uirevision,
  };
}

function sceneLayout(theme, { xtitle, ytitle, ztitle, logz, ranges, uirevision }) {
  const c = THEMES[theme] ?? THEMES.dark;
  const ax = (t, log, r) => ({
    ...axis(c, t, log, r),
    backgroundcolor: c.bg_elev,
    showbackground: true,
  });
  return {
    paper_bgcolor: c.bg_elev,
    font: { family: FONT_SANS, color: c.text, size: 12 },
    margin: { l: 0, r: 0, t: 8, b: 0 },
    scene: {
      xaxis: ax(xtitle, false, ranges?.x),
      yaxis: ax(ytitle, false, ranges?.y),
      zaxis: ax(ztitle, logz, ranges?.z),
      bgcolor: c.bg_elev,
      aspectmode: 'cube',
      camera: { eye: { x: 1.6, y: -1.6, z: 0.9 } },
    },
    legend: {
      font: { family: FONT_MONO, size: 11, color: c.text_muted },
      bgcolor: 'rgba(0,0,0,0)',
    },
    hoverlabel: {
      bgcolor: c.bg_elev,
      bordercolor: c.border_strong,
      font: { family: FONT_MONO, size: 11, color: c.text },
      align: 'left',
    },
    uirevision,
  };
}

// Padded [lo, hi] of finite values (in log10 units for a log axis).
export function paddedRange(values, log = false, frac = 0.05) {
  let lo = Infinity;
  let hi = -Infinity;
  for (const v0 of values) {
    if (v0 == null || !Number.isFinite(v0)) continue;
    if (log && v0 <= 0) continue;
    const v = log ? Math.log10(v0) : v0;
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  if (!Number.isFinite(lo)) return null;
  const span = hi - lo || Math.abs(hi) * 0.1 || 1;
  return [lo - frac * span, hi + frac * span];
}

// ---------------------------------------------------------------------------
// Color modes
// ---------------------------------------------------------------------------
export const COLOR_MODES = [
  { value: 'other', label: 'Other variable (P or T)' },
  { value: 'source_key', label: 'Source' },
  { value: 'data_type', label: 'Data type' },
  { value: 'extraction', label: 'Extraction' },
  { value: 'state', label: 'State' },
  { value: 'sample', label: 'Sample' },
  { value: 'substance', label: 'Substance' },
  { value: 'qc_flag', label: 'QC flag' },
  { value: 'method', label: 'Method' },
  { value: 'ice_phase', label: 'Ice phase' },
  { value: 'fits', label: 'Used in fits' },
  { value: 'year', label: 'Year' },
  { value: 'rate_K_min', label: 'Heating/cooling rate' },
];
export const COVERAGE_COLOR_MODES = [
  { value: 'table', label: 'Table' },
  { value: 'group', label: 'Group' },
  ...COLOR_MODES.filter((m) => !['other', 'year', 'rate_K_min'].includes(m.value)),
];
const CONTINUOUS = new Set(['other', 'year', 'rate_K_min']);

function categoryOf(c, i, mode, names) {
  if (mode === 'fits') return fitsLabel(c.used_in_fits[i]);
  if (mode === 'table') return names?.tables?.[c.table[i]] ?? c.table[i];
  if (mode === 'group') return names?.groups?.[c.group[i]] ?? c.group[i];
  if (mode === 'source_key') return names?.cites?.[c.source_key[i]] ?? c.source_key[i];
  return blankLabel(c[mode][i]);
}

function symbolsFor(c, idx, three, nonThermo) {
  return idx.map((i) => {
    const open = c.substance[i] === 'D2O';
    if (nonThermo && nonThermo.has(c.group[i])) return open ? 'x-open' : 'x';
    if (three) return open ? 'circle-open' : 'circle';
    return open ? 'circle-open' : 'circle';
  });
}

// Error bars of the rows `idx`; zero where no uncertainty is given. With
// `keepEmpty` false, undefined when none is given at all.
function errorBar(arr, idx, color, keepEmpty) {
  const a = idx.map((i) => (arr && arr[i] != null && Number.isFinite(arr[i]) ? arr[i] : 0));
  if (!keepEmpty && !a.some((v) => v > 0)) return undefined;
  return { type: 'data', array: a, visible: true, thickness: 1, width: 0, color };
}

// Marker traces for the rows `idx`, colored by `colorMode`.
function markerTraces(o) {
  const { c, idx, X, Y, Z, colorMode, other, theme, errors, three, hover, names, nonThermo } = o;
  const th = THEMES[theme] ?? THEMES.dark;
  const type = three ? 'scatter3d' : 'scattergl';
  const size = three ? 3.2 : 6;
  const traces = [];
  let coloraxis;
  const xyz = (ids) => {
    const t = { x: ids.map((i) => X[i]), y: ids.map((i) => Y[i]) };
    if (three) t.z = ids.map((i) => Z[i]);
    return t;
  };
  // In 2D, every trace carries both error_x and error_y while error bars are
  // on, even when no row of it has an uncertainty. Plotly's WebGL error bars
  // (regl-error2d) keep the previous bars of a trace whose error_x or error_y
  // disappears, with their old point count and offset; when the plot then has
  // fewer points, the update throws ("offset is out of bounds"), Plotly is
  // left half-updated, and the next update draws the old points again on the
  // new axes.
  const errs = (ids, color) => {
    const e = {};
    if (!errors) return e;
    const ex = errorBar(errors.x, ids, color, !three);
    const ey = errorBar(errors.y, ids, color, !three);
    const ez = three ? errorBar(errors.z, ids, color, false) : undefined;
    if (ex) e.error_x = ex;
    if (ey) e.error_y = ey;
    if (ez) e.error_z = ez;
    return e;
  };
  const common = (ids) => ({
    type,
    mode: 'markers',
    customdata: ids.map((i) => c.id[i]),
    text: hover ? ids.map((i) => hover(i)) : undefined,
    hovertemplate: hover ? '%{text}<extra></extra>' : undefined,
  });

  if (CONTINUOUS.has(colorMode)) {
    const key = colorMode === 'other' ? other.key : colorMode;
    const raw = c[key];
    const isLog = colorMode === 'rate_K_min';
    const val = (i) => {
      const v = raw[i];
      if (v == null || !Number.isFinite(v)) return null;
      if (isLog) return v > 0 ? Math.log10(v) : null;
      return v;
    };
    const has = idx.filter((i) => val(i) != null);
    const miss = idx.filter((i) => val(i) == null);
    const vals = has.map(val);
    const lo = Math.min(...vals);
    const hi = Math.max(...vals);
    const title =
      colorMode === 'other' ? other.label : colorMode === 'year' ? 'Year' : 'log₁₀ rate (K/min)';
    if (has.length) {
      coloraxis = {
        colorscale: SCALE,
        cmin: lo,
        cmax: hi === lo ? lo + 1 : hi,
        colorbar: {
          title: { text: title, font: { color: th.text_muted, size: 11 } },
          tickfont: { family: FONT_MONO, color: th.text_faint, size: 10 },
          thickness: 12,
          len: 0.6,
          y: 0,
          yanchor: 'bottom',
          outlinewidth: 0,
        },
      };
      traces.push({
        ...common(has),
        ...xyz(has),
        ...errs(has, withAlpha(th.text_faint, 0.6)),
        name: 'data',
        showlegend: false,
        marker: {
          size,
          color: vals,
          coloraxis: 'coloraxis',
          symbol: symbolsFor(c, has, three, nonThermo),
          line: { width: three ? 0 : 0.5, color: 'rgba(0,0,0,0.25)' },
        },
      });
    }
    if (miss.length) {
      traces.push({
        ...common(miss),
        ...xyz(miss),
        ...errs(miss, withAlpha(th.text_faint, 0.6)),
        name: `no ${title.toLowerCase()} (${miss.length})`,
        marker: { size, color: th.text_faint, symbol: symbolsFor(c, miss, three, nonThermo) },
      });
    }
    return { traces, coloraxis };
  }

  const groups = new Map();
  for (const i of idx) {
    const k = categoryOf(c, i, colorMode, names);
    if (!groups.has(k)) groups.set(k, []);
    groups.get(k).push(i);
  }
  const ordered = [...groups.entries()].sort(
    (a, b) => b[1].length - a[1].length || a[0].localeCompare(b[0])
  );
  ordered.forEach(([k, ids], j) => {
    const color = PALETTE[j % PALETTE.length];
    traces.push({
      ...common(ids),
      ...xyz(ids),
      ...errs(ids, withAlpha(color, 0.45)),
      name: `${k} (${ids.length})`,
      legendgroup: `cat:${k}`,
      marker: {
        size,
        color,
        symbol: symbolsFor(c, ids, three, nonThermo),
        line: { width: three ? 0 : 0.5, color: 'rgba(0,0,0,0.25)' },
      },
    });
  });
  return { traces, coloraxis };
}

// ---------------------------------------------------------------------------
// Reference lines
// ---------------------------------------------------------------------------
function lineTraces(
  lines,
  theme,
  { melt = true, th = true, vapor = false, three = false, z0 = 0, clip } = {}
) {
  if (!lines) return [];
  const c = THEMES[theme] ?? THEMES.dark;
  const out = [];
  const keep = (T, P) => {
    if (!clip) return { T, P };
    const t = [];
    const p = [];
    T.forEach((v, k) => {
      const inside = v >= clip.T[0] && v <= clip.T[1] && P[k] >= clip.P[0] && P[k] <= clip.P[1];
      t.push(inside ? v : null);
      p.push(inside ? P[k] : null);
    });
    return { T: t, P: p };
  };
  const add = (name, group, T0, P0, dash, show) => {
    const { T, P } = keep(T0, P0);
    const t = {
      type: three ? 'scatter3d' : 'scatter',
      mode: 'lines',
      x: T,
      y: P,
      name,
      legendgroup: group,
      showlegend: show,
      line: { color: c.text_muted, width: three ? 3 : 1.5, dash },
      hovertemplate: `${name}<br>T = %{x:.2f} K<br>P = %{y:.4g} MPa<extra></extra>`,
    };
    if (three) t.z = T.map(() => z0);
    out.push(t);
  };
  if (melt) {
    lines.melting.branches.forEach((b, k) =>
      add(`Melting, ${lines.melting.source}`, 'melt', b.T_K, b.P_MPa, 'solid', k === 0)
    );
  }
  if (th) add(`T_H, ${lines.th.source}`, 'th', lines.th.T_K, lines.th.P_MPa, 'dash', true);
  if (vapor) {
    lines.vapor.branches.forEach((b) =>
      add(
        `Vapor pressure over ${b.label}, ${lines.vapor.source}`,
        `vap:${b.label}`,
        b.T_K,
        b.P_MPa,
        b.label === 'liquid' ? 'dot' : 'dashdot',
        true
      )
    );
  }
  return out;
}

// ---------------------------------------------------------------------------
// Model curves (2D)
// ---------------------------------------------------------------------------
function curveTraces(curves, modelLabels, along, theme, [lo, hi]) {
  const c = THEMES[theme] ?? THEMES.dark;
  const out = [];
  const order = [...new Set(curves.map((k) => k.model))];
  const fixedName = along === 'T' ? 'P' : 'T';
  const fixedUnit = along === 'T' ? 'MPa' : 'K';
  order.forEach((m, j) => {
    const dash = DASHES[j % DASHES.length];
    const label = modelLabels[m] ?? m;
    out.push({
      type: 'scatter',
      mode: 'lines',
      x: [null],
      y: [null],
      name: label,
      legendgroup: `model:${m}`,
      line: { color: c.text_muted, width: 2, dash },
      hoverinfo: 'skip',
    });
    for (const cv of curves.filter((k) => k.model === m)) {
      const color = scaleColor((cv.fixed - lo) / (hi - lo || 1));
      const n = cv.x.length;
      const yin = cv.y.map((y, k) => (cv.in_range[k] ? y : null));
      const yout = cv.y.map((y, k) =>
        !cv.in_range[k] || (k > 0 && !cv.in_range[k - 1]) || (k < n - 1 && !cv.in_range[k + 1])
          ? y
          : null
      );
      const tip =
        `${label}<br>${fixedName} = ${fmtNum(cv.fixed)} ${fixedUnit}` +
        `<br>${along} = %{x:.4g}<br>%{y:.6g}`;
      out.push({
        type: 'scatter',
        mode: 'lines',
        x: cv.x,
        y: yin,
        legendgroup: `model:${m}`,
        showlegend: false,
        line: { color, width: 2, dash },
        hovertemplate: `${tip}<extra></extra>`,
      });
      out.push({
        type: 'scatter',
        mode: 'lines',
        x: cv.x,
        y: yout,
        legendgroup: `model:${m}`,
        showlegend: false,
        opacity: 0.3,
        line: { color, width: 2, dash },
        hovertemplate: `${tip}<br>(outside the model's stated range)<extra></extra>`,
      });
    }
  });
  return out;
}

// ---------------------------------------------------------------------------
// Property view
// ---------------------------------------------------------------------------
export function propertyFigure2d(o) {
  const {
    c,
    idx,
    prop,
    xVar,
    colorMode,
    errorBars,
    logX,
    logY,
    theme,
    names,
    curves,
    modelLabels,
    at,
    lines,
  } = o;
  const locus = prop.locus;
  const T = c.T_K;
  const P = c.P_MPa;
  const hover = (i) => hoverText(c, i, names.cites, prop.label, prop.unit);
  let X;
  let Y;
  let errors;
  let xtitle;
  let ytitle;
  let mode = colorMode;
  let other;
  if (locus) {
    // Loci are curves in the T-P plane.
    X = T;
    Y = P;
    errors = errorBars ? { x: c.u_T_K, y: c.u_P_MPa } : null;
    xtitle = 'Temperature T (K)';
    ytitle = 'Pressure P (MPa)';
    if (mode === 'other') mode = 'source_key';
  } else {
    X = xVar === 'T' ? T : P;
    Y = c.value;
    errors = errorBars ? { x: xVar === 'T' ? c.u_T_K : c.u_P_MPa, y: c.u_std } : null;
    xtitle = xVar === 'T' ? 'Temperature T (K)' : 'Pressure P (MPa)';
    ytitle = axisTitle(prop.label, prop.unit);
    other = xVar === 'T' ? { key: 'P_MPa', label: 'P (MPa)' } : { key: 'T_K', label: 'T (K)' };
  }
  const pts = idx.filter((i) => X[i] != null && Y[i] != null);
  const { traces, coloraxis } = markerTraces({
    c,
    idx: pts,
    X,
    Y,
    colorMode: mode,
    other,
    theme,
    errors,
    hover,
    names,
  });
  const extra = [];
  if (locus) {
    extra.push(...lineTraces(lines, theme, { vapor: prop.key === 'vapor_pressure' }));
  } else if (curves?.length) {
    const fixedVals = curves.map((k) => k.fixed);
    let cr = [Math.min(...fixedVals), Math.max(...fixedVals)];
    if (mode === 'other' && coloraxis) cr = [coloraxis.cmin, coloraxis.cmax];
    extra.push(...curveTraces(curves, modelLabels, xVar, theme, cr));
  }
  const layout = baseLayout(theme, {
    xtitle,
    ytitle,
    logx: logX,
    logy: logY,
    xrange: paddedRange(
      pts.map((i) => X[i]),
      logX,
      0.03
    ),
    yrange: paddedRange(
      pts.map((i) => Y[i]),
      logY,
      0.08
    ),
    uirevision: o.uirevision,
  });
  if (coloraxis) layout.coloraxis = coloraxis;
  return { data: [...traces, ...extra], layout, n: pts.length, at };
}

export function propertyFigure3d(o) {
  const {
    c,
    idx,
    prop,
    xVar,
    colorMode,
    errorBars,
    logZ,
    theme,
    names,
    surfaces,
    modelLabels,
    lines,
    floorLines,
  } = o;
  const hover = (i) => hoverText(c, i, names.cites, prop.label, prop.unit);
  const pts = idx.filter((i) => c.T_K[i] != null && c.P_MPa[i] != null && c.value[i] != null);
  const other = xVar === 'T' ? { key: 'P_MPa', label: 'P (MPa)' } : { key: 'T_K', label: 'T (K)' };
  const { traces, coloraxis } = markerTraces({
    c,
    idx: pts,
    X: c.T_K,
    Y: c.P_MPa,
    Z: c.value,
    colorMode,
    other,
    theme,
    three: true,
    hover,
    names,
    errors: errorBars ? { x: c.u_T_K, y: c.u_P_MPa, z: c.u_std } : null,
  });
  const zr = paddedRange(
    pts.map((i) => c.value[i]),
    logZ,
    0.08
  );
  const zSpan = zr ? zr[1] - zr[0] : 1;
  const zClip = zr ? [zr[0] - 0.4 * zSpan, zr[1] + 0.4 * zSpan] : null;
  const extra = [];
  (surfaces || []).forEach((s, j) => {
    const color = PALETTE[(j + 4) % PALETTE.length];
    const clipZ = (v) => {
      if (v == null) return null;
      const t = logZ ? (v > 0 ? Math.log10(v) : null) : v;
      if (t == null || (zClip && (t < zClip[0] || t > zClip[1]))) return null;
      return v;
    };
    const zAll = s.z.map((row) => row.map(clipZ));
    const zIn = zAll.map((row, a) => row.map((v, b) => (s.in_range[a][b] ? v : null)));
    const label = modelLabels[s.model] ?? s.model;
    const base = {
      type: 'surface',
      x: s.T_K,
      y: s.P_MPa,
      colorscale: [
        [0, color],
        [1, color],
      ],
      showscale: false,
      hovertemplate: `${label}<br>T = %{x:.4g} K<br>P = %{y:.4g} MPa<br>%{z:.6g}<extra></extra>`,
      legendgroup: `model:${s.model}`,
    };
    extra.push({
      ...base,
      z: zAll,
      opacity: 0.15,
      name: `${label} (outside range)`,
      showlegend: false,
    });
    extra.push({ ...base, z: zIn, opacity: 0.55, name: label, showlegend: true });
  });
  if (floorLines && zr) {
    const tr = paddedRange(pts.map((i) => c.T_K[i]));
    const pr = paddedRange(pts.map((i) => c.P_MPa[i]));
    if (tr && pr) {
      const z0 = logZ ? 10 ** zr[0] : zr[0];
      extra.push(...lineTraces(lines, theme, { three: true, z0, clip: { T: tr, P: pr } }));
    }
  }
  const layout = sceneLayout(theme, {
    xtitle: 'T (K)',
    ytitle: 'P (MPa)',
    ztitle: axisTitle(prop.label, prop.unit),
    logz: logZ,
    ranges: { z: zr },
    uirevision: o.uirevision,
  });
  if (coloraxis) layout.coloraxis = coloraxis;
  return { data: [...traces, ...extra], layout, n: pts.length };
}

// ---------------------------------------------------------------------------
// Coverage in T-P
// ---------------------------------------------------------------------------
// T_H at each pressure, by interpolation of the reference line (null outside it).
export function thAt(lines, P) {
  const Ps = lines.th.P_MPa;
  const Ts = lines.th.T_K;
  if (P < Ps[0] || P > Ps[Ps.length - 1]) return null;
  for (let k = 1; k < Ps.length; k++) {
    if (P <= Ps[k]) {
      const f = (P - Ps[k - 1]) / (Ps[k] - Ps[k - 1] || 1);
      return Ts[k - 1] + f * (Ts[k] - Ts[k - 1]);
    }
  }
  return Ts[Ts.length - 1];
}

// No man's land as outlined by the database's reports: 150 K <= T < T_H(P), 0-1000 MPa.
export function inNoMansLand(lines, T, P) {
  if (T == null || P == null || P < 0 || P > 1000 || T < 150) return false;
  const th = thAt(lines, P);
  return th != null && T < th;
}

function nmlTrace(lines, theme) {
  const c = THEMES[theme] ?? THEMES.dark;
  const Ps = [];
  for (let p = 0; p <= 1000; p += 10) Ps.push(p);
  const right = Ps.map((p) => thAt(lines, p));
  const x = [150, ...right, 150, 150];
  const y = [0, ...Ps, 1000, 0];
  return {
    type: 'scatter',
    mode: 'lines',
    x,
    y,
    fill: 'toself',
    fillcolor: c.nml,
    line: { color: 'rgba(91,141,239,0.55)', width: 1 },
    name: "No man's land (150 K ≤ T < T_H, 0–1000 MPa)",
    hoverinfo: 'skip',
  };
}

export function coverageFigure(o) {
  const { c, idx, colorMode, mode, theme, names, lines, showNml, logY, binT, binP, nonThermo } = o;
  const pts = idx.filter((i) => c.T_K[i] != null && c.P_MPa[i] != null);
  const data = [];
  let coloraxis;
  const th = THEMES[theme] ?? THEMES.dark;
  if (showNml && lines) data.push(nmlTrace(lines, theme));
  if (mode === 'counts') {
    const t0 = Math.floor(Math.min(...pts.map((i) => c.T_K[i])) / binT) * binT;
    const p0 = Math.floor(Math.min(...pts.map((i) => c.P_MPa[i])) / binP) * binP;
    const counts = new Map();
    let nt = 0;
    let np = 0;
    for (const i of pts) {
      const a = Math.floor((c.T_K[i] - t0) / binT);
      const b = Math.floor((c.P_MPa[i] - p0) / binP);
      nt = Math.max(nt, a + 1);
      np = Math.max(np, b + 1);
      const k = `${a},${b}`;
      counts.set(k, (counts.get(k) || 0) + 1);
    }
    const z = [];
    const zc = [];
    for (let b = 0; b < np; b++) {
      const row = [];
      const rowc = [];
      for (let a = 0; a < nt; a++) {
        const n = counts.get(`${a},${b}`) || 0;
        row.push(n ? Math.log10(n) : null);
        rowc.push(n);
      }
      z.push(row);
      zc.push(rowc);
    }
    const maxLog = Math.max(1, ...z.flat().filter((v) => v != null));
    const ticks = [0, 1, 2, 3, 4].filter((v) => v <= maxLog + 0.01);
    data.push({
      type: 'heatmap',
      x: Array.from({ length: nt }, (_, a) => t0 + (a + 0.5) * binT),
      y: Array.from({ length: np }, (_, b) => p0 + (b + 0.5) * binP),
      z,
      customdata: zc,
      colorscale: SCALE,
      zmin: 0,
      zmax: maxLog,
      hovertemplate: 'T ≈ %{x:.4g} K, P ≈ %{y:.4g} MPa<br>%{customdata} rows<extra></extra>',
      colorbar: {
        title: { text: 'rows per bin', font: { color: th.text_muted, size: 11 } },
        tickvals: ticks,
        ticktext: ticks.map((v) => String(10 ** v)),
        tickfont: { family: FONT_MONO, color: th.text_faint, size: 10 },
        thickness: 12,
        len: 0.6,
        y: 0,
        yanchor: 'bottom',
        outlinewidth: 0,
      },
    });
  } else {
    const hover = (i) => {
      const prop = names.props?.[c.property[i]] ?? c.property[i];
      return (
        `<b>${names.cites?.[c.source_key[i]] ?? c.source_key[i]}</b><br>` +
        `${names.tables?.[c.table[i]] ?? c.table[i]} · ${prop}<br>` +
        `T = ${fmtNum(c.T_K[i])} K, P = ${fmtNum(c.P_MPa[i])} MPa<br>` +
        `${fmtNum(c.value[i])} ${unitPretty(c.unit[i])} · ${c.data_type[i]} · ${c.substance[i]}<br>` +
        '<i>click for every column</i>'
      );
    };
    const r = markerTraces({
      c,
      idx: pts,
      X: c.T_K,
      Y: c.P_MPa,
      colorMode,
      theme,
      hover,
      names,
      nonThermo,
    });
    data.push(...r.traces);
    coloraxis = r.coloraxis;
  }
  if (lines) data.push(...lineTraces(lines, theme, { melt: true, th: true }));
  const layout = baseLayout(theme, {
    xtitle: 'Temperature T (K)',
    ytitle: 'Pressure P (MPa)',
    logy: logY,
    xrange: paddedRange(
      pts.map((i) => c.T_K[i]),
      false,
      0.03
    ),
    yrange: paddedRange(
      pts.map((i) => c.P_MPa[i]),
      logY,
      0.05
    ),
    uirevision: o.uirevision,
  });
  if (coloraxis) layout.coloraxis = coloraxis;
  return { data, layout, n: pts.length };
}
