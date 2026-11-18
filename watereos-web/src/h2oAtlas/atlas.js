// The logic of the cryo-isochoric atlas page (cryo_isochoric/gallery.py), as
// plain functions: composition units, figure conversion, figure geometry.
// Every figure is computed and stored in wt %; the page redraws it in the
// other units. The process view of the atlas is not part of this tab.

// ---- composition units --------------------------------------------------
export const MW = 18.0153; // g/mol
export const RHO_W = 0.99705; // g/cm^3: water at 25 °C, for vol %

// Maps from wt %. S is the solute: its molar mass S.M [g/mol] and the density
// of the pure substance S.rho [g/cm^3]. vol % is the volume of the pure solute
// in the volume of the unmixed components at 25 °C (how a v/v solution is
// made up).
export const UNITS = {
  wt: { name: 'wt %', sym: 'x', unit: 'wt %', f: (w) => w },
  mol: {
    name: 'mol %',
    sym: 'x',
    unit: 'mol %',
    f: (w, S) => (100 * (w / S.M)) / (w / S.M + (100 - w) / MW),
  },
  molal: { name: 'mol/kg', sym: 'm', unit: 'mol/kg', f: (w, S) => (1000 * w) / (S.M * (100 - w)) },
  vol: {
    name: 'vol %',
    sym: 'x',
    unit: 'vol %',
    f: (w, S) => (100 * (w / S.rho)) / (w / S.rho + (100 - w) / RHO_W),
  },
};

// vol % needs the density of the pure solute: without it, wt %.
export const uFor = (S, u) => (u === 'vol' && !(S && S.rho) ? 'wt' : u);
export const toU = (w, S, u) => UNITS[uFor(S, u)].f(w, S);

export const num = (v) =>
  !isFinite(v)
    ? '∞'
    : Math.abs(v) >= 100
      ? v.toFixed(0)
      : Math.abs(v) >= 10
        ? v.toFixed(1)
        : v.toFixed(2);

// A composition w (wt %) as text in the unit u.
export function comp(w, S, u) {
  const k = uFor(S, u);
  return k === 'wt' ? w.toFixed(1) + ' wt %' : num(UNITS[k].f(w, S)) + ' ' + UNITS[k].unit;
}

// ---- figures --------------------------------------------------------------
const TYPES = {
  f4: Float32Array,
  f8: Float64Array,
  i1: Int8Array,
  u1: Uint8Array,
  i2: Int16Array,
  u2: Uint16Array,
  i4: Int32Array,
  u4: Uint32Array,
};

// Plotly's {dtype, bdata} array -> typed array (anything else as it is).
export function decode(v) {
  if (
    v == null ||
    typeof v !== 'object' ||
    Array.isArray(v) ||
    ArrayBuffer.isView(v) ||
    v.bdata === undefined
  )
    return v;
  const bin = atob(v.bdata);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new TYPES[v.dtype](bytes.buffer);
}
const bad = (v) => v == null || v !== v;

// '… wt %' in a label, hover text or hover template, in the unit U.
export function compText(s, U, S) {
  if (typeof s !== 'string' || s.indexOf('wt %') < 0) return s;
  return s
    .replace(
      /x = %\{([^}:]+)(?::[^}]*)?\} wt %/g,
      (m, k) => U.sym + ' = %{' + k + ':.2f} ' + U.unit
    )
    .replace(/%\{([^}:]+)(?::[^}]*)?\} wt %/g, (m, k) => '%{' + k + ':.2f} ' + U.unit)
    .replace(
      /x = (-?\d+(?:\.\d+)?) wt %/g,
      (m, a) => U.sym + ' = ' + num(U.f(parseFloat(a), S)) + ' ' + U.unit
    )
    .replace(/(-?\d+(?:\.\d+)?) wt %/g, (m, a) => num(U.f(parseFloat(a), S)) + ' ' + U.unit)
    .replace(/x \(wt %\)/g, U.sym + ' (' + U.unit + ')')
    .replace(/\(wt %\)/g, '(' + U.unit + ')');
}

// A straight line in wt % is a curve in mol % or molality (tie-lines, the
// edges of tie-triangles), so paths are first refined in the coordinates they
// were drawn in and then mapped point by point.
export function dense2d(tr, ax, f, step, logOther) {
  const other = ax === 'x' ? 'y' : 'x';
  const a = decode(tr[ax]);
  const b = decode(tr[other]);
  if (!a || !b || a.length !== b.length) return tr;
  const lines = !tr.mode || tr.mode.indexOf('lines') >= 0;
  const A = [];
  const B = [];
  for (let i = 0; i < a.length; i++) {
    if (lines && i > 0 && !bad(a[i]) && !bad(a[i - 1]) && !bad(b[i]) && !bad(b[i - 1])) {
      const n = Math.min(400, Math.ceil(Math.abs(a[i] - a[i - 1]) / step));
      for (let k = 1; k < n; k++) {
        const t = k / n;
        A.push(f(a[i - 1] + t * (a[i] - a[i - 1])));
        B.push(
          logOther
            ? Math.exp(Math.log(b[i - 1]) + t * (Math.log(b[i]) - Math.log(b[i - 1])))
            : b[i - 1] + t * (b[i] - b[i - 1])
        );
      }
    }
    A.push(bad(a[i]) ? a[i] : f(a[i]));
    B.push(b[i]);
  }
  return Object.assign({}, tr, { [ax]: A, [other]: B });
}

// A 3-D polyline, NaN-separated.
export function dense3d(tr, ax, f, step) {
  const X = { x: decode(tr.x), y: decode(tr.y), z: decode(tr.z) };
  if (!X.x || !X.y || !X.z) return tr;
  const O = { x: [], y: [], z: [] };
  const lines = (tr.mode || '').indexOf('lines') >= 0;
  const texts = Array.isArray(tr.text) ? [] : null;
  for (let i = 0; i < X.x.length; i++) {
    if (lines && i > 0 && ['x', 'y', 'z'].every((c) => !bad(X[c][i]) && !bad(X[c][i - 1]))) {
      const n = Math.min(200, Math.ceil(Math.abs(X[ax][i] - X[ax][i - 1]) / step));
      for (let k = 1; k < n; k++) {
        for (const c of 'xyz') {
          const v = X[c][i - 1] + (k / n) * (X[c][i] - X[c][i - 1]);
          O[c].push(c === ax ? f(v) : v);
        }
        if (texts) texts.push(tr.text[i]);
      }
    }
    for (const c of 'xyz') {
      const v = X[c][i];
      O[c].push(c === ax && !bad(v) ? f(v) : v);
    }
    if (texts) texts.push(tr.text[i]);
  }
  const out = Object.assign({}, tr, O);
  if (texts) out.text = texts;
  return out;
}

// A triangle mesh: triangles that span much composition are subdivided.
export function denseMesh(tr, ax, f, step) {
  const X = { x: decode(tr.x), y: decode(tr.y), z: decode(tr.z) };
  const I = decode(tr.i);
  const J = decode(tr.j);
  const K = decode(tr.k);
  if (!X.x || !I || !I.length) return tr;
  const inten = decode(tr.intensity);
  const cd0 = decode(tr.customdata);
  const cd = cd0 && !(tr.customdata && tr.customdata.shape) && !Array.isArray(cd0[0]) ? cd0 : null;
  const O = { x: [], y: [], z: [] };
  const oi = [];
  const oj = [];
  const ok = [];
  const on = inten ? [] : null;
  const oc = cd ? [] : null;
  const w = X[ax];
  const ids = new Map();
  const put = (key, a, b, c, wa, wb, wc) => {
    let id = ids.get(key);
    if (id !== undefined) return id;
    id = O.x.length;
    for (const q of 'xyz') {
      const v = wa * X[q][a] + wb * X[q][b] + wc * X[q][c];
      O[q].push(q === ax ? f(v) : v);
    }
    if (on) on.push(wa * inten[a] + wb * inten[b] + wc * inten[c]);
    if (oc) oc.push(wa * cd[a] + wb * cd[b] + wc * cd[c]);
    ids.set(key, id);
    return id;
  };
  // q/n of the way from a to b
  const edge = (a, b, n, q) =>
    a < b ? a + '_' + b + '_' + n + '_' + q : b + '_' + a + '_' + n + '_' + (n - q);
  for (let t = 0; t < I.length; t++) {
    const a = I[t];
    const b = J[t];
    const c = K[t];
    const n = Math.max(
      1,
      Math.min(8, Math.ceil((Math.max(w[a], w[b], w[c]) - Math.min(w[a], w[b], w[c])) / step))
    );
    const row = [];
    for (let p = 0; p <= n; p++) {
      row.push([]);
      for (let q = 0; q <= n - p; q++) {
        // barycentric weights: a (n-p-q)/n, b p/n, c q/n
        const r = n - p - q;
        const key =
          r === n
            ? 'v' + a
            : p === n
              ? 'v' + b
              : q === n
                ? 'v' + c
                : q === 0
                  ? edge(a, b, n, p)
                  : p === 0
                    ? edge(a, c, n, q)
                    : r === 0
                      ? edge(b, c, n, q)
                      : 't' + t + '_' + p + '_' + q;
        row[p].push(put(key, a, b, c, r / n, p / n, q / n));
      }
    }
    for (let p = 0; p < n; p++)
      for (let q = 0; q < n - p; q++) {
        oi.push(row[p][q]);
        oj.push(row[p + 1][q]);
        ok.push(row[p][q + 1]);
        if (q < n - p - 1) {
          oi.push(row[p + 1][q]);
          oj.push(row[p + 1][q + 1]);
          ok.push(row[p][q + 1]);
        }
      }
  }
  const out = Object.assign({}, tr, O, { i: oi, j: oj, k: ok });
  if (on) out.intensity = on;
  if (oc) out.customdata = oc;
  return out;
}

// The figure in another composition unit (``key`` already resolved with
// uFor). layout.meta.comp, set by the atlas, says where the composition lives.
export function inUnit(fig, key, S) {
  const meta = fig.layout && fig.layout.meta && fig.layout.meta.comp;
  const U = UNITS[key];
  if (key === 'wt' || !meta || meta.mass_only) return fig;
  const f = (w) => U.f(w, S);
  const txt = (s) => compText(s, U, S);
  const layout = JSON.parse(JSON.stringify(fig.layout));
  const texts = (tr) => {
    const o = {};
    for (const k of ['name', 'hovertemplate']) if (typeof tr[k] === 'string') o[k] = txt(tr[k]);
    if (typeof tr.text === 'string') o.text = txt(tr.text);
    else if (
      Array.isArray(tr.text) &&
      typeof tr.text[0] === 'string' &&
      tr.text[0].indexOf('wt %') >= 0
    )
      o.text = tr.text.map(txt);
    return o;
  };
  let one = (tr) => Object.assign({}, tr, texts(tr));
  if (meta.axis) {
    // a 2-D diagram with composition on an axis
    const ax = meta.axis;
    const A = layout[ax + 'axis'];
    const r = A.range;
    const step = (r[1] - r[0]) / 240;
    const logOther = (layout[(ax === 'x' ? 'y' : 'x') + 'axis'] || {}).type === 'log';
    one = (tr) => Object.assign(dense2d(tr, ax, f, step, logOther), texts(tr));
    A.range = [f(r[0]), f(r[1])];
    A.title.text = txt(A.title.text);
  } else if (meta.scene) {
    // a 3-D view with composition on an axis
    const ax = meta.scene;
    const A = layout.scene[ax + 'axis'];
    const r = A.range || [0, 100];
    const step = (r[1] - r[0]) / 24;
    one = (tr) =>
      Object.assign(
        tr.type === 'mesh3d'
          ? denseMesh(tr, ax, f, step)
          : tr.type === 'scatter3d'
            ? dense3d(tr, ax, f, step / 4)
            : Object.assign({}, tr),
        texts(tr)
      );
    if (A.range) A.range = [f(r[0]), f(r[1])];
    A.title.text = txt(A.title.text);
  } else if (meta.color) {
    // composition is the optional color of the sheets
    one = (tr) => {
      const o = Object.assign({}, tr, texts(tr));
      const v = decode(tr.intensity);
      if (v) {
        o.intensity = Array.from(v, f);
        if (tr.cmin != null) o.cmin = f(tr.cmin);
        if (tr.cmax != null) o.cmax = f(tr.cmax);
      }
      return o;
    };
    for (const m of layout.updatemenus || [])
      for (const b of m.buttons || []) {
        if (b.label === 'Color: x') b.label = 'Color: ' + U.sym;
        const a0 = b.args && b.args[0];
        if (a0 && a0.colorbar && a0.colorbar.title)
          a0.colorbar.title.text = txt(a0.colorbar.title.text);
      }
  } else if (meta.hover) {
    // composition only in the hover text
    one = (tr) => {
      const o = Object.assign({}, tr, texts(tr));
      const cd = decode(tr.customdata);
      if (cd && !(tr.customdata && tr.customdata.shape) && !Array.isArray(cd[0]))
        o.customdata = Array.from(cd, f);
      return o;
    };
  }
  if (meta.slider)
    for (const sl of layout.sliders || []) {
      if (sl.currentvalue) sl.currentvalue.prefix = U.sym + ' = ';
      for (const st of sl.steps || []) st.label = txt(st.label);
    }
  return {
    data: fig.data.map(one),
    layout,
    frames: (fig.frames || []).map((fr) => Object.assign({}, fr, { data: fr.data.map(one) })),
  };
}

// ---- views and geometry ---------------------------------------------------
export const shortTitle = (t) => t.replace(' · ', ' ').replace(/ surface\s+/, ' ');

// The file of a view: its logarithmic twin when the log axes are on and it has one.
export const fileOf = (view, logAxes) => view.id + (logAxes && view.log ? '-log' : '');

// The page owns the pixel geometry: the figure takes whatever room the stage
// has (so nothing scrolls), with fixed margins that hold the legend above the
// plotting area and the slider below it. stageW and stageH are the stage's
// client size.
export function geometry(view, stageW, stageH) {
  const W = stageW - 12;
  const H = stageH - 8;
  if (view.startsWith('d-')) {
    // 2-D diagrams: a square plotting area, key in the page
    const m = { l: 70, r: 24, t: 14, b: 126 };
    const side = Math.max(120, Math.min(W - m.l - m.r, H - m.t - m.b));
    return {
      width: side + m.l + m.r,
      height: side + m.t + m.b,
      margin: m,
      slider: 52,
      legend: false,
    };
  }
  const side = Math.max(200, Math.min(W, H)); // 3-D: a square figure
  const rows = (px) => Math.ceil(px / side); // rows a horizontal legend of that total width needs
  if (view.startsWith('s3-'))
    return {
      width: side,
      height: side,
      margin: { l: 0, r: 0, t: 8, b: 84 },
      slider: 8,
      inset: 32,
      legend: false,
    };
  if (view === 'hull')
    return {
      width: side,
      height: side,
      margin: { l: 0, r: 0, t: 10 + 29 * rows(1060), b: 84 },
      slider: 8,
      inset: 32,
    };
  if (view === 'field')
    return { width: side, height: side, margin: { l: 0, r: 0, t: 10 + 29 * rows(520), b: 0 } };
  return { width: side, height: side, margin: { l: 0, r: 0, t: 44, b: 14 + 29 * rows(1420) } }; // projections: legend below
}

// The layout properties that depend on the window.
export function fitted(view, stageW, stageH) {
  const g = geometry(view, stageW, stageH);
  const out = {
    width: g.width,
    height: g.height,
    autosize: false,
    margin: Object.assign({ autoexpand: false, pad: 0 }, g.margin),
  };
  if (g.legend === false) out.showlegend = false;
  return [g, out];
}

// The figure's layout fitted to the stage, with its slider hung a fixed
// number of pixels under the plotting area.
export function stageLayout(fig, view, stageW, stageH) {
  const [g, fit] = fitted(view, stageW, stageH);
  const layout = Object.assign({}, fig.layout, fit);
  if (layout.sliders && g.slider != null)
    layout.sliders = layout.sliders.map((sl) =>
      Object.assign({}, sl, {
        y: 0,
        yanchor: 'top',
        pad: { t: g.slider, b: 0, l: g.inset || 0, r: g.inset || 0 },
      })
    );
  return layout;
}

// A trace that is not in the legend must not carry a legend group: one group
// anywhere makes a horizontal legend lay its entries out in a column per
// group, stacking the ungrouped color keys.
export const ungroup = (traces) =>
  traces.map((t) => {
    if (t.showlegend !== false || t.legendgroup == null) return t;
    const c = Object.assign({}, t);
    delete c.legendgroup;
    return c;
  });

// A legend wraps to as many rows as its width demands; once drawn, give it
// exactly the margin it needs (the figure keeps its size, the plot inside
// gives way), so it is never cut by the edge of the figure.
export async function settle(Plotly, p) {
  const L = p._fullLayout;
  if (!L || L.showlegend === false || !L.legend || !L.legend._height) return;
  const need = Math.ceil(L.legend._height) + 8;
  const top = L.legend.yanchor === 'bottom';
  if (top && need > L.margin.t) await Plotly.relayout(p, { 'margin.t': need });
  if (!top && L.legend.yanchor === 'top' && L.legend.y <= 0 && need > L.margin.b)
    await Plotly.relayout(p, { 'margin.b': need });
}

// What the notes add to a view's caption in the unit chosen.
export function captionNote(view, unit, S) {
  return (
    (view === 'hull' && unit !== 'wt'
      ? ' (Drawn in wt % whatever unit is chosen: the hull is convex per unit mass, in mass fractions.)'
      : '') +
    (uFor(S, unit) !== unit
      ? ' (The molecule database has no density for this solute, so it is shown in wt %, not vol %.)'
      : '')
  );
}

// The views whose diagrams carry tie-lines that can be hidden.
export const hasTies = (view) => view.startsWith('d-T') || view.startsWith('d-p');

// The step a view opens on. The atlas opens its isothermal sections on their
// coldest step; with NaCl every step below about -22 °C (the eutectic) lies
// outside the solution model, so a temperature slider that runs from cold to
// warm opens at OPEN_T_C instead. Other sliders keep the figure's own step.
export const OPEN_T_C = -10;
export function openingStep(layout) {
  const sl = layout && layout.sliders && layout.sliders[0];
  const first = (sl && sl.active) || 0;
  if (!sl || !sl.currentvalue || sl.currentvalue.prefix !== 'T = ') return first;
  const t = (sl.steps || []).map((s) => parseFloat(s.label));
  if (t.length < 2 || t.some((v) => !isFinite(v)) || !(t[0] < t[t.length - 1])) return first;
  let best = 0;
  t.forEach((v, i) => {
    if (Math.abs(v - OPEN_T_C) < Math.abs(t[best] - OPEN_T_C)) best = i;
  });
  return best;
}
