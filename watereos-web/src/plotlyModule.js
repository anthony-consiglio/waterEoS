// Loading plotly.js-dist-min lazily, in dev and in the production build.
//
// Vite's dev-mode ESM-from-CJS interop can wrap a CJS module's `default`
// export inside another `default` (the module already shapes itself as
// `{ default: fn }`, then Vite places the whole exports object on
// `namespace.default`). The production Rollup build resolves this in one
// step. To work in both, we walk the `.default` chain looking for the
// actual value that matches a shape predicate.
export function unwrap(mod, isMatch, maxDepth = 4) {
  let v = mod;
  for (let i = 0; i < maxDepth && v != null; i++) {
    if (isMatch(v)) return v;
    if (typeof v === 'object' && 'default' in v && v.default !== v) {
      v = v.default;
    } else {
      break;
    }
  }
  return null;
}

let plotlyPromise = null;

// The Plotly instance (newPlot, react, relayout, ...), loaded once.
export function loadPlotly() {
  if (!plotlyPromise) {
    plotlyPromise = import('plotly.js-dist-min').then((mod) => {
      const Plotly = unwrap(mod, (x) => x && typeof x.newPlot === 'function');
      if (!Plotly) throw new Error('could not resolve a Plotly instance from plotly.js-dist-min');
      return Plotly;
    });
    plotlyPromise.catch(() => {
      plotlyPromise = null;
    });
  }
  return plotlyPromise;
}
