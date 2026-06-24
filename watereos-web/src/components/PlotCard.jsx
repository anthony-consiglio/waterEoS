import { Suspense, lazy } from 'react';

// Use react-plotly.js's factory + plotly.js-dist-min so we don't pull in the
// full plotly.js (~3 MB unminified). Both modules are loaded lazily.
//
// Vite's dev-mode ESM-from-CJS interop can wrap a CJS module's `default`
// export inside another `default` (the module already shapes itself as
// `{ default: fn }`, then Vite places the whole exports object on
// `namespace.default`). The production Rollup build resolves this in one
// step. To work in both, we walk the `.default` chain looking for the
// actual value that matches a shape predicate.
function unwrap(mod, isMatch, maxDepth = 4) {
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

const Plot = lazy(async () => {
  const [plotlyMod, factoryMod] = await Promise.all([
    import('plotly.js-dist-min'),
    import('react-plotly.js/factory'),
  ]);
  const Plotly = unwrap(plotlyMod, (x) => x && typeof x.newPlot === 'function');
  const createPlotlyComponent = unwrap(factoryMod, (x) => typeof x === 'function');
  if (!Plotly) {
    throw new Error('PlotCard: could not resolve a Plotly instance from plotly.js-dist-min');
  }
  if (!createPlotlyComponent) {
    throw new Error('PlotCard: could not resolve factory from react-plotly.js/factory');
  }
  return { default: createPlotlyComponent(Plotly) };
});

export function PlotCard({ title, subtitle, figure, loading, error, toolbar, bare = false, className }) {
  return (
    <div className={'plot-card' + (className ? ' ' + className : '')}>
      {!bare && (
        <header className="plot-head">
          <div>
            <h2 className="plot-title">{title}</h2>
            {subtitle && <p className="plot-subtitle">{subtitle}</p>}
          </div>
          {toolbar && <div className="plot-toolbar">{toolbar}</div>}
        </header>
      )}
      <div className="plot-body">
        {error && (
          <div className="plot-error" role="alert">
            {String(error?.message ?? error)}
          </div>
        )}
        {loading && !figure && <div className="plot-loading">Loading…</div>}
        {figure && (
          <Suspense fallback={<div className="plot-loading">Rendering chart…</div>}>
            <Plot
              data={figure.data ?? []}
              layout={figure.layout ?? {}}
              config={{ displaylogo: false, responsive: true }}
              useResizeHandler
              style={{ width: '100%', height: '100%' }}
            />
          </Suspense>
        )}
      </div>
    </div>
  );
}
