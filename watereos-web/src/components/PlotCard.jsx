import { Suspense, lazy } from 'react';

import { unwrap } from '../plotlyModule.js';

// Use react-plotly.js's factory + plotly.js-dist-min so we don't pull in the
// full plotly.js (~3 MB unminified). Both modules are loaded lazily; unwrap
// resolves Vite's dev-mode interop (see plotlyModule.js).
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

export function PlotCard({
  title,
  subtitle,
  figure,
  loading,
  error,
  toolbar,
  bare = false,
  className,
  onClick,
  footer,
}) {
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
              onClick={onClick}
            />
          </Suspense>
        )}
      </div>
      {footer}
    </div>
  );
}
