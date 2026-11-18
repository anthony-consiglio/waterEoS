import useSWR from 'swr';
import * as client from './client.js';
import { useTheme } from '../theme/ThemeContext.jsx';
import { useSettings } from '../settings/SettingsContext.jsx';

const SWR_OPTS = { keepPreviousData: true, revalidateOnFocus: false };

export function useMetadata() {
  return useSWR('/api/metadata', client.fetchMetadata, SWR_OPTS);
}

// `includeUnits` controls whether the user's unit selections are spread into
// the request body. The phase-diagram endpoints (EoS, H2O) use Pydantic
// schemas with `extra="forbid"` and no `units` field, so passing it yields a
// 422. Property/comparison endpoints accept it.
function _figureHook(path, clientKey, { includeUnits = true } = {}) {
  return function useFigure(params, enabled = true) {
    const { theme } = useTheme();
    const { units } = useSettings();
    const body = enabled && params ? { ...params, theme, ...(includeUnits ? { units } : {}) } : null;
    return useSWR(body && [path, body], ([, b]) => client[clientKey](b), SWR_OPTS);
  };
}

export const useCurvesFigure = _figureHook('/api/figures/curves', 'fetchCurves');
export const useSurface2dFigure = _figureHook('/api/figures/surface2d', 'fetchSurface2');
export const useSurface3dFigure = _figureHook('/api/figures/surface3d', 'fetchSurface3');
export const useCompareFigure = _figureHook('/api/figures/compare', 'fetchCompare');
export const useEosPhaseFigure = _figureHook('/api/figures/eos-phase-diagram', 'fetchEosPhase', {
  includeUnits: false,
});
export const useH2OPhaseFigure = _figureHook('/api/figures/h2o-phase-diagram', 'fetchH2OPhase', {
  includeUnits: false,
});

export function usePoint(params, enabled = true) {
  const { units } = useSettings();
  const body = enabled && params ? { ...params, units } : null;
  return useSWR(body && ['/api/point', body], ([, b]) => client.fetchPoint(b), SWR_OPTS);
}

// ---------------------------------------------------------------------------
// Measured data
// ---------------------------------------------------------------------------
const DATA_OPTS = { ...SWR_OPTS, revalidateIfStale: false };

export function useDataRelease() {
  return useSWR('/api/data/release', client.fetchDataRelease, DATA_OPTS);
}

// `table` null loads every row (the coverage map); `enabled` false loads nothing.
export function useDataPoints(table, property, enabled = true) {
  const key = enabled ? ['/api/data/points', table ?? '', property ?? ''] : null;
  return useSWR(key, ([, t, p]) => client.fetchDataPoints(t || null, p || null), DATA_OPTS);
}

export function useDataSources(enabled = true) {
  return useSWR(enabled ? '/api/data/sources' : null, client.fetchDataSources, DATA_OPTS);
}

export function useDataLines(enabled = true) {
  return useSWR(enabled ? '/api/data/lines' : null, client.fetchDataLines, DATA_OPTS);
}

export function useDataRow(id) {
  return useSWR(id ? ['/api/data/rows', id] : null, ([, i]) => client.fetchDataRow(i), DATA_OPTS);
}

export function useModelCurves(body) {
  return useSWR(
    body ? ['/api/data/model-curves', body] : null,
    ([, b]) => client.fetchModelCurves(b),
    DATA_OPTS
  );
}

// One request per model; returns { data: [surface, ...], isLoading, error }.
export function useModelSurfaces(property, models, T_range, P_range, n = 60) {
  const key =
    property && models.length && T_range && P_range
      ? [
          '/api/data/model-surface',
          property,
          models.join(','),
          T_range.join(','),
          P_range.join(','),
          n,
        ]
      : null;
  return useSWR(
    key,
    () =>
      Promise.all(
        models.map((model) => client.fetchModelSurface({ property, model, T_range, P_range, n }))
      ),
    DATA_OPTS
  );
}
