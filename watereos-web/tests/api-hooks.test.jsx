import { describe, it, expect, vi, beforeEach } from 'vitest';
import { renderHook, waitFor } from '@testing-library/react';
import { SWRConfig } from 'swr';

import * as client from '../src/api/client.js';
import { useMetadata, useCurvesFigure } from '../src/api/hooks.js';
import { ThemeProvider } from '../src/theme/ThemeContext.jsx';
import { SettingsProvider } from '../src/settings/SettingsContext.jsx';

function wrapper({ children }) {
  return (
    <SWRConfig value={{ provider: () => new Map(), dedupingInterval: 0 }}>
      <ThemeProvider>
        <SettingsProvider>{children}</SettingsProvider>
      </ThemeProvider>
    </SWRConfig>
  );
}

beforeEach(() => {
  vi.restoreAllMocks();
});

describe('api/hooks', () => {
  it('useMetadata fetches /api/metadata once', async () => {
    const spy = vi.spyOn(client, 'fetchMetadata').mockResolvedValue({ models: [] });
    const { result } = renderHook(() => useMetadata(), { wrapper });
    await waitFor(() => expect(result.current.data).toEqual({ models: [] }));
    expect(spy).toHaveBeenCalledTimes(1);
  });

  it('useCurvesFigure threads theme + units into the request body', async () => {
    const spy = vi.spyOn(client, 'fetchCurves').mockResolvedValue({ figure: { data: [] } });
    const params = {
      model: 'duska2020',
      property: 'rho',
      T_range: [200, 300],
      P_range: [0.1, 200],
      n_curves: 5,
      n_points: 200,
      isobar_mode: true,
      show_phase_boundaries: false,
    };
    const { result } = renderHook(() => useCurvesFigure(params), { wrapper });
    await waitFor(() => expect(result.current.data).toBeTruthy());
    const callBody = spy.mock.calls[0][0];
    expect(callBody).toMatchObject(params);
    expect(callBody.theme).toBe('dark'); // default from ThemeProvider
    expect(callBody.units).toMatchObject({ unit_density: 'kg/m³' });
  });

  it('useCurvesFigure with null params skips the fetch', async () => {
    const spy = vi.spyOn(client, 'fetchCurves').mockResolvedValue({ figure: { data: [] } });
    renderHook(() => useCurvesFigure(null), { wrapper });
    // Give SWR a tick
    await new Promise((r) => setTimeout(r, 10));
    expect(spy).not.toHaveBeenCalled();
  });
});
