import { describe, it, expect, vi, beforeEach } from 'vitest';
import * as client from '../src/api/client.js';

beforeEach(() => {
  global.fetch = vi.fn(async () => ({
    ok: true,
    status: 200,
    json: async () => ({ ok: true }),
  }));
});

describe('api/client', () => {
  it('fetchHealth GETs /api/health', async () => {
    await client.fetchHealth();
    expect(global.fetch).toHaveBeenCalledWith('/api/health');
  });

  it('fetchMetadata GETs /api/metadata', async () => {
    await client.fetchMetadata();
    expect(global.fetch).toHaveBeenCalledWith('/api/metadata');
  });

  it('fetchPoint POSTs JSON to /api/point with the given body', async () => {
    const body = { model_keys: ['duska2020'], T_K: 273.15, P_MPa: 0.1 };
    await client.fetchPoint(body);
    expect(global.fetch).toHaveBeenCalledWith(
      '/api/point',
      expect.objectContaining({
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      })
    );
  });

  it('fetchCurves POSTs to /api/figures/curves', async () => {
    await client.fetchCurves({ model: 'duska2020' });
    expect(global.fetch).toHaveBeenCalledWith('/api/figures/curves', expect.any(Object));
  });

  it('fetchSurface2 POSTs to /api/figures/surface2d', async () => {
    await client.fetchSurface2({ model: 'duska2020' });
    expect(global.fetch).toHaveBeenCalledWith('/api/figures/surface2d', expect.any(Object));
  });

  it('fetchSurface3 POSTs to /api/figures/surface3d', async () => {
    await client.fetchSurface3({ model: 'duska2020' });
    expect(global.fetch).toHaveBeenCalledWith('/api/figures/surface3d', expect.any(Object));
  });

  it('fetchCompare POSTs to /api/figures/compare', async () => {
    await client.fetchCompare({ model_keys: ['duska2020'] });
    expect(global.fetch).toHaveBeenCalledWith('/api/figures/compare', expect.any(Object));
  });

  it('fetchEosPhase POSTs to /api/figures/eos-phase-diagram', async () => {
    await client.fetchEosPhase({ model: 'duska2020' });
    expect(global.fetch).toHaveBeenCalledWith('/api/figures/eos-phase-diagram', expect.any(Object));
  });

  it('fetchH2OPhase POSTs to /api/figures/h2o-phase-diagram', async () => {
    await client.fetchH2OPhase({ projection: 'tv' });
    expect(global.fetch).toHaveBeenCalledWith('/api/figures/h2o-phase-diagram', expect.any(Object));
  });

  it('non-200 responses raise an Error with detail from body', async () => {
    global.fetch = vi.fn(async () => ({
      ok: false,
      status: 404,
      statusText: 'Not Found',
      json: async () => ({ detail: 'unknown model: nope' }),
    }));
    await expect(client.fetchCurves({ model: 'nope' })).rejects.toThrow('unknown model: nope');
  });

  it('honors VITE_API_BASE_URL for the base prefix', async () => {
    vi.stubEnv('VITE_API_BASE_URL', 'http://example.test');
    try {
      await client.fetchHealth();
      expect(global.fetch).toHaveBeenCalledWith('http://example.test/api/health');
    } finally {
      vi.unstubAllEnvs();
    }
  });
});
