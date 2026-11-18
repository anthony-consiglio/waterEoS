import { describe, it, expect, vi } from 'vitest';
import { render, screen, waitFor, fireEvent } from '@testing-library/react';
import { SWRConfig } from 'swr';
import MeasuredData from '../src/screens/MeasuredData.jsx';
import { ThemeProvider } from '../src/theme/ThemeContext.jsx';
import { SettingsProvider } from '../src/settings/SettingsContext.jsx';
import * as client from '../src/api/client.js';
import { DEFAULT_FILTERS, applyFilters, countFits } from '../src/measured/filters.js';
import {
  propertyFigure2d,
  coverageFigure,
  scaleColor,
  inNoMansLand,
} from '../src/measured/figures.js';

// Made-up rows in the shape of /api/data/points (no real measurements).
const COLUMNS = {
  id: ['d:1', 'd:2', 'd:3', 'd:4'],
  table: ['density', 'density', 'density', 'density'],
  group: ['liquid', 'liquid', 'liquid', 'liquid'],
  property: ['density', 'density', 'density', 'density'],
  T_K: [280, 260, 250, 280],
  P_MPa: [0.1, 100, 200, 0.1],
  value: [999, 1050, 1100, 1105],
  unit: ['kg/m3', 'kg/m3', 'kg/m3', 'kg/m3'],
  u_std: [0.5, 0.5, null, 0.5],
  u_T_K: [null, null, null, null],
  u_P_MPa: [null, null, null, null],
  substance: ['H2O', 'H2O', 'H2O', 'D2O'],
  state: ['stable', 'supercooled', 'supercooled', 'stable'],
  state_final: ['', '', '', ''],
  ice_phase: ['', '', '', ''],
  sample: ['bulk', 'bulk', 'emulsion', 'bulk'],
  method: ['', '', '', ''],
  data_type: ['measured', 'measured', 'quoted', 'measured'],
  extraction: [
    'machine-readable file',
    'transcribed table',
    'transcribed table',
    'machine-readable file',
  ],
  verification: ['supplied file', 'page image', 'page image', 'supplied file'],
  qc_flag: ['ok', 'ok', 'ok', 'no-reference'],
  source_key: ['alpha2001', 'alpha2001', 'beta2010', 'beta2010'],
  year: [2001, 2001, 2010, 2010],
  used_in_fits: ['Holten2014', '', '', ''],
  used_in_fits_level: ['Holten2014:point', '', '', ''],
  definition: ['', '', '', ''],
  rate_K_min: [null, null, null, null],
  location: ['Table 1', 'Table 2', 'p. 3', 'Table 1'],
};
const PROP = {
  key: 'density',
  label: 'Density ρ',
  unit: 'kg/m3',
  n: 4,
  locus: null,
  log: false,
  models: [],
};
const RELEASE = {
  configured: true,
  release: { release_id: '2026-01-01-abcdef12', n_rows: 4, n_sources: 2, n_tables: 1 },
  groups: [{ key: 'liquid', label: 'Liquid', thermodynamic: true }],
  tables: [
    {
      key: 'density',
      label: 'liquid/density',
      group: 'liquid',
      n: 4,
      properties: [{ ...PROP, models: [{ key: 'IAPWS95', label: 'IAPWS-95' }] }],
    },
  ],
  years: [2001, 2010],
  notice: 'These values were retrieved or transcribed from published studies.',
  cite: 'waterEoS (measured-data compilation, release 2026-01-01-abcdef12).',
};
const LINES = {
  melting: {
    source: 'IAPWS R14-08(2011)',
    branches: [{ label: 'ice Ih', T_K: [251, 273], P_MPa: [209, 0] }],
  },
  th: { source: 'Holten 2014', T_K: [236, 225, 181], P_MPa: [0, 100, 1000] },
  vapor: { source: 'Murphy & Koop (2005)', branches: [] },
};

function wrap(ui) {
  return (
    <SWRConfig value={{ provider: () => new Map(), dedupingInterval: 0 }}>
      <ThemeProvider>
        <SettingsProvider>{ui}</SettingsProvider>
      </ThemeProvider>
    </SWRConfig>
  );
}

function mockApi(release = RELEASE) {
  vi.spyOn(client, 'fetchDataRelease').mockResolvedValue(release);
  vi.spyOn(client, 'fetchDataPoints').mockResolvedValue({
    release_id: 'x',
    n: 4,
    columns: COLUMNS,
  });
  vi.spyOn(client, 'fetchDataSources').mockResolvedValue({
    sources: [
      {
        source_key: 'alpha2001',
        cite_short: 'Alpha (2001)',
        citation: 'A. Alpha (2001). Made up.',
        doi: '10.0/a',
        n_rows: 2,
        tables: 'liquid/density (2)',
        extraction: 'machine-readable file',
      },
      {
        source_key: 'beta2010',
        cite_short: 'Beta (2010)',
        citation: 'B. Beta (2010). Made up.',
        doi: '',
        n_rows: 2,
        tables: 'liquid/density (2)',
        extraction: 'transcribed table',
      },
    ],
  });
  vi.spyOn(client, 'fetchDataLines').mockResolvedValue(LINES);
  vi.spyOn(client, 'fetchModelCurves').mockResolvedValue({ curves: [] });
}

describe('MeasuredData screen', () => {
  it('shows the release, the table and the default filtered count', async () => {
    mockApi();
    render(wrap(<MeasuredData />));
    await waitFor(() => expect(screen.getByText('Measured water data')).toBeInTheDocument());
    expect(screen.getByText(/release 2026-01-01-abcdef12/)).toBeInTheDocument();
    expect(screen.getByRole('option', { name: 'liquid/density (4)' })).toBeInTheDocument();
    // H2O only and no quoted rows by default: rows 1 and 2.
    await waitFor(() => expect(screen.getByText('2 of 4 rows plotted')).toBeInTheDocument());
    expect(screen.getByText('IAPWS-95')).toBeInTheDocument();
    expect(screen.getByRole('link', { name: 'Download all (zip)' })).toHaveAttribute(
      'href',
      '/api/data/download'
    );
  });

  it('lists the filtered rows and the sources', async () => {
    mockApi();
    render(wrap(<MeasuredData />));
    await waitFor(() => expect(screen.getByRole('button', { name: 'Rows' })).toBeInTheDocument());
    fireEvent.click(screen.getByRole('button', { name: 'Rows' }));
    await waitFor(() => expect(screen.getByText(/2 of 4 rows after filters/)).toBeInTheDocument());
    expect(screen.getAllByText('Alpha (2001)').length).toBeGreaterThan(0);
    fireEvent.click(screen.getByRole('button', { name: 'Sources' }));
    await waitFor(() => expect(screen.getByText('2 of 2 published studies')).toBeInTheDocument());
    expect(screen.getByRole('link', { name: 'doi:10.0/a' })).toHaveAttribute(
      'href',
      'https://doi.org/10.0/a'
    );
  });

  it('explains when the server has no measured data', async () => {
    vi.spyOn(client, 'fetchDataRelease').mockResolvedValue({
      configured: false,
      detail: 'Measured data are not configured on this server.',
    });
    render(wrap(<MeasuredData />));
    await waitFor(() =>
      expect(
        screen.getByText('Measured data are not available on this server.')
      ).toBeInTheDocument()
    );
  });
});

describe('measured-data helpers', () => {
  const all = [0, 1, 2, 3];

  it('filters by excluded categories, sources, fits and year', () => {
    expect(applyFilters(COLUMNS, all, DEFAULT_FILTERS)).toEqual([0, 1]);
    const everything = {
      ...DEFAULT_FILTERS,
      exclude: { ...DEFAULT_FILTERS.exclude, substance: [], data_type: [] },
    };
    expect(applyFilters(COLUMNS, all, everything)).toEqual([0, 1, 2, 3]);
    expect(applyFilters(COLUMNS, all, { ...everything, sources: ['beta2010'] })).toEqual([2, 3]);
    expect(applyFilters(COLUMNS, all, { ...everything, fits: ['Holten2014'] })).toEqual([0]);
    expect(applyFilters(COLUMNS, all, { ...everything, fits: ['none'] })).toEqual([1, 2, 3]);
    expect(applyFilters(COLUMNS, all, { ...everything, yearMin: 2005 })).toEqual([2, 3]);
    // skipping a filter counts that filter's options within the others
    expect(applyFilters(COLUMNS, all, DEFAULT_FILTERS, 'substance')).toEqual([0, 1, 3]);
    expect(countFits(COLUMNS, all, true).get('Holten2014')).toBe(1);
  });

  it('builds a 2D property figure with D2O as open markers', () => {
    const fig = propertyFigure2d({
      c: COLUMNS,
      idx: all,
      prop: PROP,
      xVar: 'T',
      colorMode: 'substance',
      errorBars: true,
      logX: false,
      logY: false,
      theme: 'dark',
      names: { cites: {} },
      modelLabels: {},
      lines: LINES,
    });
    expect(fig.n).toBe(4);
    const d2o = fig.data.find((t) => t.name.startsWith('D2O'));
    expect(d2o.marker.symbol).toEqual(['circle-open']);
    expect(fig.layout.xaxis.title.text).toBe('Temperature T (K)');
    expect(fig.layout.yaxis.title.text).toBe('Density ρ (kg/m³)');
  });

  it('gives every 2D marker trace both error bars while error bars are on', () => {
    // Plotly's WebGL error bars break when a trace's error_x or error_y comes
    // and goes between updates (stale points drawn on the new axes), so a
    // trace without any T uncertainty still carries an all-zero error_x.
    const fig = (errorBars, colorMode) =>
      propertyFigure2d({
        c: COLUMNS,
        idx: all,
        prop: PROP,
        xVar: 'T',
        colorMode,
        errorBars,
        logX: false,
        logY: false,
        theme: 'dark',
        names: { cites: {} },
        modelLabels: {},
      });
    for (const mode of ['other', 'substance', 'source_key']) {
      const markers = fig(true, mode).data.filter((t) => t.type === 'scattergl');
      expect(markers.length).toBeGreaterThan(0);
      for (const t of markers) {
        expect(t.error_x.visible).toBe(true);
        expect(t.error_x.array).toEqual(t.x.map(() => 0));
        expect(t.error_y.visible).toBe(true);
        expect(t.error_y.array).toHaveLength(t.x.length);
      }
      for (const t of fig(false, mode).data.filter((t) => t.type === 'scattergl')) {
        expect(t.error_x).toBeUndefined();
        expect(t.error_y).toBeUndefined();
      }
    }
  });

  it('colors model curves on the data scale and fades them outside the model range', () => {
    const curves = [
      { model: 'm', fixed: 100, x: [250, 260, 270], y: [1, 2, 3], in_range: [false, true, true] },
    ];
    const fig = propertyFigure2d({
      c: COLUMNS,
      idx: all,
      prop: PROP,
      xVar: 'T',
      colorMode: 'other',
      errorBars: false,
      logX: false,
      logY: false,
      theme: 'light',
      names: { cites: {} },
      modelLabels: { m: 'Model M' },
      curves,
    });
    const lines = fig.data.filter((t) => t.legendgroup === 'model:m');
    expect(lines[0].name).toBe('Model M');
    const [inside, outside] = lines.slice(1);
    expect(inside.y).toEqual([null, 2, 3]);
    expect(outside.y).toEqual([1, 2, null]);
    expect(outside.opacity).toBeLessThan(1);
    expect(inside.line.color).toBe(scaleColor((100 - 0.1) / (200 - 0.1)));
  });

  it('bins coverage counts without losing rows', () => {
    const fig = coverageFigure({
      c: COLUMNS,
      idx: all,
      colorMode: 'table',
      mode: 'counts',
      theme: 'dark',
      names: {},
      lines: LINES,
      showNml: true,
      logY: false,
      binT: 7,
      binP: 13,
      nonThermo: new Set(),
    });
    const heat = fig.data.find((t) => t.type === 'heatmap');
    expect(heat.customdata.flat().reduce((a, b) => a + b, 0)).toBe(4);
    expect(inNoMansLand(LINES, 200, 50)).toBe(true);
    expect(inNoMansLand(LINES, 260, 50)).toBe(false);
  });
});

describe('method filter', () => {
  it('hides rows by measurement method', () => {
    const c = { ...COLUMNS, method: ['DDM', 'DDM', 'Brillouin scattering', 'viscometer'] };
    const all = [0, 1, 2, 3];
    const base = {
      ...DEFAULT_FILTERS,
      exclude: { ...DEFAULT_FILTERS.exclude, substance: [], data_type: [] },
    };
    expect(
      applyFilters(c, all, { ...base, exclude: { ...base.exclude, method: ['DDM'] } })
    ).toEqual([2, 3]);
  });
});
