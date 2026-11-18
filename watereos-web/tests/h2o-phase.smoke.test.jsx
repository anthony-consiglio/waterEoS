import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor, fireEvent } from '@testing-library/react';

// A Plotly that records what it is asked to draw.
const plotly = vi.hoisted(() => {
  const P = {
    newPlot: vi.fn(async (el, data, layout) => {
      el.data = data;
      el.layout = layout;
      el._fullLayout = { ...layout, showlegend: false };
      el.on = () => {};
    }),
    purge: vi.fn(),
    addFrames: vi.fn(async () => {}),
    animate: vi.fn(async () => {}),
    relayout: vi.fn(async () => {}),
    restyle: vi.fn(),
  };
  return P;
});
vi.mock('../src/plotlyModule.js', () => ({ loadPlotly: async () => plotly, unwrap: (x) => x }));
vi.mock('../src/api/client.js', async (orig) => ({
  ...(await orig()),
  fetchAtlasManifest: vi.fn(),
  fetchAtlasFile: vi.fn(),
}));

import * as client from '../src/api/client.js';
import { clearAtlasFiles } from '../src/h2oAtlas/data.js';
import H2OPhaseDiagram from '../src/screens/H2OPhaseDiagram.jsx';

const MANIFEST = {
  configured: true,
  release_id: '2026-01-01-abcdef12',
  created_at: '2026-01-01T00:00:00+00:00',
  quality: 'standard',
  solutes: [
    { id: 'nacl', name: 'Sodium chloride', electrolyte: true, W_fit: 0.22, M: 58.443, rho: 2.165 },
  ],
  groups: [
    { id: 'd', title: 'Phase diagrams (2-D sections)', short: 'Phase diagrams' },
    { id: 's3', title: 'Sections as surfaces', short: 'Surfaces' },
  ],
  views: [
    {
      id: 'd-T-xv',
      group: 'd',
      title: 'Isothermal · x – v',
      caption: 'The density plane.',
      log: 'log v',
    },
    {
      id: 'd-x-Tp',
      group: 'd',
      title: 'Isoplethal · T – p',
      caption: 'At x = 0 the p–T diagram of water.',
      log: 'log p',
    },
    {
      id: 's3-T',
      group: 's3',
      title: 'Isothermal surface  p(x, v)',
      caption: 'A pressure surface.',
      log: null,
    },
  ],
  key: {
    fill: ['#e3edfa', '#a9caf3', '#5b97e0'],
    surf: ['#d5e4f8', '#86b4ee', '#3472c8'],
    outside: '#efeeea',
    tie: '#8b8f94',
  },
};
const INFO = {
  id: 'nacl',
  name: 'Sodium chloride',
  molar_mass: 58.443,
  liquid: 'holten2014',
  model: 'ove_fitted_BC',
  ices: ['Ih', 'III'],
  vapor: true,
  T_min: 229.2,
  p_min: 1e-6,
  p_max: 400,
  W_cap: 0.22,
  volume: { V0: 16.6, source: 'ionic volumes' },
  coefficients: { k: 1.8, B: 0.0027, C: 0.0023, m_fit: 4.826 },
  properties: { density: 2.165, cas: '7647-14-5' },
  points: [{ phases: ['L', 'Ih', 'vap'], T: 273.16, p: 0.000612, W: 0 }],
};
const figure = (name) => ({
  data: [{ type: 'scatter', name, x: [0, 1], y: [0, 1] }],
  layout: { xaxis: { range: [0, 1], title: { text: 'T (°C)' } }, yaxis: {} },
});

beforeEach(() => {
  clearAtlasFiles();
  vi.clearAllMocks();
  client.fetchAtlasManifest.mockResolvedValue(MANIFEST);
  client.fetchAtlasFile.mockImplementation(async (solute, name) =>
    name === 'info' ? INFO : figure(name)
  );
});

const lastDrawn = () => plotly.newPlot.mock.calls.at(-1)?.[1]?.[0]?.name;

describe('H2OPhaseDiagram screen (the water + NaCl atlas)', () => {
  it('shows the atlas groups without a process view and opens on the T–p diagram', async () => {
    render(<H2OPhaseDiagram />);
    expect(await screen.findByRole('tab', { name: 'Phase diagrams' })).toBeInTheDocument();
    expect(screen.getByRole('tab', { name: 'Surfaces' })).toBeInTheDocument();
    expect(screen.queryByRole('tab', { name: /process/i })).toBeNull();
    expect(screen.getByText('water + Sodium chloride')).toBeInTheDocument();
    expect(screen.getByRole('tab', { name: 'Isoplethal T – p' })).toHaveAttribute(
      'aria-selected',
      'true'
    );
    await waitFor(() => expect(lastDrawn()).toBe('d-x-Tp'));
    expect(client.fetchAtlasFile).toHaveBeenCalledWith('nacl', 'info');
    const units = screen.getByLabelText('Composition unit');
    expect([...units.options].map((o) => o.textContent)).toEqual([
      'wt %',
      'mol %',
      'mol/kg',
      'vol %',
    ]);
    expect(screen.getByText('At x = 0 the p–T diagram of water.')).toBeInTheDocument();
  });

  it('switches views, shows the key and draws the logarithmic twin', async () => {
    render(<H2OPhaseDiagram />);
    fireEvent.click(await screen.findByRole('tab', { name: 'Isothermal x – v' }));
    await waitFor(() => expect(lastDrawn()).toBe('d-T-xv'));
    expect(screen.getByRole('button', { name: /tie-lines/ })).toHaveAttribute(
      'aria-pressed',
      'true'
    );
    expect(screen.getByText('one phase')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'log v' }));
    await waitFor(() => expect(lastDrawn()).toBe('d-T-xv-log'));
    fireEvent.click(screen.getByRole('tab', { name: 'Surfaces' }));
    await waitFor(() => expect(lastDrawn()).toBe('s3-T'));
    expect(screen.queryByRole('button', { name: /tie-lines/ })).toBeNull();
  });

  it('shows the model and the invariant points in the notes', async () => {
    render(<H2OPhaseDiagram />);
    fireEvent.click(await screen.findByRole('button', { name: 'Model' }));
    expect(await screen.findByText(/release 2026-01-01-abcdef12/)).toBeInTheDocument();
    expect(screen.getByText('Ih, III (SeaFreeze)')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Invariant points' }));
    expect(await screen.findByText('L + Ih + vapor')).toBeInTheDocument();
    expect(screen.getByText('612 Pa')).toBeInTheDocument();
  });

  it('says so when the phase diagrams are not loaded', async () => {
    client.fetchAtlasManifest.mockResolvedValue({
      configured: false,
      detail: 'The phase diagrams are not configured on this server.',
    });
    render(<H2OPhaseDiagram />);
    expect(
      await screen.findByText('The phase diagrams are not configured on this server.')
    ).toBeInTheDocument();
    expect(plotly.newPlot).not.toHaveBeenCalled();
  });
});
