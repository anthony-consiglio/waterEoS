import { describe, it, expect, vi } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import { SWRConfig } from 'swr';
import PropertyExplorer from '../src/screens/PropertyExplorer.jsx';
import { ThemeProvider } from '../src/theme/ThemeContext.jsx';
import { SettingsProvider } from '../src/settings/SettingsContext.jsx';
import * as client from '../src/api/client.js';

function wrap(ui) {
  return (
    <SWRConfig value={{ provider: () => new Map(), dedupingInterval: 0 }}>
      <ThemeProvider>
        <SettingsProvider>{ui}</SettingsProvider>
      </ThemeProvider>
    </SWRConfig>
  );
}

describe('PropertyExplorer screen', () => {
  it('renders sidebar controls and a plot card after metadata arrives', async () => {
    vi.spyOn(client, 'fetchMetadata').mockResolvedValue({
      models: [
        {
          key: 'duska2020',
          display_name: 'Duska (2020)',
          is_two_state: true,
          has_phase_diagram: true,
          has_transport: false,
          T_min: 200,
          T_max: 370,
          P_min: 0.1,
          P_max: 200,
          properties: ['rho', 'Cp'],
        },
      ],
      properties: {
        rho: { label: 'Density', unit: 'kg/m³' },
        Cp: { label: 'Isobaric heat capacity', unit: 'J/(kg·K)' },
      },
      units: { options: {}, defaults: {}, category_labels: {} },
    });
    vi.spyOn(client, 'fetchCurves').mockResolvedValue({
      figure: { data: [{ type: 'scatter', x: [], y: [] }], layout: {} },
      warnings: [],
    });
    render(wrap(<PropertyExplorer />));
    await waitFor(() => expect(screen.getAllByText(/density/i).length).toBeGreaterThan(0));
    // PlotCard renders with the .plot-card root class
    expect(document.querySelector('.plot-card')).toBeInTheDocument();
  });
});
