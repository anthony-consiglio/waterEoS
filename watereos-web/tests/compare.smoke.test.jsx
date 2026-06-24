import { describe, it, expect, vi } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import { SWRConfig } from 'swr';
import ModelComparison from '../src/screens/ModelComparison.jsx';
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

describe('ModelComparison screen', () => {
  it('renders model checkboxes, property select, layout switch, and plot card', async () => {
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
        {
          key: 'holten2014',
          display_name: 'Holten (2014)',
          is_two_state: true,
          has_phase_diagram: true,
          has_transport: false,
          T_min: 180,
          T_max: 320,
          P_min: 0.1,
          P_max: 400,
          properties: ['rho'],
        },
      ],
      properties: {
        rho: { label: 'Density', unit: 'kg/m³' },
        Cp: { label: 'Isobaric heat capacity', unit: 'J/(kg·K)' },
      },
      units: { options: {}, defaults: {}, category_labels: {} },
    });
    vi.spyOn(client, 'fetchCompare').mockResolvedValue({
      figure: { data: [{ type: 'scatter', x: [], y: [] }], layout: {} },
      warnings: [],
    });
    render(wrap(<ModelComparison />));
    await waitFor(() => expect(screen.getByText('Duska (2020)')).toBeInTheDocument());
    expect(screen.getByText('Holten (2014)')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Overlay' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Side by side' })).toBeInTheDocument();
    expect(document.querySelector('.plot-card')).toBeInTheDocument();
  });
});
