import { describe, it, expect, vi } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import { SWRConfig } from 'swr';
import EoSPhaseDiagram from '../src/screens/EoSPhaseDiagram.jsx';
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

describe('EoSPhaseDiagram screen', () => {
  it('renders only phase-diagram models in selector and a plot card', async () => {
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
          properties: ['rho'],
        },
        {
          key: 'iapws95',
          display_name: 'IAPWS-95',
          is_two_state: false,
          has_phase_diagram: false,
          has_transport: false,
          T_min: 250,
          T_max: 1000,
          P_min: 0.1,
          P_max: 1000,
          properties: ['rho'],
        },
      ],
      properties: {},
      units: { options: {}, defaults: {}, category_labels: {} },
    });
    vi.spyOn(client, 'fetchEosPhase').mockResolvedValue({
      figure: { data: [{ type: 'scatter', x: [], y: [] }], layout: {} },
      warnings: [],
    });
    render(wrap(<EoSPhaseDiagram />));
    // Phase-diagram model selectable
    await waitFor(() =>
      expect(screen.getByRole('option', { name: 'Duska (2020)' })).toBeInTheDocument()
    );
    // Non-phase-diagram model NOT in select
    expect(screen.queryByRole('option', { name: 'IAPWS-95' })).not.toBeInTheDocument();
    // At least one overlay checkbox label
    expect(screen.getByText('Binodal')).toBeInTheDocument();
    expect(document.querySelector('.plot-card')).toBeInTheDocument();
  });
});
