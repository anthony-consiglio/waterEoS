import { describe, it, expect, vi } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import { SWRConfig } from 'swr';
import PointCalculator from '../src/screens/PointCalculator.jsx';
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

describe('PointCalculator screen', () => {
  it('renders density value and property cards after API resolves', async () => {
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
          properties: ['rho', 'Cp', 'x'],
        },
      ],
      properties: {
        rho: { label: 'Density', unit: 'kg/m³' },
        Cp: { label: 'Isobaric heat capacity', unit: 'J/(kg·K)' },
        x: { label: 'LDL fraction', unit: '' },
      },
      units: { options: {}, defaults: {}, category_labels: {} },
    });
    vi.spyOn(client, 'fetchPoint').mockResolvedValue({
      results: {
        duska2020: { rho: 999.819, Cp: 4218.5, x: 0.42 },
      },
      warnings: [],
    });
    render(wrap(<PointCalculator />));
    await waitFor(() => {
      // density hero value rounds appears as "999.8" or "999.82" — match any prefix
      expect(screen.getByText(/999\.8/)).toBeInTheDocument();
    });
    // Cp property card label present
    expect(screen.getByText(/Isobaric heat capacity/i)).toBeInTheDocument();
  });
});
