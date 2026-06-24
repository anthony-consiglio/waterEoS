import { describe, it, expect, vi } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import { SWRConfig } from 'swr';
import Info from '../src/screens/Info.jsx';
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

describe('Info screen', () => {
  it('renders the hero heading and metadata-driven models table', async () => {
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
      ],
      properties: {},
      units: { options: {}, defaults: {}, category_labels: {} },
    });
    render(wrap(<Info />));
    expect(
      screen.getByRole('heading', { name: /thermodynamic|supercooled water/i })
    ).toBeInTheDocument();
    await waitFor(() => expect(screen.getByText('Duska (2020)')).toBeInTheDocument());
  });
});
