import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { SWRConfig } from 'swr';
import Settings from '../src/screens/Settings.jsx';
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

describe('Settings screen', () => {
  it('changing the density unit select persists to localStorage', async () => {
    vi.spyOn(client, 'fetchMetadata').mockResolvedValue({
      models: [],
      properties: {},
      units: {
        options: {
          unit_density: [
            { label: 'kg/m³', value: 'kg/m³' },
            { label: 'g/cm³', value: 'g/cm³' },
          ],
          unit_volume: [
            { label: 'm³/kg', value: 'm³/kg' },
            { label: 'cm³/g', value: 'cm³/g' },
          ],
          unit_energy: [],
          unit_entropy: [],
          unit_bulk_modulus: [],
          unit_viscosity: [],
        },
        defaults: {
          unit_density: 'kg/m³',
          unit_volume: 'm³/kg',
          unit_energy: 'J/kg',
          unit_entropy: 'J/(kg·K)',
          unit_bulk_modulus: 'MPa',
          unit_viscosity: 'Pa·s',
        },
        category_labels: {
          unit_density: 'Density',
          unit_volume: 'Volume',
          unit_energy: 'Energy',
          unit_entropy: 'Entropy',
          unit_bulk_modulus: 'Bulk modulus',
          unit_viscosity: 'Viscosity',
        },
      },
    });
    render(wrap(<Settings />));
    // wait until the density select shows its options
    await waitFor(() => expect(screen.getByRole('option', { name: 'g/cm³' })).toBeInTheDocument());
    const densitySelect = screen.getByLabelText(/density/i);
    fireEvent.change(densitySelect, { target: { value: 'g/cm³' } });
    // localStorage should now hold the updated unit
    const persisted = JSON.parse(localStorage.getItem('watereos_settings'));
    expect(persisted.unit_density).toBe('g/cm³');
  });
});
