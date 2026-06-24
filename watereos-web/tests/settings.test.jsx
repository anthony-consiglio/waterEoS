import { describe, it, expect } from 'vitest';
import { render, screen, act } from '@testing-library/react';
import {
  SettingsProvider,
  useSettings,
  DEFAULT_SETTINGS,
} from '../src/settings/SettingsContext.jsx';

function Harness() {
  const { settings, units, setUnit, setSetting } = useSettings();
  return (
    <>
      <span data-testid="density">{units.unit_density}</span>
      <span data-testid="ncurves">{settings.default_n_curves}</span>
      <button onClick={() => setUnit('unit_density', 'g/cm³')}>change-density</button>
      <button onClick={() => setSetting('default_n_curves', 7)}>change-ncurves</button>
    </>
  );
}

describe('SettingsContext', () => {
  it('defaults match DEFAULT_SETTINGS', () => {
    render(
      <SettingsProvider>
        <Harness />
      </SettingsProvider>
    );
    expect(screen.getByTestId('density').textContent).toBe(DEFAULT_SETTINGS.unit_density);
    expect(screen.getByTestId('ncurves').textContent).toBe(
      String(DEFAULT_SETTINGS.default_n_curves)
    );
  });

  it('setUnit updates and persists to localStorage', () => {
    render(
      <SettingsProvider>
        <Harness />
      </SettingsProvider>
    );
    act(() => screen.getByText('change-density').click());
    expect(screen.getByTestId('density').textContent).toBe('g/cm³');
    const persisted = JSON.parse(localStorage.getItem('watereos_settings'));
    expect(persisted.unit_density).toBe('g/cm³');
  });

  it('setSetting updates other prefs', () => {
    render(
      <SettingsProvider>
        <Harness />
      </SettingsProvider>
    );
    act(() => screen.getByText('change-ncurves').click());
    expect(screen.getByTestId('ncurves').textContent).toBe('7');
  });

  it('reload restores persisted settings', () => {
    localStorage.setItem(
      'watereos_settings',
      JSON.stringify({ unit_density: 'g/cm³', default_n_curves: 9 })
    );
    render(
      <SettingsProvider>
        <Harness />
      </SettingsProvider>
    );
    expect(screen.getByTestId('density').textContent).toBe('g/cm³');
    expect(screen.getByTestId('ncurves').textContent).toBe('9');
  });
});
