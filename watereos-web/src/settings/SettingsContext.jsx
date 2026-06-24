import { createContext, useCallback, useContext, useEffect, useMemo, useState } from 'react';

const KEY = 'watereos_settings';

export const DEFAULT_SETTINGS = {
  unit_temperature: 'K',
  unit_pressure: 'MPa',
  unit_density: 'kg/m³',
  unit_volume: 'm³/kg',
  unit_energy: 'J/kg',
  unit_entropy: 'J/(kg·K)',
  unit_bulk_modulus: 'MPa',
  unit_viscosity: 'Pa·s',
  default_n_curves: 5,
  default_n_points: 200,
};

const UNIT_KEYS = [
  'unit_temperature',
  'unit_pressure',
  'unit_density',
  'unit_volume',
  'unit_energy',
  'unit_entropy',
  'unit_bulk_modulus',
  'unit_viscosity',
];

const SettingsContext = createContext({
  settings: DEFAULT_SETTINGS,
  units: {},
  setSetting: () => {},
  setUnit: () => {},
  reset: () => {},
});

function _load() {
  try {
    const raw = localStorage.getItem(KEY);
    if (!raw) return DEFAULT_SETTINGS;
    const parsed = JSON.parse(raw);
    return { ...DEFAULT_SETTINGS, ...parsed };
  } catch {
    return DEFAULT_SETTINGS;
  }
}

export function SettingsProvider({ children }) {
  const [settings, setSettings] = useState(_load);

  useEffect(() => {
    try {
      localStorage.setItem(KEY, JSON.stringify(settings));
    } catch {
      // ignore
    }
  }, [settings]);

  const setSetting = useCallback((key, value) => {
    setSettings((s) => ({ ...s, [key]: value }));
  }, []);

  const setUnit = useCallback(
    (key, value) => {
      if (!UNIT_KEYS.includes(key)) {
        return;
      }
      setSetting(key, value);
    },
    [setSetting]
  );

  const reset = useCallback(() => setSettings(DEFAULT_SETTINGS), []);

  const units = useMemo(() => {
    const u = {};
    for (const k of UNIT_KEYS) u[k] = settings[k];
    return u;
  }, [settings]);

  return (
    <SettingsContext.Provider value={{ settings, units, setSetting, setUnit, reset }}>
      {children}
    </SettingsContext.Provider>
  );
}

export function useSettings() {
  return useContext(SettingsContext);
}
