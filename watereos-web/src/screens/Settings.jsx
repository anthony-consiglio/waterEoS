import { Field } from '../components/Field.jsx';
import { Stepper } from '../components/Stepper.jsx';
import { useMetadata } from '../api/hooks.js';
import { useSettings, DEFAULT_SETTINGS } from '../settings/SettingsContext.jsx';

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

export default function Settings() {
  const { data: metadata } = useMetadata();
  const { settings, setSetting, setUnit, reset } = useSettings();

  const unitOptions = metadata?.units?.options ?? {};
  const categoryLabels = metadata?.units?.category_labels ?? {};

  return (
    <div className="settings-shell scroll-y">
      <section className="card" style={{ marginBottom: 'var(--pad-5)' }}>
        <div className="section-title" style={{ marginBottom: 12 }}>
          Units
        </div>
        {UNIT_KEYS.map((key) => {
          const options = unitOptions[key] ?? [];
          const label = categoryLabels[key] ?? key;
          return (
            <Field key={key} label={label}>
              <select
                id={key}
                aria-label={label}
                className="select"
                value={settings[key] ?? DEFAULT_SETTINGS[key]}
                onChange={(e) => setUnit(key, e.target.value)}
              >
                {options.length === 0 ? (
                  <option value={settings[key]}>{settings[key]}</option>
                ) : (
                  options.map((o) => (
                    <option key={o.value} value={o.value}>
                      {o.label}
                    </option>
                  ))
                )}
              </select>
            </Field>
          );
        })}
      </section>

      <section className="card" style={{ marginBottom: 'var(--pad-5)' }}>
        <div className="section-title" style={{ marginBottom: 12 }}>
          Plot defaults
        </div>
        <Field label="Default curves">
          <Stepper
            value={settings.default_n_curves}
            onChange={(v) => setSetting('default_n_curves', v)}
            min={1}
            max={20}
            step={1}
          />
        </Field>
        <Field label="Default points per curve">
          <Stepper
            value={settings.default_n_points}
            onChange={(v) => setSetting('default_n_points', v)}
            min={20}
            max={500}
            step={10}
          />
        </Field>
      </section>

      <section className="card">
        <button type="button" className="btn btn-ghost" onClick={reset}>
          Reset all settings to defaults
        </button>
      </section>
    </div>
  );
}
