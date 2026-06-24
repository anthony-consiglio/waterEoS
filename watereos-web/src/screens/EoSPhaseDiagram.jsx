import { useEffect, useMemo, useState } from 'react';
import { Sidebar } from '../components/Sidebar.jsx';
import { Field } from '../components/Field.jsx';
import { Stepper } from '../components/Stepper.jsx';
import { Checkbox } from '../components/Checkbox.jsx';
import { PlotCard } from '../components/PlotCard.jsx';
import { useMetadata, useEosPhaseFigure } from '../api/hooks.js';

const OVERLAY_KEYS = [
  { key: 'binodal', label: 'Binodal' },
  { key: 'hdl_spinodal', label: 'HDL Spinodal' },
  { key: 'ldl_spinodal', label: 'LDL Spinodal' },
  { key: 'LLCP', label: 'LLCP marker' },
  { key: 'tmd', label: 'TMD (max density)' },
  { key: 'widom', label: 'Widom line' },
  { key: 'ice_ih', label: 'Ice Ih liquidus' },
  { key: 'ice_iii', label: 'Ice III liquidus' },
  { key: 'nuc_ih', label: 'Ih nucleation' },
  { key: 'nuc_iii', label: 'III nucleation' },
  { key: 'kauzmann', label: 'Kauzmann' },
];

const DEFAULT_SHOW = ['binodal', 'hdl_spinodal', 'ldl_spinodal', 'LLCP'];

// Fallback ranges in case metadata isn't loaded yet on first render.
const FALLBACK = { Tmin: 180, Tmax: 300, Pmin: 0.1, Pmax: 200 };

export default function EoSPhaseDiagram() {
  const { data: metadata } = useMetadata();
  const phaseModels = useMemo(
    () => (metadata?.models ?? []).filter((m) => m.has_phase_diagram),
    [metadata]
  );

  const [modelKey, setModelKey] = useState('duska2020');
  const [show, setShow] = useState(DEFAULT_SHOW);
  const [Tmin, setTmin] = useState(FALLBACK.Tmin);
  const [Tmax, setTmax] = useState(FALLBACK.Tmax);
  const [Pmin, setPmin] = useState(FALLBACK.Pmin);
  const [Pmax, setPmax] = useState(FALLBACK.Pmax);

  // When the user picks a new model, reset the range steppers to that
  // model's published validity envelope so the chart isn't unexpectedly
  // empty (e.g. switching to Caupin reaches into negative pressures).
  useEffect(() => {
    const info = phaseModels.find((m) => m.key === modelKey);
    if (!info) return;
    setTmin(info.T_min);
    setTmax(info.T_max);
    setPmin(info.P_min);
    setPmax(info.P_max);
  }, [modelKey, phaseModels]);

  const params = useMemo(
    () => ({
      model: modelKey,
      show,
      auto_limits: false,
      T_range: [Tmin, Tmax],
      P_range: [Pmin, Pmax],
    }),
    [modelKey, show, Tmin, Tmax, Pmin, Pmax]
  );

  const fig = useEosPhaseFigure(params);

  const toggle = (k) =>
    setShow((cur) => (cur.includes(k) ? cur.filter((x) => x !== k) : [...cur, k]));

  return (
    <>
      <Sidebar>
        <Field label="Model">
          <select className="select" value={modelKey} onChange={(e) => setModelKey(e.target.value)}>
            {phaseModels.map((m) => (
              <option key={m.key} value={m.key}>
                {m.display_name}
              </option>
            ))}
          </select>
        </Field>
        <Field label="T range [K]">
          <div className="input-pair">
            <Stepper value={Tmin} onChange={setTmin} step={1} />
            <Stepper value={Tmax} onChange={setTmax} step={1} />
          </div>
        </Field>
        <Field label="P range [MPa]">
          <div className="input-pair">
            <Stepper value={Pmin} onChange={setPmin} step={1} />
            <Stepper value={Pmax} onChange={setPmax} step={1} />
          </div>
        </Field>
        <Field label="Overlays">
          <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
            {OVERLAY_KEYS.map((o) => (
              <Checkbox
                key={o.key}
                label={o.label}
                checked={show.includes(o.key)}
                onChange={() => toggle(o.key)}
              />
            ))}
          </div>
        </Field>
      </Sidebar>
      <section className="main">
        <PlotCard
          bare
          className="plot-card-square"
          figure={fig.data?.figure}
          loading={fig.isLoading}
          error={fig.error}
        />
      </section>
    </>
  );
}
