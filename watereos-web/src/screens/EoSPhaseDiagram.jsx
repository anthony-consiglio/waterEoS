import { useMemo, useState } from 'react';
import { Sidebar } from '../components/Sidebar.jsx';
import { Field } from '../components/Field.jsx';
import { Stepper } from '../components/Stepper.jsx';
import { Checkbox } from '../components/Checkbox.jsx';
import { PlotCard } from '../components/PlotCard.jsx';
import { useMetadata, useEosPhaseFigure } from '../api/hooks.js';
import { OVERLAY_GROUPS, DEFAULT_SHOW, DEFAULT_RANGES } from './eosPhaseLayers.js';

export default function EoSPhaseDiagram() {
  const { data: metadata } = useMetadata();
  const phaseModels = useMemo(
    () => (metadata?.models ?? []).filter((m) => m.has_phase_diagram),
    [metadata]
  );

  const [modelKey, setModelKey] = useState('caupin2019');
  const [show, setShow] = useState(DEFAULT_SHOW);
  const [Tmin, setTmin] = useState(DEFAULT_RANGES.Tmin);
  const [Tmax, setTmax] = useState(DEFAULT_RANGES.Tmax);
  const [Pmin, setPmin] = useState(DEFAULT_RANGES.Pmin);
  const [Pmax, setPmax] = useState(DEFAULT_RANGES.Pmax);

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

  const resetView = () => {
    setTmin(DEFAULT_RANGES.Tmin);
    setTmax(DEFAULT_RANGES.Tmax);
    setPmin(DEFAULT_RANGES.Pmin);
    setPmax(DEFAULT_RANGES.Pmax);
  };

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
            <Stepper value={Tmin} onChange={setTmin} step={5} />
            <Stepper value={Tmax} onChange={setTmax} step={5} />
          </div>
        </Field>
        <Field label="P range [MPa]">
          <div className="input-pair">
            <Stepper value={Pmin} onChange={setPmin} step={10} />
            <Stepper value={Pmax} onChange={setPmax} step={10} />
          </div>
        </Field>
        <button type="button" className="btn btn-ghost" onClick={resetView}>
          Reset view
        </button>
        {OVERLAY_GROUPS.map((group) => (
          <Field key={group.label} label={group.label}>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
              {group.items.map((o) => (
                <Checkbox
                  key={o.key}
                  label={o.label}
                  checked={show.includes(o.key)}
                  onChange={() => toggle(o.key)}
                />
              ))}
            </div>
          </Field>
        ))}
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
