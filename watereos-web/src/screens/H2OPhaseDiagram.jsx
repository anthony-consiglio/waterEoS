import { useMemo, useState } from 'react';
import { Sidebar } from '../components/Sidebar.jsx';
import { Field } from '../components/Field.jsx';
import { Stepper } from '../components/Stepper.jsx';
import { Segmented } from '../components/Segmented.jsx';
import { PlotCard } from '../components/PlotCard.jsx';
import { useH2OPhaseFigure } from '../api/hooks.js';

export default function H2OPhaseDiagram() {
  const [projection, setProjection] = useState('tv');
  const [Vmin, setVmin] = useState(7e-4);
  const [Vmax, setVmax] = useState(1.1e-3);
  const [Tmin, setTmin] = useState(190);
  const [Tmax, setTmax] = useState(300);
  const [Pmin, setPmin] = useState(1e-4);
  const [Pmax, setPmax] = useState(1000);

  const params = useMemo(
    () => ({
      projection,
      V_range: [Vmin, Vmax],
      T_range: [Tmin, Tmax],
      P_range: [Pmin, Pmax],
    }),
    [projection, Vmin, Vmax, Tmin, Tmax, Pmin, Pmax]
  );

  const fig = useH2OPhaseFigure(params);

  return (
    <>
      <Sidebar>
        <Field label="Projection">
          <Segmented
            options={[
              { value: 'tv', label: 'T–V' },
              { value: 'tp', label: 'T–P' },
              { value: 'ptv', label: '3D P–T–V' },
            ]}
            value={projection}
            onChange={setProjection}
          />
        </Field>
        <Field label="V range [m³/kg]">
          <div className="input-pair">
            <Stepper value={Vmin} onChange={setVmin} step={1e-4} />
            <Stepper value={Vmax} onChange={setVmax} step={1e-4} />
          </div>
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
