import { useMemo, useState } from 'react';
import { Sidebar } from '../components/Sidebar.jsx';
import { Field } from '../components/Field.jsx';
import { Stepper } from '../components/Stepper.jsx';
import { Segmented } from '../components/Segmented.jsx';
import { Checkbox } from '../components/Checkbox.jsx';
import { PlotCard } from '../components/PlotCard.jsx';
import { useMetadata, useCompareFigure } from '../api/hooks.js';
import { useSettings } from '../settings/SettingsContext.jsx';

export default function ModelComparison() {
  const { data: metadata } = useMetadata();
  const { settings } = useSettings();

  const [modelKeys, setModelKeys] = useState(['duska2020']);
  const [property, setProperty] = useState('rho');
  const [Tmin, setTmin] = useState(200);
  const [Tmax, setTmax] = useState(300);
  const [Pmin, setPmin] = useState(0.1);
  const [Pmax, setPmax] = useState(200);
  const [nCurves, setNCurves] = useState(settings.default_n_curves);
  const [nPoints, setNPoints] = useState(settings.default_n_points);
  const [curveType, setCurveType] = useState('isobar');
  const [layout, setLayout] = useState('overlay');

  const params = useMemo(
    () => ({
      model_keys: modelKeys,
      property,
      T_range: [Tmin, Tmax],
      P_range: [Pmin, Pmax],
      n_curves: nCurves,
      n_points: nPoints,
      isobar_mode: curveType === 'isobar',
      layout,
    }),
    [modelKeys, property, Tmin, Tmax, Pmin, Pmax, nCurves, nPoints, curveType, layout]
  );

  const fig = useCompareFigure(params, modelKeys.length > 0);

  const toggleModel = (k) =>
    setModelKeys((cur) => (cur.includes(k) ? cur.filter((x) => x !== k) : [...cur, k]));

  const allModels = useMemo(() => metadata?.models ?? [], [metadata]);
  const allProperties = useMemo(() => metadata?.properties ?? {}, [metadata]);

  // property options: intersection of properties from the selected models
  const propertyOptions = useMemo(() => {
    if (modelKeys.length === 0) return [];
    const selectedModels = allModels.filter((m) => modelKeys.includes(m.key));
    if (selectedModels.length === 0) return [];
    const sets = selectedModels.map((m) => new Set(m.properties));
    const common = [...sets[0]].filter((p) => sets.every((s) => s.has(p)));
    return common.map((p) => ({ value: p, label: allProperties[p]?.label ?? p }));
  }, [modelKeys, allModels, allProperties]);

  return (
    <>
      <Sidebar>
        <Field label="Models">
          <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
            {allModels.map((m) => (
              <Checkbox
                key={m.key}
                label={m.display_name}
                checked={modelKeys.includes(m.key)}
                onChange={() => toggleModel(m.key)}
              />
            ))}
          </div>
        </Field>
        <Field label="Property">
          <select className="select" value={property} onChange={(e) => setProperty(e.target.value)}>
            {propertyOptions.map((o) => (
              <option key={o.value} value={o.value}>
                {o.label}
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
        <Field label="Curves">
          <Stepper value={nCurves} onChange={setNCurves} min={1} max={20} step={1} />
        </Field>
        <Field label="Points / curve">
          <Stepper value={nPoints} onChange={setNPoints} min={20} max={500} step={10} />
        </Field>
        <Field label="Curve type">
          <Segmented
            options={[
              { value: 'isobar', label: 'Isobars' },
              { value: 'isotherm', label: 'Isotherms' },
            ]}
            value={curveType}
            onChange={setCurveType}
          />
        </Field>
        <Field label="Layout">
          <Segmented
            options={[
              { value: 'overlay', label: 'Overlay' },
              { value: 'sidebyside', label: 'Side by side' },
            ]}
            value={layout}
            onChange={setLayout}
          />
        </Field>
      </Sidebar>
      <section className="main">
        <PlotCard
          bare
          figure={fig.data?.figure}
          loading={fig.isLoading}
          error={fig.error}
        />
      </section>
    </>
  );
}
