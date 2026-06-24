import { useMemo, useState } from 'react';
import { Sidebar } from '../components/Sidebar.jsx';
import { Field } from '../components/Field.jsx';
import { Stepper } from '../components/Stepper.jsx';
import { Segmented } from '../components/Segmented.jsx';
import { Checkbox } from '../components/Checkbox.jsx';
import { PlotCard } from '../components/PlotCard.jsx';
import {
  useMetadata,
  useCurvesFigure,
  useSurface2dFigure,
  useSurface3dFigure,
} from '../api/hooks.js';
import { useSettings } from '../settings/SettingsContext.jsx';

export default function PropertyExplorer() {
  const { data: metadata } = useMetadata();
  const { settings } = useSettings();

  const [modelKey, setModelKey] = useState('duska2020');
  const [property, setProperty] = useState('rho');
  const [Tmin, setTmin] = useState(200);
  const [Tmax, setTmax] = useState(300);
  const [Pmin, setPmin] = useState(0.1);
  const [Pmax, setPmax] = useState(200);
  const [nCurves, setNCurves] = useState(settings.default_n_curves);
  const [nPoints, setNPoints] = useState(settings.default_n_points);
  const [curveType, setCurveType] = useState('isobar');
  const [displayMode, setDisplayMode] = useState('curves');
  const [showPhase, setShowPhase] = useState(false);

  const baseParams = useMemo(
    () => ({
      model: modelKey,
      property,
      T_range: [Tmin, Tmax],
      P_range: [Pmin, Pmax],
    }),
    [modelKey, property, Tmin, Tmax, Pmin, Pmax]
  );

  const curvesParams = useMemo(
    () => ({
      ...baseParams,
      n_curves: nCurves,
      n_points: nPoints,
      isobar_mode: curveType === 'isobar',
      show_phase_boundaries: showPhase,
    }),
    [baseParams, nCurves, nPoints, curveType, showPhase]
  );
  // Surface grid resolution is bounded above by the schema (n_points <= 400
  // on /figures/surface{2d,3d}); the curves Stepper allows up to 500 because
  // a per-curve sample count of 500 is cheap. We clamp here so the user
  // gets a refreshed surface as soon as they change the slider rather than
  // a 422 from pydantic.
  const surfacePoints = Math.min(nPoints, 400);
  const surface2dParams = useMemo(
    () => ({
      ...baseParams,
      n_points: surfacePoints,
      colormap: 'rdbu',
      show_phase_boundaries: showPhase,
    }),
    [baseParams, surfacePoints, showPhase]
  );
  const surface3dParams = useMemo(
    () => ({
      ...baseParams,
      n_points: surfacePoints,
      colormap: 'rdbu',
      show_phase_boundaries: showPhase,
    }),
    [baseParams, surfacePoints, showPhase]
  );

  const curvesQ = useCurvesFigure(curvesParams, displayMode === 'curves');
  const s2Q = useSurface2dFigure(surface2dParams, displayMode === 'surface2d');
  const s3Q = useSurface3dFigure(surface3dParams, displayMode === 'surface3d');

  const active = displayMode === 'curves' ? curvesQ : displayMode === 'surface2d' ? s2Q : s3Q;

  const modelOptions = (metadata?.models ?? []).map((m) => ({
    value: m.key,
    label: m.display_name,
  }));
  const propertyOptions = useMemo(() => {
    const props = metadata?.models?.find((m) => m.key === modelKey)?.properties ?? [];
    const labels = metadata?.properties ?? {};
    return props.map((p) => ({ value: p, label: labels[p]?.label ?? p }));
  }, [metadata, modelKey]);

  return (
    <>
      <Sidebar>
        <Field label="Model">
          <select className="select" value={modelKey} onChange={(e) => setModelKey(e.target.value)}>
            {modelOptions.map((o) => (
              <option key={o.value} value={o.value}>
                {o.label}
              </option>
            ))}
          </select>
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
        <Field
          label={displayMode === 'curves' ? 'Points / curve' : 'Resolution'}
        >
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
        <Field label="Display">
          <Segmented
            options={[
              { value: 'curves', label: 'Curves' },
              { value: 'surface2d', label: '2D' },
              { value: 'surface3d', label: '3D' },
            ]}
            value={displayMode}
            onChange={setDisplayMode}
          />
        </Field>
        <Field>
          <Checkbox label="Show phase boundaries" checked={showPhase} onChange={setShowPhase} />
        </Field>
      </Sidebar>
      <section className="main">
        <PlotCard
          bare
          figure={active.data?.figure}
          loading={active.isLoading}
          error={active.error}
        />
      </section>
    </>
  );
}
