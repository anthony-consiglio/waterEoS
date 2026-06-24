import { useMemo, useState } from 'react';
import { Sidebar } from '../components/Sidebar.jsx';
import { Field } from '../components/Field.jsx';
import { Stepper } from '../components/Stepper.jsx';
import { useMetadata, usePoint } from '../api/hooks.js';

function formatValue(v) {
  if (v === null || v === undefined) return '—';
  if (typeof v !== 'number') return String(v);
  if (!Number.isFinite(v)) return String(v);
  const abs = Math.abs(v);
  if (abs === 0) return '0';
  if (abs >= 1e5 || abs < 1e-3) return v.toExponential(4);
  if (abs >= 100) return v.toFixed(3);
  if (abs >= 1) return v.toFixed(4);
  return v.toFixed(5);
}

export default function PointCalculator() {
  const { data: metadata } = useMetadata();
  const [modelKey, setModelKey] = useState('duska2020');
  const [TK, setTK] = useState(273.15);
  const [PMPa, setPMPa] = useState(0.1);

  const params = useMemo(
    () => ({ model_keys: [modelKey], T_K: TK, P_MPa: PMPa }),
    [modelKey, TK, PMPa]
  );

  const pointQ = usePoint(params);
  const results = pointQ.data?.results?.[modelKey] ?? null;
  const warnings = pointQ.data?.warnings ?? [];

  const modelInfo = (metadata?.models ?? []).find((m) => m.key === modelKey);
  const propLabels = metadata?.properties ?? {};
  const allModels = metadata?.models ?? [];

  const x = results?.x;

  // Preserve the model's natural property order from the registry; fall
  // back to whatever the result object yields if the registry data isn't
  // loaded yet.
  const orderedRows = useMemo(() => {
    if (!results) return [];
    const order = modelInfo?.properties ?? Object.keys(results);
    const rows = [];
    for (const k of order) {
      const v = results[k];
      if (v === null || v === undefined) continue;
      rows.push([k, v]);
    }
    return rows;
  }, [results, modelInfo]);

  return (
    <>
      <Sidebar>
        <Field label="Model">
          <select className="select" value={modelKey} onChange={(e) => setModelKey(e.target.value)}>
            {allModels.map((m) => (
              <option key={m.key} value={m.key}>
                {m.display_name}
              </option>
            ))}
          </select>
        </Field>
        <Field label="T [K]">
          <Stepper value={TK} onChange={setTK} step={0.1} />
        </Field>
        <Field label="P [MPa]">
          <Stepper value={PMPa} onChange={setPMPa} step={0.1} />
        </Field>
      </Sidebar>
      <section className="main scroll-y" style={{ padding: 'var(--pad-5)' }}>
        {pointQ.error && (
          <div className="plot-error" role="alert" style={{ marginBottom: 'var(--pad-4)' }}>
            {String(pointQ.error?.message ?? pointQ.error)}
          </div>
        )}

        {modelInfo?.is_two_state && typeof x === 'number' && (
          <div className="card" style={{ marginBottom: 'var(--pad-4)' }}>
            <div className="section-title" style={{ marginBottom: 8 }}>
              State composition
            </div>
            <div
              style={{
                position: 'relative',
                height: 14,
                borderRadius: 7,
                background: 'var(--bg-tint)',
                overflow: 'hidden',
              }}
            >
              <div
                style={{
                  position: 'absolute',
                  inset: 0,
                  width: `${(1 - x) * 100}%`,
                  background: 'var(--curve-2)',
                }}
              />
              <div
                style={{
                  position: 'absolute',
                  top: 0,
                  bottom: 0,
                  left: `${(1 - x) * 100}%`,
                  right: 0,
                  background: 'var(--curve-5)',
                }}
              />
            </div>
            <div
              style={{
                display: 'flex',
                justifyContent: 'space-between',
                marginTop: 6,
                fontFamily: 'var(--font-mono)',
                fontSize: 11,
                color: 'var(--text-muted)',
              }}
            >
              <span>HDL {(1 - x).toFixed(3)}</span>
              <span>LDL {x.toFixed(3)}</span>
            </div>
          </div>
        )}

        <div className="card" style={{ padding: 0, overflow: 'hidden' }}>
          <table className="results-table">
            <thead>
              <tr>
                <th>Property</th>
                <th className="num">Value</th>
                <th>Unit</th>
              </tr>
            </thead>
            <tbody>
              {orderedRows.map(([k, v]) => (
                <tr key={k}>
                  <td>{propLabels[k]?.label ?? k}</td>
                  <td className="num">{formatValue(v)}</td>
                  <td className="unit">{propLabels[k]?.unit ?? ''}</td>
                </tr>
              ))}
              {orderedRows.length === 0 && (
                <tr>
                  <td colSpan={3} style={{ textAlign: 'center', color: 'var(--text-faint)' }}>
                    {pointQ.isLoading ? 'Computing…' : 'No results'}
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>

        {warnings.length > 0 && (
          <div className="card" style={{ marginTop: 'var(--pad-4)' }}>
            <div className="section-title" style={{ marginBottom: 8 }}>
              Validity warnings
            </div>
            <ul style={{ margin: 0, paddingLeft: 18 }}>
              {warnings.map((w, i) => (
                <li key={i}>
                  <strong>{w.model}</strong>: {w.message}
                </li>
              ))}
            </ul>
          </div>
        )}
      </section>
    </>
  );
}
