import { useMemo } from 'react';
import { useMetadata, useEosPhaseFigure } from '../api/hooks.js';
import { PlotCard } from '../components/PlotCard.jsx';

const CONCEPTS = [
  {
    t: 'Two-state model',
    b: 'Liquid water is treated as a mixture of a high-density (HDL) and a low-density (LDL) local structure. The fraction x ∈ [0, 1] is the LDL share.',
    m: 'x',
  },
  {
    t: 'Liquid–liquid critical point',
    b: 'The end of the HDL/LDL coexistence line, predicted by the two-state models but not yet confirmed experimentally.',
    m: 'Tc, Pc',
  },
  {
    t: 'Coexistence line',
    b: 'Where HDL and LDL have equal Gibbs energy. Below about 150 K some models’ lines are reentrant and turn back toward lower pressure.',
    m: 'G_HDL = G_LDL',
  },
  {
    t: 'Stability limits',
    b: 'Beyond its stability limit (spinodal) the metastable liquid ceases to exist. The HDL and LDL limits meet at the critical point.',
    m: '∂²G/∂x² = 0',
  },
  {
    t: 'Widom line',
    b: 'The continuation of the coexistence line above the critical point, where the two structures are equally abundant. Response functions peak near it.',
    m: 'x = 1/2',
  },
  {
    t: 'Density maximum',
    b: 'The temperature of maximum density (TMD) along each isobar, where the thermal expansivity α changes sign: 277.13 K at 1 atm.',
    m: 'α = 0',
  },
  {
    t: 'Response extrema',
    b: 'Lines where compressibility and sound speed peak or dip along isobars, and heat capacity along isotherms. Solid lines are maxima, dashed lines minima.',
    m: 'max / min',
  },
  {
    t: 'Liquid–vapor spinodal',
    b: 'The largest tension the liquid can bear before it becomes mechanically unstable and cavitates. It bounds stretched (negative-pressure) water; Caupin (2019) and Duška (2020) include it.',
    m: '(∂P/∂V)_T = 0',
  },
  {
    t: 'Homogeneous nucleation',
    b: 'Below this line ice nucleates spontaneously in small samples, so bulk-liquid data are scarce and the models are extrapolated (Holten et al. 2014, App. A).',
    m: 'T_H',
  },
];

// Note rendered under "References" describing where the bundled water +
// ice spline data comes from. Kept as plain prose (not a citation entry)
// because it's a data-provenance note, not a paper reference.
const SEAFREEZE_NOTE = (
  <>
    The thermodynamic data driving the <code>water1</code> liquid model and every ice phase in the
    H₂O Phase Diagram tab (Ice Ih, II, III, V, VI, and the French–Redmer Ice VII/X parametrization)
    is taken from the{' '}
    <a href="https://github.com/Bjournaux/SeaFreeze" target="_blank" rel="noopener noreferrer">
      SeaFreeze
    </a>{' '}
    project.
  </>
);

// Hero diagram: Caupin & Anisimov (2019) in the EoS Phase Diagram tab's style
// and standard window (100-375 K, -400 to 420 MPa), model curves only.
const HERO_PARAMS = {
  model: 'caupin2019',
  show: [
    'binodal',
    'LLCP',
    'hdl_spinodal',
    'ldl_spinodal',
    'widom',
    'lv_spinodal',
    'vapor_liquid',
    'nucleation',
    'ice_ih',
    'ice_iii',
    'ice_v',
    'density',
    'compressibility',
    'sound_speed',
    'heat_capacity',
  ],
  auto_limits: true,
};

// Drawing order in Plotly is array order — later traces sit on top. Lift any
// binodal trace (matched by name or legendgroup) to the end so it never gets
// occluded by spinodals or other overlays.
function _liftBinodalToTop(data) {
  if (!Array.isArray(data)) return data;
  const isBinodal = (t) => {
    const name = (t?.name ?? '').toLowerCase();
    const group = (t?.legendgroup ?? '').toLowerCase();
    return name.includes('binodal') || group.includes('binodal');
  };
  const front = data.filter((t) => !isBinodal(t));
  const top = data.filter(isBinodal);
  return [...front, ...top];
}

// A DOI as a link; with children, the children are the link text.
function DoiLink({ doi, children }) {
  if (!doi) return children ?? null;
  return (
    <a
      className={children ? undefined : 'info-references-doi'}
      href={`https://doi.org/${doi}`}
      target="_blank"
      rel="noopener noreferrer"
    >
      {children ?? `doi:${doi}`}
    </a>
  );
}

export default function Info({ setTab }) {
  const { data: metadata, isLoading } = useMetadata();
  // reference key -> model keys that cite it
  const usedBy = useMemo(() => {
    const out = {};
    for (const m of metadata?.models ?? []) {
      for (const key of m.references ?? []) (out[key] ??= []).push(m.key);
    }
    return out;
  }, [metadata]);
  const heroFig = useEosPhaseFigure(HERO_PARAMS);

  // Strip axis titles/legend for the compact hero rendering so the visual
  // reads as a stylized chart rather than a full analysis pane.
  const heroFigure = useMemo(() => {
    const f = heroFig.data?.figure;
    if (!f) return null;
    return {
      // Model curves only: drop the measured points, keep the critical-point marker.
      data: _liftBinodalToTop(
        (f.data ?? []).filter((t) => t.mode !== 'markers' || t.legendgroup === 'LLCP')
      ),
      layout: {
        ...(f.layout ?? {}),
        // The server now adds an internal title to all phase-diagram
        // figures so the EoS / H2O / Compare cards can render "bare". The
        // Info hero is a compact stylized visual that doesn't want a
        // title, so suppress it here.
        title: { text: '' },
        annotations: [],
        // automargin lets Plotly grow the margin as needed for the axis
        // title; the explicit numbers act as a safe floor so a quick first
        // paint isn't clipped before automargin kicks in.
        margin: { l: 64, r: 32, t: 16, b: 72, autoexpand: true },
        showlegend: false,
        xaxis: {
          ...(f.layout?.xaxis ?? {}),
          automargin: true,
          title: { text: 'Temperature [K] →', standoff: 12 },
        },
        yaxis: {
          ...(f.layout?.yaxis ?? {}),
          automargin: true,
          title: { text: 'Pressure [MPa] →', standoff: 12 },
        },
      },
    };
  }, [heroFig.data]);

  return (
    <div className="info-shell scroll-y" style={{ flex: 1 }}>
      <section className="info-hero">
        <div className="info-hero-text">
          <span className="tag tag-accent info-hero-badge">
            <span className="info-hero-dot" />
            Open source{metadata?.version ? ` · v${metadata.version}` : ''}
          </span>
          <h1>
            Thermodynamic equations of state for <span className="info-hero-em">supercooled</span>{' '}
            water.
          </h1>
          <p>
            A unified toolkit for {metadata?.models?.length ?? 'many'} equation-of-state models —
            explore properties, phase diagrams, measured data, and the liquid-liquid phase
            separation through interactive visualizations.
          </p>
          <div className="info-hero-actions">
            <button
              className="btn btn-primary"
              onClick={() => setTab && setTab('explorer')}
              type="button"
            >
              Open Property Explorer →
            </button>
            <button
              className="btn btn-ghost mono"
              onClick={() => {
                try {
                  navigator.clipboard?.writeText('pip install waterEoS');
                } catch (_) {
                  /* clipboard unavailable */
                }
              }}
              type="button"
              title="Copy to clipboard"
            >
              pip install waterEoS
            </button>
            <a
              className="btn btn-ghost"
              href="https://github.com/anthony-consiglio/waterEoS"
              target="_blank"
              rel="noopener noreferrer"
              title="Open the waterEoS repository on GitHub"
            >
              <svg
                aria-hidden="true"
                width="14"
                height="14"
                viewBox="0 0 16 16"
                fill="currentColor"
                style={{ marginRight: 6 }}
              >
                <path d="M8 0C3.58 0 0 3.58 0 8a8 8 0 005.47 7.59c.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82a7.42 7.42 0 014 0c1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01 8.01 0 0016 8c0-4.42-3.58-8-8-8z" />
              </svg>
              GitHub
            </a>
          </div>
        </div>
        <div className="info-hero-visual">
          <PlotCard
            bare
            className="info-hero-plot"
            figure={heroFigure}
            loading={heroFig.isLoading}
            error={heroFig.error}
          />
        </div>
      </section>

      <div className="section-title info-section-title">Key concepts</div>
      <section className="info-concepts">
        {CONCEPTS.map((c) => (
          <div
            key={c.t}
            className="card"
            style={{ display: 'flex', flexDirection: 'column', gap: 10 }}
          >
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
              <div style={{ fontWeight: 600, fontSize: 14, letterSpacing: '-0.005em' }}>{c.t}</div>
              <span className="tag">{c.m}</span>
            </div>
            <div style={{ fontSize: 13, color: 'var(--text-muted)', lineHeight: 1.5 }}>{c.b}</div>
          </div>
        ))}
      </section>

      <section className="info-models">
        <h2>Models</h2>
        {isLoading && <p>Loading models…</p>}
        {metadata && (
          <table className="models-table">
            <thead>
              <tr>
                <th>Model</th>
                <th>Two-state</th>
                <th>Phase diagram</th>
                <th>Transport</th>
                <th>T range [K]</th>
                <th>P range [MPa]</th>
                <th>Reference</th>
              </tr>
            </thead>
            <tbody>
              {metadata.models.map((m) => (
                <tr key={m.key}>
                  <td>{m.display_name}</td>
                  <td>{m.is_two_state ? '✓' : ''}</td>
                  <td>{m.has_phase_diagram ? '✓' : ''}</td>
                  <td>{m.has_transport ? '✓' : ''}</td>
                  <td className="nowrap">
                    {m.T_min} – {m.T_max}
                  </td>
                  <td className="nowrap">
                    {m.P_min} – {m.P_max}
                  </td>
                  <td>
                    {(m.references ?? []).map((key, i) => {
                      const r = metadata.references?.[key];
                      if (!r) return null;
                      return (
                        <span key={key}>
                          {i > 0 && '; '}
                          <DoiLink doi={r.doi}>{r.short}</DoiLink>
                        </span>
                      );
                    })}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </section>

      <section className="info-references">
        <h2>References</h2>
        {isLoading && <p>Loading references…</p>}
        {(metadata?.reference_groups ?? []).map((g) => (
          <div key={g.title} className="info-references-group">
            <div className="info-references-group-title">{g.title}</div>
            <ul className="info-references-list">
              {g.keys.map((key) => {
                const r = metadata.references?.[key];
                if (!r) return null;
                const models = usedBy[key] ?? [];
                return (
                  <li key={key}>
                    <span className="tag info-references-key">{r.short}</span>
                    <span>
                      {r.citation} <DoiLink doi={r.doi} />
                      {(r.notes ?? []).map((n) => (
                        <span key={n.doi} className="info-references-extra">
                          {n.text} <DoiLink doi={n.doi} />
                        </span>
                      ))}
                      {models.length > 0 && (
                        <span className="info-references-extra info-references-models">
                          Models: {models.join(', ')}
                        </span>
                      )}
                    </span>
                  </li>
                );
              })}
            </ul>
          </div>
        ))}
        <div className="info-references-group">
          <div className="info-references-group-title">H₂O phase-diagram data source</div>
          <p className="info-references-note">{SEAFREEZE_NOTE}</p>
        </div>
        <div className="info-references-group">
          <div className="info-references-group-title">Measured data</div>
          <p className="info-references-note">
            Every point in the Measured Data tab carries its source; the tab&apos;s Sources view
            lists them with full citations.
          </p>
        </div>
      </section>

      <section className="info-credits">
        <h2>Credits</h2>
        <div className="info-credits-grid">
          <div>
            <div className="info-credits-label">Author</div>
            <div>
              Anthony Consiglio ·{' '}
              <a href="mailto:aconsiglio4@berkeley.edu">aconsiglio4@berkeley.edu</a>
            </div>
          </div>
          <div>
            <div className="info-credits-label">Package</div>
            <div>
              waterEoS{metadata?.version ? ` v${metadata.version}` : ''} ·{' '}
              <a
                href="https://pypi.org/project/waterEoS/"
                target="_blank"
                rel="noopener noreferrer"
              >
                PyPI
              </a>{' '}
              ·{' '}
              <a
                href="https://github.com/anthony-consiglio/waterEoS"
                target="_blank"
                rel="noopener noreferrer"
              >
                GitHub
              </a>
            </div>
          </div>
          <div>
            <div className="info-credits-label">License</div>
            <div>GPL-3.0-only</div>
          </div>
          <div>
            <div className="info-credits-label">Built with</div>
            <div>NumPy · SciPy · contourpy · PyO3 / Rust · FastAPI · React · Plotly</div>
          </div>
        </div>
      </section>
    </div>
  );
}
