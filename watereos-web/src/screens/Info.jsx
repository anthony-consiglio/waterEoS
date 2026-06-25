import { useMemo } from 'react';
import { useMetadata, useEosPhaseFigure } from '../api/hooks.js';
import { PlotCard } from '../components/PlotCard.jsx';

const CONCEPTS = [
  {
    t: 'Two-state model',
    b: 'Water is treated as a mixture of HDL (state A, disordered) and LDL (state B, tetrahedral). The fraction x ∈ [0, 1] gives the LDL share.',
    m: 'x',
  },
  {
    t: 'LLCP',
    b: 'The liquid-liquid critical point at the top of the HDL/LDL coexistence dome — predicted but not yet experimentally observed.',
    m: 'Tc, Pc',
  },
  {
    t: 'Spinodal',
    b: 'The stability limit in (T, P) space where compressibility diverges. Each two-state model has both an HDL and LDL spinodal.',
    m: '∂P/∂V = 0',
  },
  {
    t: 'Binodal',
    b: 'The liquid-liquid coexistence curve: (T, P) where HDL and LDL have equal Gibbs energy, lying between the two spinodals.',
    m: 'GA = GB',
  },
  {
    t: 'Widom line',
    b: 'A line of maximum correlation length emanating from the LLCP — traced as the locus of isobaric Cp maxima.',
    m: 'max(Cp)',
  },
  {
    t: 'TMD',
    b: 'Temperature of maximum density: where ρ peaks at a given P, equivalently where thermal expansivity α crosses zero.',
    m: 'α = 0',
  },
];

// References for the EoS models. Each entry mirrors the citation block at
// the top of the corresponding source file in the Python package. The
// H2O phase-diagram data provenance is described separately by
// SEAFREEZE_NOTE below.
const REFERENCES = [
  {
    group: 'Equation-of-state models',
    items: [
      {
        key: 'holten2014',
        text: 'V. Holten, J. V. Sengers, and M. A. Anisimov. "Equation of state for supercooled water at pressures up to 400 MPa." J. Phys. Chem. Ref. Data 43, 014101 (2014).',
      },
      {
        key: 'caupin2019',
        text: 'F. Caupin and M. A. Anisimov. "Thermodynamics of supercooled and stretched water: Unifying two-structure description and liquid-vapor spinodal." J. Chem. Phys. 151, 034503 (2019). Erratum: J. Chem. Phys. 163, 039902 (2025). The \'caupin2019_kim\' variant additionally incorporates the X-ray κ_T data of K. H. Kim et al., Science 358, 1589 (2017).',
      },
      {
        key: 'duska2020',
        text: 'M. Duška. "Water above the spinodal." J. Chem. Phys. 152, 174501 (2020).',
      },
      {
        key: 'shi_tanaka2020',
        text: 'R. Shi and H. Tanaka. "The anomalies and criticality of liquid water." Proc. Natl. Acad. Sci. 117, 26591–26599 (2020). Provides both the hierarchical two-state thermodynamic EoS (\'shi_tanaka2020\') and a coupled transport model (\'shi_tanaka2020_transport\') for viscosity, self-diffusion, and rotational relaxation.',
      },
      {
        key: 'grenke2025',
        text: 'L. M. Grenke and J. A. W. Elliott. "A new Tait-Tammann equation of state for liquid water." J. Phys. Chem. B 129, 1997–2012 (2025). Correction: J. Phys. Chem. B 129, 9850–9853 (2025).',
      },
      {
        key: 'singh2017',
        text: 'L. P. Singh, B. Issenmann, and F. Caupin. "Pressure dependence of viscosity in supercooled water and a unified approach for thermodynamic and dynamic anomalies of water." Proc. Natl. Acad. Sci. 114, 4312–4317 (2017).',
      },
      {
        key: 'IAPWS95',
        text: 'W. Wagner and A. Pruß. "The IAPWS formulation 1995 for the thermodynamic properties of ordinary water substance for general and scientific use." J. Phys. Chem. Ref. Data 31, 387–535 (2002).',
      },
      {
        key: 'water1',
        text: 'B. Journaux et al. "Holistic approach for studying planetary hydrospheres: Gibbs representations, ices thermodynamics, transport, and the example of Europa." J. Geophys. Res.: Planets 125, e2019JE006176 (2020). (Source of the SeaFreeze GLBF splines used for "water1" and the ice phases.)',
      },
    ],
  },
];

// Note rendered under "References" describing where the bundled water +
// ice spline data comes from. Kept as plain prose (not a citation entry)
// because it's a data-provenance note, not a paper reference.
const SEAFREEZE_NOTE = (
  <>
    The thermodynamic data driving the <code>water1</code> liquid model and
    every ice phase in the H₂O Phase Diagram tab (Ice Ih, II, III, V, VI, and
    the French–Redmer Ice VII/X parametrization) is taken from the{' '}
    <a
      href="https://github.com/Bjournaux/SeaFreeze"
      target="_blank"
      rel="noopener noreferrer"
    >
      SeaFreeze
    </a>{' '}
    project.
  </>
);

// Default hero diagram — Duska 2020 with every available overlay, on a fixed
// T = 180-280 K, P = 0-200 MPa window so the LLCP, both spinodals, ice
// liquidus, nucleation and Kauzmann lines all sit in frame.
const HERO_PARAMS = {
  model: 'duska2020',
  show: [
    'binodal',
    'hdl_spinodal',
    'ldl_spinodal',
    'LLCP',
    'tmd',
    'widom',
    'ice_ih',
    'ice_iii',
    'nuc_ih',
    'nuc_iii',
  ],
  auto_limits: false,
  T_range: [180, 280],
  P_range: [0, 200],
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

export default function Info({ setTab }) {
  const { data: metadata, isLoading } = useMetadata();
  const heroFig = useEosPhaseFigure(HERO_PARAMS);

  // Strip axis titles/legend for the compact hero rendering so the visual
  // reads as a stylized chart rather than a full analysis pane.
  const heroFigure = useMemo(() => {
    const f = heroFig.data?.figure;
    if (!f) return null;
    return {
      data: _liftBinodalToTop(f.data),
      layout: {
        ...(f.layout ?? {}),
        // The server now adds an internal title to all phase-diagram
        // figures so the EoS / H2O / Compare cards can render "bare". The
        // Info hero is a compact stylized visual that doesn't want a
        // title, so suppress it here.
        title: { text: '' },
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
            Open source · v{metadata?.version ?? '0.5.1'}
          </span>
          <h1>
            Thermodynamic equations of state for{' '}
            <span className="info-hero-em">supercooled</span> water.
          </h1>
          <p>
            A unified toolkit for {metadata?.models?.length ?? 10} equation-of-state models — explore properties, phase diagrams,
            and the liquid-liquid phase separation through interactive visualizations.
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
              </tr>
            </thead>
            <tbody>
              {metadata.models.map((m) => (
                <tr key={m.key}>
                  <td>{m.display_name}</td>
                  <td>{m.is_two_state ? '✓' : ''}</td>
                  <td>{m.has_phase_diagram ? '✓' : ''}</td>
                  <td>{m.has_transport ? '✓' : ''}</td>
                  <td>
                    {m.T_min} – {m.T_max}
                  </td>
                  <td>
                    {m.P_min} – {m.P_max}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </section>

      <section className="info-references">
        <h2>References</h2>
        {REFERENCES.map((g) => (
          <div key={g.group} className="info-references-group">
            <div className="info-references-group-title">{g.group}</div>
            <ul className="info-references-list">
              {g.items.map((r, i) => (
                <li key={r.key ?? `${g.group}-${i}`}>
                  {r.key && <span className="tag info-references-key">{r.key}</span>}
                  <span>{r.text}</span>
                </li>
              ))}
            </ul>
          </div>
        ))}
        <div className="info-references-group">
          <div className="info-references-group-title">
            H₂O phase-diagram data source
          </div>
          <p className="info-references-note">{SEAFREEZE_NOTE}</p>
        </div>
      </section>

      <section className="info-credits">
        <h2>Credits</h2>
        <div className="info-credits-grid">
          <div>
            <div className="info-credits-label">Author</div>
            <div>
              Anthony Consiglio ·{' '}
              <a href="mailto:aconsiglio4@berkeley.edu">
                aconsiglio4@berkeley.edu
              </a>
            </div>
          </div>
          <div>
            <div className="info-credits-label">Package</div>
            <div>
              waterEoS v{metadata?.version ?? '0.5.1'} ·{' '}
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
            <div>NumPy · SciPy · PyO3 / Rust · FastAPI · React · Plotly</div>
          </div>
        </div>
      </section>
    </div>
  );
}
