// The H2O Phase Diagram tab: phase equilibria of water + NaCl, the atlas of
// cryo-isochoric (without its process view), in that atlas's own style.
// The figures are precomputed and served as stored by /api/atlas.
import { useCallback, useEffect, useRef, useState } from 'react';
import { fetchAtlasManifest } from '../api/client.js';
import { loadPlotly } from '../plotlyModule.js';
import { atlasFile } from '../h2oAtlas/data.js';
import {
  UNITS,
  captionNote,
  comp,
  fileOf,
  fitted,
  hasTies,
  inUnit,
  openingStep,
  settle,
  shortTitle,
  stageLayout,
  uFor,
  ungroup,
} from '../h2oAtlas/atlas.js';
import '../h2oAtlas/atlas.css';

const BREAKPOINT = 1020; // below it the notes slide over the figure instead of standing beside it
const wide = () => typeof window === 'undefined' || window.innerWidth > BREAKPOINT;

// What the tab shows, kept while the visitor is on other tabs. It opens on
// the isoplethal T-p diagram, whose first step (x = 0) is the phase diagram
// of pure water.
const memory = {
  view: 'd-x-Tp',
  unit: 'wt',
  logAxes: false,
  ties: true,
  card: 'solute',
  frame: null,
  lastOf: {},
};

// A view chosen by the visitor opens on its default step, and is the one its
// group returns to.
function rememberViewChoice(group, id) {
  memory.frame = null;
  if (group) memory.lastOf[group] = id;
}

function showTies(Plotly, plot, visible) {
  if (!plot || !plot.data) return;
  const idx = plot.data.map((t, i) => (t.name === 'tie-lines' ? i : -1)).filter((i) => i >= 0);
  if (idx.length) Plotly.restyle(plot, { visible }, idx);
}

export default function H2OPhaseDiagram() {
  const [manifest, setManifest] = useState(null);
  const [view, setView] = useState(memory.view);
  const [unit, setUnit] = useState(memory.unit);
  const [logAxes, setLogAxes] = useState(memory.logAxes);
  const [ties, setTies] = useState(memory.ties);
  const [notes, setNotes] = useState(wide);
  const [card, setCard] = useState(memory.card);
  const [info, setInfo] = useState(null);
  const [status, setStatus] = useState('loading…');
  const plotRef = useRef(null);
  const stageRef = useRef(null);
  const token = useRef(0);
  const viewRef = useRef(view);
  const tiesRef = useRef(ties);

  useEffect(() => {
    let live = true;
    fetchAtlasManifest()
      .then((m) => {
        if (!live) return;
        if (!m.configured)
          setStatus(m.detail || 'The phase diagrams are not loaded on this server.');
        else setManifest(m);
      })
      .catch((e) => live && setStatus(e.message || String(e)));
    return () => {
      live = false;
    };
  }, []);

  useEffect(() => {
    Object.assign(memory, { view, unit, logAxes, ties, card });
    viewRef.current = view;
    tiesRef.current = ties;
  }, [view, unit, logAxes, ties, card]);

  const views = manifest?.views ?? [];
  const viewDef = views.find((v) => v.id === view) ?? views[0];
  const solute = manifest?.solutes?.[0];
  const group = manifest?.groups.find((g) => g.id === viewDef?.group);

  // Draw the view: the stored figure, in the unit chosen, fitted to the stage.
  useEffect(() => {
    if (!manifest || !viewDef || !solute) return undefined;
    const my = ++token.current;
    const id = viewDef.id;
    (async () => {
      setStatus('loading…');
      try {
        const Plotly = await loadPlotly();
        const plot = plotRef.current;
        if (my !== token.current || !plot) return;
        Plotly.purge(plot);
        const file = fileOf(viewDef, logAxes);
        const [stored, inf] = await Promise.all([
          atlasFile(manifest.release_id, solute.id, file),
          atlasFile(manifest.release_id, solute.id, 'info'),
        ]);
        if (my !== token.current) return;
        setInfo(inf);
        const fig = inUnit(stored, uFor(solute, unit), solute); // wt % is what is stored
        const stage = stageRef.current;
        const layout = stageLayout(fig, id, stage.clientWidth, stage.clientHeight);
        setStatus(null);
        await Plotly.newPlot(plot, ungroup(fig.data), layout, {
          displaylogo: false,
          responsive: false,
          modeBarButtonsToRemove: ['lasso2d', 'select2d', 'autoScale2d'],
          toImageButtonOptions: { format: 'svg', filename: solute.id + '_' + file },
        });
        if (my !== token.current) return;
        await settle(Plotly, plot);
        if (!tiesRef.current) showTies(Plotly, plot, false);
        if (fig.frames && fig.frames.length) {
          await Plotly.addFrames(
            plot,
            fig.frames.map((f) => Object.assign({}, f, { data: ungroup(f.data) }))
          );
          const first = fig.layout.sliders?.[0]?.active || 0; // the step the figure was drawn at
          const want = memory.frame?.view === id ? memory.frame.k : openingStep(fig.layout);
          const k = Math.min(want, fig.frames.length - 1);
          if (k !== first && my === token.current) {
            await Plotly.animate(plot, [String(k)], {
              mode: 'immediate',
              frame: { duration: 0, redraw: true },
              transition: { duration: 0 },
            });
            await Plotly.relayout(plot, { 'sliders[0].active': k });
            if (!tiesRef.current) showTies(Plotly, plot, false);
          }
          plot.on('plotly_sliderchange', (e) => {
            memory.frame = { view: id, k: e.slider.active };
          });
        }
      } catch (e) {
        if (my === token.current) setStatus(e.message || String(e));
      }
    })();
    return undefined;
  }, [manifest, viewDef, solute, unit, logAxes]);

  // The plot goes with the tab.
  useEffect(
    () => () => {
      token.current++;
      const plot = plotRef.current;
      loadPlotly()
        .then((Plotly) => plot && Plotly.purge(plot))
        .catch(() => {});
    },
    []
  );

  const refit = useCallback(async () => {
    const plot = plotRef.current;
    const stage = stageRef.current;
    if (!plot || !plot.data || !stage) return;
    const Plotly = await loadPlotly();
    await Plotly.relayout(plot, fitted(viewRef.current, stage.clientWidth, stage.clientHeight)[1]);
    await settle(Plotly, plot);
  }, []);

  // The notes open or close: the figure takes the room they leave.
  const firstNotes = useRef(true);
  useEffect(() => {
    if (firstNotes.current) {
      firstNotes.current = false;
      return;
    }
    refit();
  }, [notes, refit]);

  // The window is resized: crossing the breakpoint puts the notes beside the
  // figure or tucks them away; either way the figure is fitted again.
  useEffect(() => {
    let timer;
    let was = wide();
    const onResize = () => {
      clearTimeout(timer);
      timer = setTimeout(() => {
        if (wide() !== was) {
          was = wide();
          setNotes(was);
        }
        refit();
      }, 120);
    };
    window.addEventListener('resize', onResize);
    return () => {
      clearTimeout(timer);
      window.removeEventListener('resize', onResize);
    };
  }, [refit]);

  const toggleTies = async () => {
    const next = !ties;
    setTies(next);
    tiesRef.current = next;
    const Plotly = await loadPlotly();
    showTies(Plotly, plotRef.current, next);
  };

  const goView = (id) => {
    if (id === view) return;
    rememberViewChoice(views.find((x) => x.id === id)?.group, id);
    setView(id);
  };
  const goGroup = (g) => {
    const vs = views.filter((v) => v.group === g);
    if (vs.length) goView(memory.lastOf[g] || vs[0].id);
  };

  const groupViews = views.filter((v) => v.group === viewDef?.group);
  const note = viewDef && solute ? captionNote(viewDef.id, unit, solute) : '';
  const caption = viewDef ? viewDef.caption + note : '';

  return (
    <div className="ci-atlas">
      <div className="ci-bar">
        <span className="ci-brand">water + {solute ? solute.name : 'NaCl'}</span>
        <label
          className="ci-pick"
          title="composition unit: wt % is what is computed; mol % counts a salt as formula units; molality is mol of solute per kg of water; vol % is the volume of the pure solute in the volume of the unmixed components at 25 °C"
        >
          in{' '}
          <select
            aria-label="Composition unit"
            value={unit}
            onChange={(e) => setUnit(e.target.value)}
          >
            {Object.entries(UNITS).map(([k, u]) => (
              <option key={k} value={k} disabled={k === 'vol' && !(solute && solute.rho)}>
                {u.name}
              </option>
            ))}
          </select>
        </label>
        <div className="ci-tabs" role="tablist" aria-label="Kind of view">
          {(manifest?.groups ?? []).map((g) => (
            <button
              key={g.id}
              type="button"
              role="tab"
              aria-selected={g.id === viewDef?.group}
              className={g.id === viewDef?.group ? 'on' : ''}
              title={g.title}
              onClick={() => goGroup(g.id)}
            >
              {g.short}
            </button>
          ))}
        </div>
        <button
          type="button"
          className={'ci-notes-toggle' + (notes ? ' on' : '')}
          aria-pressed={notes}
          onClick={() => setNotes(!notes)}
        >
          Notes
        </button>
      </div>
      <div className="ci-pills">
        <div className="ci-pillset" role="tablist" aria-label="View">
          {groupViews.map((v) => (
            <button
              key={v.id}
              type="button"
              role="tab"
              aria-selected={v.id === viewDef?.id}
              className={v.id === viewDef?.id ? 'on' : ''}
              onClick={() => goView(v.id)}
            >
              {shortTitle(v.title)}
            </button>
          ))}
        </div>
        {manifest && viewDef && (
          <ColorKey
            view={viewDef}
            colors={manifest.key}
            ties={ties}
            onTies={toggleTies}
            logAxes={logAxes}
            onLog={() => setLogAxes(!logAxes)}
          />
        )}
      </div>
      <div className="ci-body">
        <div className="ci-stage" ref={stageRef}>
          <div className="ci-plot" ref={plotRef} data-testid="atlas-plot" />
          {status && <div className="ci-status">{status}</div>}
        </div>
        <aside className={notes ? '' : 'closed'} aria-label="Notes">
          <div className="ci-crumb">
            {solute && group ? `water + ${solute.name}  ·  ${group.title}` : ''}
          </div>
          <h2>{viewDef?.title ?? ''}</h2>
          <p className="ci-caption" title={caption}>
            {caption}
          </p>
          <div className="ci-cards">
            <div className="ci-tabs">
              {[
                ['solute', 'Solute'],
                ['model', 'Model'],
                ['points', 'Invariant points'],
              ].map(([id, label]) => (
                <button
                  key={id}
                  type="button"
                  className={id === card ? 'on' : ''}
                  onClick={() => setCard(id)}
                >
                  {label}
                </button>
              ))}
            </div>
            {info && solute && (
              <Card card={card} info={info} solute={solute} unit={unit} manifest={manifest} />
            )}
          </div>
        </aside>
      </div>
    </div>
  );
}

// The color key lives in the page, not in the figure: it cannot be clipped
// and costs the figure no room.
function ColorKey({ view, colors, ties, onTies, logAxes, onLog }) {
  const flat = view.id.startsWith('d-');
  const surf = view.id.startsWith('s3-');
  if (!flat && !surf) return null;
  const cols = flat ? colors.fill : colors.surf;
  return (
    <div className="ci-key" aria-label="Color key">
      {['one phase', 'two phases', 'three phases'].map((name, n) => (
        <span key={name}>
          <i style={{ background: cols[n] }} />
          {name}
        </span>
      ))}
      {flat && (
        <span>
          <i style={{ background: colors.outside, borderColor: '#b9b8b0' }} />
          outside the model
        </span>
      )}
      {flat && hasTies(view.id) && (
        <button
          type="button"
          className={ties ? 'on' : ''}
          aria-pressed={ties}
          title="show or hide the tie-lines"
          onClick={onTies}
        >
          <i />
          tie-lines
        </button>
      )}
      {flat && view.log && (
        <button
          type="button"
          className={logAxes ? 'on' : ''}
          aria-pressed={logAxes}
          title="logarithmic axis: shows the vapor, five to eight decades from the condensed phases"
          onClick={onLog}
        >
          {view.log}
        </button>
      )}
    </div>
  );
}

const celsius = (T) => (T - 273.15).toFixed(2);

function Row({ k, v }) {
  if (v == null || v === '') return null;
  return (
    <tr>
      <th>{k}</th>
      <td>{v}</td>
    </tr>
  );
}

function Card({ card, info: s, solute, unit, manifest }) {
  const P = s.properties || {};
  if (card === 'solute')
    return (
      <>
        <table>
          <tbody>
            <Row k="name" v={s.name} />
            <Row k="molar mass" v={s.molar_mass.toFixed(2) + ' g/mol'} />
            <Row k="pure substance" v={P.phase} />
            <Row k="density, 25 °C" v={P.density == null ? null : P.density + ' g/cm³'} />
            <Row k="melting point" v={P.Tm_C == null ? null : P.Tm_C + ' °C'} />
            <Row k="glass transition" v={P.Tg_C == null ? null : P.Tg_C + ' °C'} />
            <Row
              k="enthalpy of fusion"
              v={P.dHfus_kJ_mol == null ? null : P.dHfus_kJ_mol + ' kJ/mol'}
            />
            <Row k="permittivity" v={P.dielectric} />
            <Row k="CAS" v={P.cas} />
          </tbody>
        </table>
        {s.properties && <div className="ci-src">cryo-toolbox molecule database</div>}
      </>
    );
  if (card === 'model') {
    const sym = UNITS[uFor(solute, unit)].sym;
    const pMin = s.p_min > 0 ? (1e6 * s.p_min).toFixed(0) + ' Pa' : '0';
    return (
      <table>
        <tbody>
          <Row
            k="liquid water"
            v={s.liquid + (s.liquid === 'holten2014' ? ', T > ' + celsius(s.T_min) + ' °C' : '')}
          />
          <Row k="ices" v={s.ices.join(', ') + ' (SeaFreeze)'} />
          {(s.hydrates || []).map((h) => (
            <Row
              key={h.name}
              k="hydrate"
              v={`${h.name} (${h.formula}, ${(100 * h.W).toFixed(1)} wt %): ${h.note}`}
            />
          ))}
          <Row
            k="vapor"
            v={
              s.vapor
                ? 'pure water, from waterEoS (IAPWS-95 ideal gas with second and third virial terms); the solute is taken as non-volatile'
                : null
            }
          />
          <Row k="solution" v={'osmotic virial, ' + s.model} />
          <Row
            k="k, B, C"
            v={
              s.coefficients.k.toFixed(3) +
              ', ' +
              s.coefficients.B.toFixed(4) +
              ' kg/mol, ' +
              s.coefficients.C.toExponential(2) +
              ' kg²/mol²'
            }
          />
          <Row
            k="range"
            v={`${sym} ≤ ${comp(100 * s.W_cap, solute, unit)} (fit to ${s.coefficients.m_fit.toFixed(1)} mol/kg), ${pMin} – ${s.p_max} MPa`}
          />
          <Row k="solute volume" v={s.volume.V0.toFixed(1) + ' cm³/mol'} />
          <Row k="" v={<span className="ci-src">{s.volume.source}</span>} />
          <Row
            k="atlas"
            v={`cryo-isochoric, ${manifest.quality} mesh, release ${manifest.release_id}`}
          />
        </tbody>
      </table>
    );
  }
  return (
    <table>
      <tbody>
        <tr>
          <th>phases</th>
          <th>T (°C)</th>
          <th>p</th>
          <th>liquid</th>
        </tr>
        {s.points.map((q, i) => (
          <tr key={i}>
            <td>{q.phases.map((n) => (n === 'vap' ? 'vapor' : n)).join(' + ')}</td>
            <td>{celsius(q.T)}</td>
            <td>{q.p < 0.05 ? (1e6 * q.p).toFixed(0) + ' Pa' : q.p.toFixed(1) + ' MPa'}</td>
            <td>{q.W == null ? 'out of range' : comp(100 * q.W, solute, unit)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
