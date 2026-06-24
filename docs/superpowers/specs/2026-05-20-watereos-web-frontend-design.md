# waterEoS Web Front-End (sub-project B) — Design

**Date:** 2026-05-20
**Status:** Approved (design); pending spec review → implementation plan
**Part of:** waterEoS visualizer re-architecture (React front-end + Python API). Sub-project A (`watereos_api`) is merged on `master`. Sub-project C (deployment + retire Dash) is a separate spec→plan→build cycle.

## 1. Summary

Build `watereos-web/`, a Vite + React 18 single-page app that delivers the
prototype design (`watereos-prototype-src/`) wired to the live API
(`watereos_api`). All plots use `react-plotly.js` consuming the figure JSON
the API already returns. SWR handles fetching/caching/revalidation. Theme
and unit preferences persist in localStorage and thread automatically into
every API figure call so chart theming and units stay in sync with the
chrome.

## 2. Goals / Non-goals

**Goals**
- A self-contained `watereos-web/` Vite project depending only on the API
  contract (no Python interop, no Dash).
- Feature parity with the prototype's 7 screens: Info, Property Explorer,
  H₂O Phase Diagram, EoS Phase Diagram, Model Comparison, Point Calculator,
  Settings.
- All plots rendered via `react-plotly.js` from figure JSON returned by
  `watereos_api/figures/*` endpoints.
- All models, properties, units, and validity ranges sourced from
  `/api/metadata`.
- Theme (`dark`/`light`) and unit prefs persisted to localStorage; passed
  to the API on every figure request.
- Minimal smoke-test suite via Vitest + React Testing Library.
- Dev workflow: Vite dev server on :5173 proxies `/api` to FastAPI on :8000.

**Non-goals (sub-project C)**
- Production build pipeline + Render deployment.
- Removing `watereos_visualizer/` (Dash) and the `units.py` back-compat
  shim.
- Deleting `watereos-prototype-src/`.
- Browser e2e tests (Playwright).
- OpenAPI-driven type generation (we stay on JavaScript for v1).

## 3. Locked decisions

| # | Decision |
|---|----------|
| D1 | **JavaScript**, not TypeScript. Lower migration friction; small single-developer app; API contracts captured via JSDoc on the fetch client. |
| D2 | **`watereos-web/` at repo root**, sibling to `watereos_api/`. Kebab-case follows JS conventions. |
| D3 | **Fresh Vite scaffold + port files** — `npm create vite@latest watereos-web -- --template react`, then copy the prototype's `.jsx`/`.css` and refactor. The prototype dir stays untracked as reference. |
| D4 | **SWR** for data fetching/caching. `keepPreviousData: true`, `revalidateOnFocus: false`. |
| D5 | **`react-plotly.js`** for all plots (dynamically imported in `PlotCard` so Plotly's bundle stays out of the first paint). |
| D6 | **No `tweaks-panel`** and **no `__edit_mode_set_keys`** in production (prototype-only artifacts). The Settings screen replaces the runtime tweaks UI. |

## 4. Architecture

### 4.1 Project layout

```
watereos-web/
├── package.json
├── vite.config.js            # /api proxy → http://localhost:8000
├── index.html                # Vite entry
├── eslint.config.js
├── .prettierrc
├── .gitignore                # node_modules, dist, .vite
├── public/                   # favicon, og image (TBD by user; placeholder OK)
├── src/
│   ├── main.jsx              # React root mount
│   ├── App.jsx               # nav + screen routing (ported from prototype/app.jsx)
│   ├── tokens.css            # ported verbatim from prototype/tokens.css
│   ├── api/
│   │   ├── client.js         # one async fn per endpoint
│   │   └── hooks.js          # SWR wrappers
│   ├── theme/
│   │   ├── ThemeContext.jsx  # React context + useTheme hook
│   │   └── theme.css         # any theme-toggle specific styles
│   ├── settings/
│   │   ├── SettingsContext.jsx
│   │   └── useSettings.js
│   ├── components/
│   │   ├── TopBar.jsx
│   │   ├── Sidebar.jsx
│   │   ├── PlotCard.jsx      # wraps react-plotly.js
│   │   ├── Field.jsx
│   │   ├── Stepper.jsx
│   │   ├── Segmented.jsx
│   │   ├── Checkbox.jsx
│   │   └── CmdPalette.jsx
│   └── screens/
│       ├── Info.jsx
│       ├── PropertyExplorer.jsx
│       ├── H2OPhaseDiagram.jsx
│       ├── EoSPhaseDiagram.jsx
│       ├── ModelComparison.jsx
│       ├── PointCalculator.jsx
│       └── Settings.jsx
└── tests/
    ├── setup.js              # JSDOM + RTL setup
    ├── api-client.test.js
    ├── theme.test.js
    ├── settings.test.js
    └── screens.smoke.test.js
```

### 4.2 Dependencies (`package.json`)

Runtime: `react`, `react-dom`, `swr`, `react-plotly.js`, `plotly.js-dist-min`.
Dev: `vite`, `@vitejs/plugin-react`, `eslint`, `@eslint/js`,
`eslint-plugin-react`, `eslint-plugin-react-hooks`,
`eslint-plugin-react-refresh`, `prettier`, `vitest`,
`@testing-library/react`, `@testing-library/jest-dom`, `jsdom`,
`@vitest/coverage-v8` (optional).

Node version: lock with `"engines": { "node": ">=20" }`.

### 4.3 Module boundaries

- `api/client.js` is the only place `fetch` is called. It knows endpoint
  paths and request body shapes; it knows nothing about React or SWR.
- `api/hooks.js` is the only place `useSWR` appears. Each hook composes a
  `client` function with the current `theme` and `units` from context.
- `theme/`, `settings/` own their own localStorage keys and React context;
  no other module reads localStorage directly.
- `components/` are presentational — they receive props, never call the
  API directly. `PlotCard` is the only component allowed to import Plotly.
- `screens/` orchestrate one tab each: pull data via hooks, manage local
  control state, render components.

## 5. API client + SWR data layer

### 5.1 `src/api/client.js`

```js
const BASE = import.meta.env.VITE_API_BASE_URL ?? '';

async function _post(path, body) {
  const r = await fetch(`${BASE}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!r.ok) throw await _httpError(r);
  return r.json();
}
async function _get(path) {
  const r = await fetch(`${BASE}${path}`);
  if (!r.ok) throw await _httpError(r);
  return r.json();
}
async function _httpError(r) {
  let detail = `${r.status} ${r.statusText}`;
  try { detail = (await r.json()).detail ?? detail; } catch {}
  return Object.assign(new Error(detail), { status: r.status });
}

export const fetchHealth   = ()          => _get ('/api/health');
export const fetchMetadata = ()          => _get ('/api/metadata');
export const fetchPoint    = (b)         => _post('/api/point', b);
export const fetchCurves   = (b)         => _post('/api/figures/curves', b);
export const fetchSurface2 = (b)         => _post('/api/figures/surface2d', b);
export const fetchSurface3 = (b)         => _post('/api/figures/surface3d', b);
export const fetchCompare  = (b)         => _post('/api/figures/compare', b);
export const fetchEosPhase = (b)         => _post('/api/figures/eos-phase-diagram', b);
export const fetchH2OPhase = (b)         => _post('/api/figures/h2o-phase-diagram', b);
```

### 5.2 `src/api/hooks.js`

```js
import useSWR from 'swr';
import * as client from './client';
import { useTheme }    from '../theme/ThemeContext';
import { useSettings } from '../settings/SettingsContext';

const SWR_OPTS = { keepPreviousData: true, revalidateOnFocus: false };

export const useMetadata = () =>
  useSWR('/api/metadata', client.fetchMetadata, SWR_OPTS);

export function useCurvesFigure(params, enabled = true) {
  const { theme } = useTheme();
  const { units } = useSettings();
  const body = enabled && params ? { ...params, theme, units } : null;
  return useSWR(body && ['/api/figures/curves', body],
                ([, b]) => client.fetchCurves(b), SWR_OPTS);
}
// analogous: useSurface2dFigure, useSurface3dFigure, useCompareFigure,
//            useEosPhaseFigure, useH2OPhaseFigure, usePoint
```

Conventions:
- Pass `null` as the SWR key to skip fetching until the caller provides
  inputs (e.g., user hasn't filled a required field).
- The cache key is the `[path, body]` tuple; identical requests dedupe.
- `keepPreviousData` ensures the previous figure stays visible while a new
  one loads (no flicker on input changes).

## 6. Charting

`src/components/PlotCard.jsx`:

```jsx
import { Suspense, lazy } from 'react';
const Plot = lazy(() => import('react-plotly.js'));

export function PlotCard({ figure, loading, error, title, subtitle }) {
  return (
    <div className="plot-card">
      <header className="plot-head">
        <div><h2 className="plot-title">{title}</h2>
             {subtitle && <p className="plot-subtitle">{subtitle}</p>}</div>
      </header>
      <div className="plot-body">
        {error    && <div className="plot-error">{String(error.message || error)}</div>}
        {loading  && <div className="plot-loading">Loading…</div>}
        {figure   && (
          <Suspense fallback={<div className="plot-loading">Loading chart…</div>}>
            <Plot data={figure.data} layout={figure.layout}
                  config={{ displaylogo: false, responsive: true }}
                  useResizeHandler style={{ width: '100%', height: '100%' }} />
          </Suspense>
        )}
      </div>
    </div>
  );
}
```

Plotly's ~3 MB bundle is dynamically imported the first time `PlotCard`
mounts. Initial paint shows the chrome + tokens without Plotly loaded.

## 7. Theme + settings

### 7.1 `src/theme/ThemeContext.jsx`

```jsx
import { createContext, useContext, useEffect, useState } from 'react';
const ThemeContext = createContext(null);
const KEY = 'watereos_theme';

export function ThemeProvider({ children }) {
  const [theme, setTheme] = useState(
    () => localStorage.getItem(KEY) ?? 'dark');
  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    localStorage.setItem(KEY, theme);
  }, [theme]);
  const toggle = () => setTheme(t => (t === 'dark' ? 'light' : 'dark'));
  return <ThemeContext.Provider value={{ theme, toggle }}>{children}</ThemeContext.Provider>;
}
export const useTheme = () => useContext(ThemeContext);
```

### 7.2 `src/settings/SettingsContext.jsx`

Same pattern with `KEY = 'watereos_settings'`. Defaults:

```js
const DEFAULTS = {
  unit_density: 'kg/m³', unit_volume: 'm³/kg', unit_energy: 'J/kg',
  unit_entropy: 'J/(kg·K)', unit_bulk_modulus: 'MPa', unit_viscosity: 'Pa·s',
  default_n_curves: 5, default_n_points: 200,
};
```

The Settings screen renders editable controls bound to this context.
Validity-range data and the option lists for each unit come from
`/api/metadata`.

## 8. Migration: port / replace / strip / wire

| Prototype file                | Disposition |
|------------------------------|-------------|
| `index.html`                 | Replaced by Vite's `index.html`. |
| `tokens.css`                 | Ported verbatim to `src/tokens.css`. |
| `app.jsx`                    | Ported to `src/App.jsx`, wrapped in `<ThemeProvider><SettingsProvider>`; tweaks references and `__edit_mode_set_keys` removed. |
| `tweaks-panel.jsx`           | **Deleted** — prototype-only artifact. Settings screen replaces it. |
| `chart.jsx`                  | Hardcoded `MODEL_DEFS`/`PROPERTY_DEFS`/`evalProperty` deleted. SVG `PlotChart` retired. The file's idea moves to `PlotCard` + `useMetadata` + figure hooks. |
| `screens-main.jsx`           | Split: `TopBar` → `components/TopBar.jsx`, `Stepper` → `components/Stepper.jsx`, `ExplorerScreen` → `screens/PropertyExplorer.jsx` (wired to `useCurvesFigure`/`useSurface2dFigure`/`useSurface3dFigure`). |
| `screens-info.jsx`           | Split: `InfoScreen` → `screens/Info.jsx` (data-driven from `useMetadata`'s `models` instead of the hardcoded models table). `H2OPhaseScreen` → `screens/H2OPhaseDiagram.jsx` (wired to `useH2OPhaseFigure`). |
| `screens-eos.jsx`            | Split: `EoSPhaseScreen` → `screens/EoSPhaseDiagram.jsx` (wired to `useEosPhaseFigure`). `CompareScreen` → `screens/ModelComparison.jsx` (wired to `useCompareFigure`). |
| `screens-point.jsx`          | Split: `PointScreen` → `screens/PointCalculator.jsx` (wired to `usePoint`). `SettingsScreen` → `screens/Settings.jsx` (wired to `SettingsContext`, with units options from `useMetadata`). `CmdPalette` → `components/CmdPalette.jsx` (groups populated from metadata, not hardcoded). |

After migration `watereos-prototype-src/` is no longer used by the app.
It stays untracked on master per the current state; sub-project C decides
whether to delete it.

## 9. Dev workflow

### 9.1 `vite.config.js`

```js
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
    },
  },
});
```

### 9.2 `package.json` scripts

```json
{
  "scripts": {
    "dev":     "vite",
    "build":   "vite build",
    "preview": "vite preview",
    "lint":    "eslint .",
    "format":  "prettier -w .",
    "test":    "vitest run",
    "test:watch": "vitest"
  }
}
```

### 9.3 Local two-process workflow

```
# terminal A — API
cd <repo-root>
python -m uvicorn watereos_api.app:app --port 8000 --reload

# terminal B — Vite
cd watereos-web
npm install   # first time
npm run dev
# open http://localhost:5173
```

CORS: the API's allow-list already contains `http://localhost:5173` and
`http://127.0.0.1:5173` (sub-project A, env-configurable via
`WATEREOS_API_ALLOWED_ORIGINS`).

## 10. Testing (v1 minimal)

`vitest` + `@testing-library/react` + `jsdom`. ~10–15 tests:

- `tests/api-client.test.js`: each `_post` builds the correct body and URL;
  `_httpError` extracts `detail` from a 4xx JSON body.
- `tests/theme.test.js`: `ThemeProvider` sets `data-theme`; `toggle` flips
  and persists to `localStorage`.
- `tests/settings.test.js`: `SettingsProvider` defaults are correct;
  updates persist; reload restores from localStorage.
- `tests/screens.smoke.test.js`: each `<Screen />` renders without crashing
  given mocked SWR data (use `jest.mock('swr', ...)` or pass a test-only
  context). Asserts the main affordances are present (e.g.,
  `PropertyExplorer` shows model + property dropdowns).
- `tests/setup.js`: jsdom env, `@testing-library/jest-dom` matchers,
  `localStorage` reset between tests, and a **global** Plotly mock so
  Plotly never loads in unit tests:
  ```js
  import { vi } from 'vitest';
  vi.mock('react-plotly.js', () => ({ default: () => null }));
  ```
  Screen smoke tests therefore render without Plotly's bundle; if a future
  test needs the real component, it can override the mock locally.

E2E with Playwright is out of scope (sub-project C).

## 11. Risks & mitigations

- **Plotly bundle size (~3 MB).** Mitigated by `lazy(() => import('react-plotly.js'))` so it loads only when a plot is first shown. Production build will benefit from `plotly.js-dist-min`. Acceptable for v1; sub-project C can revisit selective Plotly bundles if needed.
- **Component-prop drift between port and prototype.** Mitigated by porting JSX visually identical first, only stripping `evalProperty`/`MODEL_DEFS`/`PROPERTY_DEFS` after the screen renders correctly with mocked figure data; then wiring SWR hooks.
- **SWR + body POST keying.** SWR keys must be serializable; we use `[path, body]` where `body` is a plain object. Identical bodies dedupe via SWR's structural compare. If keying turns flaky, fall back to a stable `JSON.stringify(body)` as the key. Documented; not in v1's hot path.
- **Settings/metadata bootstrap order.** `useMetadata` may resolve after first paint. Screens that need it (every dropdown) show a skeleton/loading state from SWR until `data` arrives. `Info` works without it.
- **Theme persists locally only.** No server-side preference store. Acceptable for v1; users share theme only within their own browser.
- **CmdPalette keybindings on Windows/macOS.** Cmd vs Ctrl. Detect via `navigator.platform`; show `⌘K` on mac, `Ctrl+K` elsewhere; bind both handlers.

## 12. Rollback

All work is in worktree `.claude/worktrees/watereos-web-fe` on branch
`worktree-watereos-web-fe` off `master` (currently at `d9a184c`). Revert =
discard the branch and its commits; the API and existing repo are
untouched (the front-end lives entirely under `watereos-web/`).
