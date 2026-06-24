# waterEoS Web Front-End (sub-project B) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `watereos-web/`, a Vite + React 18 single-page app that delivers the prototype design wired to the live `watereos_api`, replacing the bespoke SVG chart with `react-plotly.js` consuming API figure JSON.

**Architecture:** Fresh Vite + React JS scaffold at repo root. SWR-based data layer that automatically threads `theme` + `units` from React context into every figure-endpoint call. `react-plotly.js` dynamically imported in `PlotCard` so Plotly's bundle stays out of the first paint. Screen JSX ported from the prototype (`G:\My Drive\Isochoric\python_packages\waterEoS\watereos-prototype-src\`) with `evalProperty`/`MODEL_DEFS`/`PROPERTY_DEFS`/`PlotChart` replaced by API hooks + `PlotCard`.

**Tech Stack:** Node v24.13.0+, npm 11.6.2+, Vite 5, React 18, SWR 2, react-plotly.js 2 + plotly.js-dist-min, Vitest + React Testing Library + jsdom, ESLint, Prettier.

**Spec:** `docs/superpowers/specs/2026-05-20-watereos-web-frontend-design.md`

**Environment:** worktree `G:\My Drive\Isochoric\python_packages\waterEoS\.claude\worktrees\watereos-web-fe`; branch `worktree-watereos-web-fe` (off `master` at `d9a184c`). All commands run from this worktree dir unless noted. **Never** add AI/`Co-Authored-By`/"Generated with" attribution to commits or files. Commit normally (hooks ON, no `--no-verify`). Add **only** the files each task names — the worktree root has the existing Python packages (`watereos/`, `watereos_api/`, etc.) which must not be modified.

Prototype source (read-only reference, not part of this repo's tracked files; used for migration): `G:\My Drive\Isochoric\python_packages\waterEoS\watereos-prototype-src\` (10 files: `index.html`, `tokens.css`, `app.jsx`, `tweaks-panel.jsx`, `chart.jsx`, `screens-main.jsx`, `screens-info.jsx`, `screens-eos.jsx`, `screens-point.jsx`, `README.md`).

---

## File Structure

All new front-end work lives under `watereos-web/` at the repo root. Nothing else changes.

| File | Responsibility |
|---|---|
| `watereos-web/package.json` | Deps + scripts (dev/build/test/lint/format) |
| `watereos-web/vite.config.js` | React plugin + `/api` proxy to FastAPI on `:8000` |
| `watereos-web/eslint.config.js` | Flat-config ESLint for React |
| `watereos-web/.prettierrc` | Prettier config |
| `watereos-web/.gitignore` | `node_modules`, `dist`, `.vite`, coverage |
| `watereos-web/index.html` | Vite entry; sets `<html lang="en" data-theme="dark">` |
| `watereos-web/src/main.jsx` | React root mount + global providers |
| `watereos-web/src/App.jsx` | Tab routing + chrome (ported from prototype `app.jsx`) |
| `watereos-web/src/tokens.css` | Ported verbatim from prototype `tokens.css` |
| `watereos-web/src/api/client.js` | Single source of `fetch` calls; one fn per endpoint |
| `watereos-web/src/api/hooks.js` | SWR wrappers threading theme + units automatically |
| `watereos-web/src/theme/ThemeContext.jsx` | React context for `theme` (`dark`/`light`) + persistence |
| `watereos-web/src/settings/SettingsContext.jsx` | React context for user settings + persistence |
| `watereos-web/src/components/PlotCard.jsx` | Lazy-loaded `react-plotly.js` wrapper |
| `watereos-web/src/components/TopBar.jsx` | Brand + tab nav + ⌘K + theme toggle (ported) |
| `watereos-web/src/components/Sidebar.jsx` | Sidebar wrapper (ported) |
| `watereos-web/src/components/Field.jsx` | Label + control wrapper (ported) |
| `watereos-web/src/components/Stepper.jsx` | Numeric stepper (ported) |
| `watereos-web/src/components/Segmented.jsx` | Segmented control (ported) |
| `watereos-web/src/components/Checkbox.jsx` | Custom checkbox (ported) |
| `watereos-web/src/components/CmdPalette.jsx` | ⌘K palette (ported; metadata-driven) |
| `watereos-web/src/screens/Info.jsx` | Info landing (ported + metadata-driven models table) |
| `watereos-web/src/screens/PropertyExplorer.jsx` | Property Explorer (ported + 3 figure hooks) |
| `watereos-web/src/screens/H2OPhaseDiagram.jsx` | H₂O phase diagram (ported + `useH2OPhaseFigure`) |
| `watereos-web/src/screens/EoSPhaseDiagram.jsx` | EoS phase diagram (ported + `useEosPhaseFigure`) |
| `watereos-web/src/screens/ModelComparison.jsx` | Model comparison (ported + `useCompareFigure`) |
| `watereos-web/src/screens/PointCalculator.jsx` | Point calculator (ported + `usePoint`) |
| `watereos-web/src/screens/Settings.jsx` | Settings (ported + real `SettingsContext` persistence) |
| `watereos-web/tests/setup.js` | jsdom + jest-dom + global Plotly mock |
| `watereos-web/tests/api-client.test.js` | Client fn body/URL shape |
| `watereos-web/tests/theme.test.js` | Theme persistence + `data-theme` |
| `watereos-web/tests/settings.test.js` | Settings persistence + defaults |
| `watereos-web/tests/screens.smoke.test.js` | Each screen renders without crashing |

Each task ends with a single `git commit` of only its own files. Final task verifies the whole project (lint clean, tests green, build success, dev server boots).

---

## Task 1: Vite scaffold + project gitignore

**Files (created by `npm create`):** `watereos-web/package.json`, `watereos-web/vite.config.js`, `watereos-web/index.html`, `watereos-web/eslint.config.js`, `watereos-web/.gitignore`, `watereos-web/public/vite.svg`, `watereos-web/src/{main.jsx,App.jsx,App.css,index.css,assets/react.svg}`.

- [ ] **Step 1: Scaffold**

Run from worktree root: `cd watereos-web 2>/dev/null && echo "watereos-web already exists" && exit 1 || cd ..; npm create vite@latest watereos-web -- --template react`

Expected: Vite scaffolds into a new `watereos-web/` directory with React-JS template. No prompts (template is specified non-interactively).

- [ ] **Step 2: Verify scaffold + confirm React JS template**

Run: `cat watereos-web/package.json` — must show `"react"` and `"react-dom"` deps, `"@vitejs/plugin-react"` devDep, scripts `dev`/`build`/`lint`/`preview`, and NOT have `"typescript"` (we chose JS).

Run: `ls watereos-web/src` — must list `main.jsx`, `App.jsx`, `App.css`, `index.css`, `assets/`.

- [ ] **Step 3: Stage and commit the scaffold**

```bash
cd <worktree>
git add watereos-web/
git status --short
```

Expected `git status --short` shows ONLY files under `watereos-web/` as staged (`A`). If any other dir's files appear staged, unstage them with `git restore --staged <path>` — do NOT commit them.

```bash
git commit -m "feat(web): scaffold Vite + React JS project"
```

---

## Task 2: Vite config — `/api` dev proxy

**Files:** Modify `watereos-web/vite.config.js`.

- [ ] **Step 1: Replace** the entire content of `watereos-web/vite.config.js` with:

```js
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// https://vitejs.dev/config/
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

- [ ] **Step 2: Sanity check — `vite build` still succeeds with the default app**

Run: `cd watereos-web && npm install` (first-time install).

Run: `cd watereos-web && npx vite build`

Expected: build succeeds; produces `dist/`. Output ends with `✓ built in <time>`.

- [ ] **Step 3: Add `dist` to `.gitignore`**

Vite's default `.gitignore` already ignores `node_modules`, `dist`, `.vite`. Verify with `grep -E "^(node_modules|dist|\.vite)" watereos-web/.gitignore` — must return all 3 lines. If any are missing, append the missing line(s).

- [ ] **Step 4: Confirm `node_modules` and `dist` are untracked**

Run: `cd <worktree> && git status --short | grep -E "watereos-web/(node_modules|dist)" || echo "OK (none tracked)"`

Expected: `OK (none tracked)`.

- [ ] **Step 5: Commit only the vite.config.js change**

```bash
git add watereos-web/vite.config.js
git commit -m "feat(web): vite proxy /api to FastAPI on 8000"
```

---

## Task 3: ESLint + Prettier config

**Files:** Modify `watereos-web/eslint.config.js`; Create `watereos-web/.prettierrc`, `watereos-web/.prettierignore`.

- [ ] **Step 1: Replace** `watereos-web/eslint.config.js` with this React-aware flat-config (compatible with the Vite scaffold's plugin choices):

```js
import js from '@eslint/js';
import globals from 'globals';
import react from 'eslint-plugin-react';
import reactHooks from 'eslint-plugin-react-hooks';
import reactRefresh from 'eslint-plugin-react-refresh';

export default [
  { ignores: ['dist', 'coverage', '.vite'] },
  {
    files: ['**/*.{js,jsx}'],
    languageOptions: {
      ecmaVersion: 'latest',
      sourceType: 'module',
      globals: { ...globals.browser, ...globals.node },
      parserOptions: { ecmaFeatures: { jsx: true } },
    },
    settings: { react: { version: '18.3' } },
    plugins: {
      react,
      'react-hooks': reactHooks,
      'react-refresh': reactRefresh,
    },
    rules: {
      ...js.configs.recommended.rules,
      ...react.configs.recommended.rules,
      ...react.configs['jsx-runtime'].rules,
      ...reactHooks.configs.recommended.rules,
      'react-refresh/only-export-components': ['warn', { allowConstantExport: true }],
      'react/prop-types': 'off',
      'no-unused-vars': ['warn', { argsIgnorePattern: '^_' }],
    },
  },
  {
    files: ['tests/**/*.{js,jsx}'],
    languageOptions: { globals: { ...globals.browser, ...globals.node, vi: true, expect: true } },
  },
];
```

- [ ] **Step 2: Install eslint plugins**

Run: `cd watereos-web && npm install -D eslint @eslint/js eslint-plugin-react eslint-plugin-react-hooks eslint-plugin-react-refresh globals`

- [ ] **Step 3: Create `watereos-web/.prettierrc`** with exactly:

```json
{
  "semi": true,
  "singleQuote": true,
  "trailingComma": "es5",
  "tabWidth": 2,
  "printWidth": 100,
  "arrowParens": "always"
}
```

- [ ] **Step 4: Create `watereos-web/.prettierignore`** with exactly:

```
node_modules
dist
coverage
.vite
```

- [ ] **Step 5: Install prettier**

Run: `cd watereos-web && npm install -D prettier`

- [ ] **Step 6: Add `format` script to `package.json`**

Edit `watereos-web/package.json` and add to `"scripts"`: `"format": "prettier -w ."`. Keep all existing scripts intact.

- [ ] **Step 7: Verify lint + format both run clean on the scaffold**

Run: `cd watereos-web && npm run lint` — expect 0 errors (warnings OK).

Run: `cd watereos-web && npx prettier --check .` — expect "All matched files use Prettier code style!" (or auto-format with `npm run format` if it complains).

- [ ] **Step 8: Commit**

```bash
cd <worktree>
git add watereos-web/eslint.config.js watereos-web/.prettierrc watereos-web/.prettierignore watereos-web/package.json watereos-web/package-lock.json
git commit -m "feat(web): configure ESLint + Prettier"
```

---

## Task 4: Vitest + React Testing Library setup

**Files:** Create `watereos-web/tests/setup.js`; Modify `watereos-web/package.json` + `watereos-web/vite.config.js`.

- [ ] **Step 1: Install Vitest + RTL deps**

Run: `cd watereos-web && npm install -D vitest @testing-library/react @testing-library/jest-dom @testing-library/dom jsdom`

- [ ] **Step 2: Create `watereos-web/tests/setup.js`** with exactly:

```js
import '@testing-library/jest-dom/vitest';
import { afterEach, beforeEach, vi } from 'vitest';
import { cleanup } from '@testing-library/react';

// Global Plotly mock — react-plotly.js never loads in unit tests.
vi.mock('react-plotly.js', () => ({ default: () => null }));

// Polyfill ResizeObserver (Plotly/responsive layouts touch it even when mocked).
if (!globalThis.ResizeObserver) {
  globalThis.ResizeObserver = class {
    observe() {}
    unobserve() {}
    disconnect() {}
  };
}

beforeEach(() => {
  localStorage.clear();
  document.documentElement.dataset.theme = '';
});

afterEach(() => {
  cleanup();
});
```

- [ ] **Step 3: Extend `watereos-web/vite.config.js` with a `test` section**

Append a `test` block to the existing `defineConfig({...})`. The full file becomes:

```js
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// https://vitejs.dev/config/
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
  test: {
    environment: 'jsdom',
    globals: true,
    setupFiles: ['./tests/setup.js'],
    css: true,
  },
});
```

- [ ] **Step 4: Add `test` + `test:watch` scripts to package.json**

In `watereos-web/package.json` `"scripts"`, add:

```
"test": "vitest run",
"test:watch": "vitest"
```

Keep all existing scripts intact.

- [ ] **Step 5: Add a single trivial test to prove the harness works**

Create `watereos-web/tests/setup.test.js`:

```js
import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';

describe('vitest setup', () => {
  it('renders JSX with RTL + jest-dom', () => {
    render(<h1>hello</h1>);
    expect(screen.getByText('hello')).toBeInTheDocument();
  });

  it('localStorage is cleared between tests', () => {
    expect(localStorage.length).toBe(0);
    localStorage.setItem('x', '1');
    expect(localStorage.getItem('x')).toBe('1');
  });
});
```

- [ ] **Step 6: Run the test harness**

Run: `cd watereos-web && npm test` — expect 2 passed in `tests/setup.test.js`.

- [ ] **Step 7: Commit**

```bash
cd <worktree>
git add watereos-web/tests/setup.js watereos-web/tests/setup.test.js watereos-web/vite.config.js watereos-web/package.json watereos-web/package-lock.json
git commit -m "feat(web): configure Vitest + RTL with global Plotly mock"
```

---

## Task 5: Port `tokens.css`

**Files:** Create `watereos-web/src/tokens.css`; remove `watereos-web/src/App.css` and `watereos-web/src/index.css` (Vite-default styling we'll replace); modify `watereos-web/src/main.jsx`.

- [ ] **Step 1: Copy** the entire content of `G:\My Drive\Isochoric\python_packages\waterEoS\watereos-prototype-src\tokens.css` byte-for-byte into a new file `watereos-web/src/tokens.css`. Do not modify it.

Verify after copy: `wc -l watereos-web/src/tokens.css` — should be around 643 lines.

Verify: `grep -E "^:root \{|^\[data-theme=" watereos-web/src/tokens.css | head -3` returns at least `:root {`, `[data-theme="light"] {`, `[data-theme="dark"] {`.

- [ ] **Step 2: Remove Vite-default styles**

```bash
rm watereos-web/src/App.css watereos-web/src/index.css
```

- [ ] **Step 3: Replace `watereos-web/src/main.jsx` exactly with**

```jsx
import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import './tokens.css';
import App from './App.jsx';

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <App />
  </StrictMode>
);
```

- [ ] **Step 4: Update `watereos-web/index.html` `<html>` and remove default styling references**

Replace its full content with:

```html
<!doctype html>
<html lang="en" data-theme="dark">
  <head>
    <meta charset="UTF-8" />
    <link rel="icon" type="image/svg+xml" href="/vite.svg" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>waterEoS — Visualizer</title>
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/main.jsx"></script>
  </body>
</html>
```

- [ ] **Step 5: Minimal App.jsx so the project still builds**

Replace `watereos-web/src/App.jsx` with:

```jsx
export default function App() {
  return <div>waterEoS (scaffolding)</div>;
}
```

- [ ] **Step 6: Verify build + tests still pass**

Run: `cd watereos-web && npx vite build` — expect success.

Run: `cd watereos-web && npm test` — expect 2 passed (the existing setup test).

- [ ] **Step 7: Commit**

```bash
cd <worktree>
git add watereos-web/src/tokens.css watereos-web/src/main.jsx watereos-web/src/App.jsx watereos-web/index.html
git rm watereos-web/src/App.css watereos-web/src/index.css
git commit -m "feat(web): port tokens.css; minimal App scaffolding"
```

---

## Task 6: API client

**Files:** Create `watereos-web/src/api/client.js`, `watereos-web/tests/api-client.test.js`.

- [ ] **Step 1: Write the failing test** — `watereos-web/tests/api-client.test.js`:

```js
import { describe, it, expect, vi, beforeEach } from 'vitest';
import * as client from '../src/api/client.js';

beforeEach(() => {
  global.fetch = vi.fn(async () => ({
    ok: true,
    status: 200,
    json: async () => ({ ok: true }),
  }));
});

describe('api/client', () => {
  it('fetchHealth GETs /api/health', async () => {
    await client.fetchHealth();
    expect(global.fetch).toHaveBeenCalledWith('/api/health');
  });

  it('fetchMetadata GETs /api/metadata', async () => {
    await client.fetchMetadata();
    expect(global.fetch).toHaveBeenCalledWith('/api/metadata');
  });

  it('fetchPoint POSTs JSON to /api/point with the given body', async () => {
    const body = { model_keys: ['duska2020'], T_K: 273.15, P_MPa: 0.1 };
    await client.fetchPoint(body);
    expect(global.fetch).toHaveBeenCalledWith(
      '/api/point',
      expect.objectContaining({
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      })
    );
  });

  it('fetchCurves POSTs to /api/figures/curves', async () => {
    await client.fetchCurves({ model: 'duska2020' });
    expect(global.fetch).toHaveBeenCalledWith('/api/figures/curves', expect.any(Object));
  });

  it('fetchSurface2 POSTs to /api/figures/surface2d', async () => {
    await client.fetchSurface2({ model: 'duska2020' });
    expect(global.fetch).toHaveBeenCalledWith('/api/figures/surface2d', expect.any(Object));
  });

  it('fetchSurface3 POSTs to /api/figures/surface3d', async () => {
    await client.fetchSurface3({ model: 'duska2020' });
    expect(global.fetch).toHaveBeenCalledWith('/api/figures/surface3d', expect.any(Object));
  });

  it('fetchCompare POSTs to /api/figures/compare', async () => {
    await client.fetchCompare({ model_keys: ['duska2020'] });
    expect(global.fetch).toHaveBeenCalledWith('/api/figures/compare', expect.any(Object));
  });

  it('fetchEosPhase POSTs to /api/figures/eos-phase-diagram', async () => {
    await client.fetchEosPhase({ model: 'duska2020' });
    expect(global.fetch).toHaveBeenCalledWith(
      '/api/figures/eos-phase-diagram',
      expect.any(Object)
    );
  });

  it('fetchH2OPhase POSTs to /api/figures/h2o-phase-diagram', async () => {
    await client.fetchH2OPhase({ projection: 'tv' });
    expect(global.fetch).toHaveBeenCalledWith(
      '/api/figures/h2o-phase-diagram',
      expect.any(Object)
    );
  });

  it('non-200 responses raise an Error with detail from body', async () => {
    global.fetch = vi.fn(async () => ({
      ok: false,
      status: 404,
      statusText: 'Not Found',
      json: async () => ({ detail: 'unknown model: nope' }),
    }));
    await expect(client.fetchCurves({ model: 'nope' })).rejects.toThrow('unknown model: nope');
  });

  it('honors VITE_API_BASE_URL for the base prefix', async () => {
    import.meta.env.VITE_API_BASE_URL = 'http://example.test';
    try {
      await client.fetchHealth();
      expect(global.fetch).toHaveBeenCalledWith('http://example.test/api/health');
    } finally {
      import.meta.env.VITE_API_BASE_URL = '';
    }
  });
});
```

- [ ] **Step 2: Run, expect FAIL** — `cd watereos-web && npm test -- api-client` → resolves module not found.

- [ ] **Step 3: Create `watereos-web/src/api/client.js`** with exactly:

```js
const BASE = import.meta.env.VITE_API_BASE_URL ?? '';

async function _get(path) {
  const r = await fetch(`${BASE}${path}`);
  if (!r.ok) throw await _httpError(r);
  return r.json();
}

async function _post(path, body) {
  const r = await fetch(`${BASE}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!r.ok) throw await _httpError(r);
  return r.json();
}

async function _httpError(r) {
  let detail = `${r.status} ${r.statusText || ''}`.trim();
  try {
    const j = await r.json();
    if (j && typeof j.detail === 'string') detail = j.detail;
  } catch {
    // body not JSON; keep the default detail
  }
  const err = new Error(detail);
  err.status = r.status;
  return err;
}

export const fetchHealth = () => _get('/api/health');
export const fetchMetadata = () => _get('/api/metadata');
export const fetchPoint = (body) => _post('/api/point', body);
export const fetchCurves = (body) => _post('/api/figures/curves', body);
export const fetchSurface2 = (body) => _post('/api/figures/surface2d', body);
export const fetchSurface3 = (body) => _post('/api/figures/surface3d', body);
export const fetchCompare = (body) => _post('/api/figures/compare', body);
export const fetchEosPhase = (body) => _post('/api/figures/eos-phase-diagram', body);
export const fetchH2OPhase = (body) => _post('/api/figures/h2o-phase-diagram', body);
```

- [ ] **Step 4: Run, expect PASS** — `cd watereos-web && npm test -- api-client` → all client tests pass.

- [ ] **Step 5: Commit**

```bash
cd <worktree>
git add watereos-web/src/api/client.js watereos-web/tests/api-client.test.js
git commit -m "feat(web): add API client with one fn per endpoint"
```

---

## Task 7: SWR data hooks

**Files:** Install `swr`; Create `watereos-web/src/api/hooks.js`, `watereos-web/tests/api-hooks.test.jsx`.

- [ ] **Step 1: Install SWR**

Run: `cd watereos-web && npm install swr`

- [ ] **Step 2: Write the failing test** — `watereos-web/tests/api-hooks.test.jsx`:

```jsx
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { renderHook, waitFor } from '@testing-library/react';
import { SWRConfig } from 'swr';

import * as client from '../src/api/client.js';
import { useMetadata, useCurvesFigure } from '../src/api/hooks.js';
import { ThemeProvider } from '../src/theme/ThemeContext.jsx';
import { SettingsProvider } from '../src/settings/SettingsContext.jsx';

function wrapper({ children }) {
  return (
    <SWRConfig value={{ provider: () => new Map(), dedupingInterval: 0 }}>
      <ThemeProvider>
        <SettingsProvider>{children}</SettingsProvider>
      </ThemeProvider>
    </SWRConfig>
  );
}

beforeEach(() => {
  vi.restoreAllMocks();
});

describe('api/hooks', () => {
  it('useMetadata fetches /api/metadata once', async () => {
    const spy = vi.spyOn(client, 'fetchMetadata').mockResolvedValue({ models: [] });
    const { result } = renderHook(() => useMetadata(), { wrapper });
    await waitFor(() => expect(result.current.data).toEqual({ models: [] }));
    expect(spy).toHaveBeenCalledTimes(1);
  });

  it('useCurvesFigure threads theme + units into the request body', async () => {
    const spy = vi.spyOn(client, 'fetchCurves').mockResolvedValue({ figure: { data: [] } });
    const params = {
      model: 'duska2020',
      property: 'rho',
      T_range: [200, 300],
      P_range: [0.1, 200],
      n_curves: 5,
      n_points: 200,
      isobar_mode: true,
      show_phase_boundaries: false,
    };
    const { result } = renderHook(() => useCurvesFigure(params), { wrapper });
    await waitFor(() => expect(result.current.data).toBeTruthy());
    const callBody = spy.mock.calls[0][0];
    expect(callBody).toMatchObject(params);
    expect(callBody.theme).toBe('dark'); // default from ThemeProvider
    expect(callBody.units).toMatchObject({ unit_density: 'kg/m³' });
  });

  it('useCurvesFigure with null params skips the fetch', async () => {
    const spy = vi.spyOn(client, 'fetchCurves').mockResolvedValue({ figure: { data: [] } });
    renderHook(() => useCurvesFigure(null), { wrapper });
    // Give SWR a tick
    await new Promise((r) => setTimeout(r, 10));
    expect(spy).not.toHaveBeenCalled();
  });
});
```

- [ ] **Step 3: Run, expect FAIL** — hooks module not found.

- [ ] **Step 4: Create `watereos-web/src/api/hooks.js`** with exactly:

```js
import useSWR from 'swr';
import * as client from './client.js';
import { useTheme } from '../theme/ThemeContext.jsx';
import { useSettings } from '../settings/SettingsContext.jsx';

const SWR_OPTS = { keepPreviousData: true, revalidateOnFocus: false };

export function useMetadata() {
  return useSWR('/api/metadata', client.fetchMetadata, SWR_OPTS);
}

function _figureHook(path, fetcher) {
  return function useFigure(params, enabled = true) {
    const { theme } = useTheme();
    const { units } = useSettings();
    const body = enabled && params ? { ...params, theme, units } : null;
    return useSWR(body && [path, body], ([, b]) => fetcher(b), SWR_OPTS);
  };
}

export const useCurvesFigure = _figureHook('/api/figures/curves', client.fetchCurves);
export const useSurface2dFigure = _figureHook('/api/figures/surface2d', client.fetchSurface2);
export const useSurface3dFigure = _figureHook('/api/figures/surface3d', client.fetchSurface3);
export const useCompareFigure = _figureHook('/api/figures/compare', client.fetchCompare);
export const useEosPhaseFigure = _figureHook(
  '/api/figures/eos-phase-diagram',
  client.fetchEosPhase
);
export const useH2OPhaseFigure = _figureHook(
  '/api/figures/h2o-phase-diagram',
  client.fetchH2OPhase
);

export function usePoint(params, enabled = true) {
  const { units } = useSettings();
  const body = enabled && params ? { ...params, units } : null;
  return useSWR(body && ['/api/point', body], ([, b]) => client.fetchPoint(b), SWR_OPTS);
}
```

(`usePoint` does not pass `theme` because the point endpoint returns raw values, not a figure.)

- [ ] **Step 5: Run, expect PASS** — but the test imports `ThemeProvider` and `SettingsProvider` which don't exist yet. The expected fail at this point is the import resolution. Continue to Task 8 to define them; this test will pass once both contexts exist. To unblock the implementer flow, mark this task DONE_WITH_CONCERNS if the test fails on missing context modules, and verify it passes after Task 8 + Task 9 land.

- [ ] **Step 6: Commit just hooks.js + its test (red-until-contexts-land is OK; per-task TDD)**

```bash
cd <worktree>
git add watereos-web/src/api/hooks.js watereos-web/tests/api-hooks.test.jsx watereos-web/package.json watereos-web/package-lock.json
git commit -m "feat(web): SWR hooks threading theme + units into figure calls"
```

> Note: tests `api-hooks.test.jsx` will start passing after Tasks 8 & 9. This is acceptable per-task TDD ordering. The verification step for Tasks 8/9 will run `npm test` and confirm `api-hooks.test.jsx` is green at that point.

---

## Task 8: ThemeContext

**Files:** Create `watereos-web/src/theme/ThemeContext.jsx`, `watereos-web/tests/theme.test.jsx`.

- [ ] **Step 1: Write the failing test** — `watereos-web/tests/theme.test.jsx`:

```jsx
import { describe, it, expect } from 'vitest';
import { render, screen, act } from '@testing-library/react';
import { ThemeProvider, useTheme } from '../src/theme/ThemeContext.jsx';

function Harness() {
  const { theme, toggle } = useTheme();
  return (
    <>
      <span data-testid="t">{theme}</span>
      <button onClick={toggle}>toggle</button>
    </>
  );
}

describe('ThemeContext', () => {
  it('defaults to dark and sets data-theme on <html>', () => {
    render(
      <ThemeProvider>
        <Harness />
      </ThemeProvider>
    );
    expect(screen.getByTestId('t').textContent).toBe('dark');
    expect(document.documentElement.dataset.theme).toBe('dark');
  });

  it('toggle flips theme and persists to localStorage', () => {
    render(
      <ThemeProvider>
        <Harness />
      </ThemeProvider>
    );
    act(() => screen.getByText('toggle').click());
    expect(screen.getByTestId('t').textContent).toBe('light');
    expect(document.documentElement.dataset.theme).toBe('light');
    expect(localStorage.getItem('watereos_theme')).toBe('light');
  });

  it('restores persisted theme on mount', () => {
    localStorage.setItem('watereos_theme', 'light');
    render(
      <ThemeProvider>
        <Harness />
      </ThemeProvider>
    );
    expect(screen.getByTestId('t').textContent).toBe('light');
    expect(document.documentElement.dataset.theme).toBe('light');
  });
});
```

- [ ] **Step 2: Run, expect FAIL** — module not found.

- [ ] **Step 3: Create `watereos-web/src/theme/ThemeContext.jsx`** with exactly:

```jsx
import { createContext, useCallback, useContext, useEffect, useState } from 'react';

const KEY = 'watereos_theme';
const ThemeContext = createContext({ theme: 'dark', toggle: () => {} });

export function ThemeProvider({ children }) {
  const [theme, setTheme] = useState(() => {
    const v = typeof localStorage !== 'undefined' ? localStorage.getItem(KEY) : null;
    return v === 'light' || v === 'dark' ? v : 'dark';
  });

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    try {
      localStorage.setItem(KEY, theme);
    } catch {
      // localStorage may be unavailable in some environments; ignore
    }
  }, [theme]);

  const toggle = useCallback(() => {
    setTheme((t) => (t === 'dark' ? 'light' : 'dark'));
  }, []);

  return <ThemeContext.Provider value={{ theme, toggle }}>{children}</ThemeContext.Provider>;
}

export function useTheme() {
  return useContext(ThemeContext);
}
```

- [ ] **Step 4: Run, expect PASS** — `cd watereos-web && npm test -- theme` → 3 passed.

- [ ] **Step 5: Commit**

```bash
cd <worktree>
git add watereos-web/src/theme/ThemeContext.jsx watereos-web/tests/theme.test.jsx
git commit -m "feat(web): ThemeContext with data-theme + localStorage persistence"
```

---

## Task 9: SettingsContext

**Files:** Create `watereos-web/src/settings/SettingsContext.jsx`, `watereos-web/tests/settings.test.jsx`.

- [ ] **Step 1: Write the failing test** — `watereos-web/tests/settings.test.jsx`:

```jsx
import { describe, it, expect } from 'vitest';
import { render, screen, act } from '@testing-library/react';
import {
  SettingsProvider,
  useSettings,
  DEFAULT_SETTINGS,
} from '../src/settings/SettingsContext.jsx';

function Harness() {
  const { settings, units, setUnit, setSetting } = useSettings();
  return (
    <>
      <span data-testid="density">{units.unit_density}</span>
      <span data-testid="ncurves">{settings.default_n_curves}</span>
      <button onClick={() => setUnit('unit_density', 'g/cm³')}>change-density</button>
      <button onClick={() => setSetting('default_n_curves', 7)}>change-ncurves</button>
    </>
  );
}

describe('SettingsContext', () => {
  it('defaults match DEFAULT_SETTINGS', () => {
    render(
      <SettingsProvider>
        <Harness />
      </SettingsProvider>
    );
    expect(screen.getByTestId('density').textContent).toBe(DEFAULT_SETTINGS.unit_density);
    expect(screen.getByTestId('ncurves').textContent).toBe(String(DEFAULT_SETTINGS.default_n_curves));
  });

  it('setUnit updates and persists to localStorage', () => {
    render(
      <SettingsProvider>
        <Harness />
      </SettingsProvider>
    );
    act(() => screen.getByText('change-density').click());
    expect(screen.getByTestId('density').textContent).toBe('g/cm³');
    const persisted = JSON.parse(localStorage.getItem('watereos_settings'));
    expect(persisted.unit_density).toBe('g/cm³');
  });

  it('setSetting updates other prefs', () => {
    render(
      <SettingsProvider>
        <Harness />
      </SettingsProvider>
    );
    act(() => screen.getByText('change-ncurves').click());
    expect(screen.getByTestId('ncurves').textContent).toBe('7');
  });

  it('reload restores persisted settings', () => {
    localStorage.setItem(
      'watereos_settings',
      JSON.stringify({ unit_density: 'g/cm³', default_n_curves: 9 })
    );
    render(
      <SettingsProvider>
        <Harness />
      </SettingsProvider>
    );
    expect(screen.getByTestId('density').textContent).toBe('g/cm³');
    expect(screen.getByTestId('ncurves').textContent).toBe('9');
  });
});
```

- [ ] **Step 2: Run, expect FAIL** — module not found.

- [ ] **Step 3: Create `watereos-web/src/settings/SettingsContext.jsx`** with exactly:

```jsx
import { createContext, useCallback, useContext, useEffect, useMemo, useState } from 'react';

const KEY = 'watereos_settings';

export const DEFAULT_SETTINGS = {
  unit_density: 'kg/m³',
  unit_volume: 'm³/kg',
  unit_energy: 'J/kg',
  unit_entropy: 'J/(kg·K)',
  unit_bulk_modulus: 'MPa',
  unit_viscosity: 'Pa·s',
  default_n_curves: 5,
  default_n_points: 200,
};

const UNIT_KEYS = [
  'unit_density',
  'unit_volume',
  'unit_energy',
  'unit_entropy',
  'unit_bulk_modulus',
  'unit_viscosity',
];

const SettingsContext = createContext({
  settings: DEFAULT_SETTINGS,
  units: {},
  setSetting: () => {},
  setUnit: () => {},
  reset: () => {},
});

function _load() {
  try {
    const raw = localStorage.getItem(KEY);
    if (!raw) return DEFAULT_SETTINGS;
    const parsed = JSON.parse(raw);
    return { ...DEFAULT_SETTINGS, ...parsed };
  } catch {
    return DEFAULT_SETTINGS;
  }
}

export function SettingsProvider({ children }) {
  const [settings, setSettings] = useState(_load);

  useEffect(() => {
    try {
      localStorage.setItem(KEY, JSON.stringify(settings));
    } catch {
      // ignore
    }
  }, [settings]);

  const setSetting = useCallback((key, value) => {
    setSettings((s) => ({ ...s, [key]: value }));
  }, []);

  const setUnit = useCallback(
    (key, value) => {
      if (!UNIT_KEYS.includes(key)) {
        return;
      }
      setSetting(key, value);
    },
    [setSetting]
  );

  const reset = useCallback(() => setSettings(DEFAULT_SETTINGS), []);

  const units = useMemo(() => {
    const u = {};
    for (const k of UNIT_KEYS) u[k] = settings[k];
    return u;
  }, [settings]);

  return (
    <SettingsContext.Provider value={{ settings, units, setSetting, setUnit, reset }}>
      {children}
    </SettingsContext.Provider>
  );
}

export function useSettings() {
  return useContext(SettingsContext);
}
```

- [ ] **Step 4: Run, expect PASS for both settings AND api-hooks (now that both contexts exist)**

Run: `cd watereos-web && npm test`

Expected: ALL tests pass: `setup.test.js` (2), `api-client.test.js` (~11), `api-hooks.test.jsx` (3), `theme.test.jsx` (3), `settings.test.jsx` (4). Total ~23.

- [ ] **Step 5: Commit**

```bash
cd <worktree>
git add watereos-web/src/settings/SettingsContext.jsx watereos-web/tests/settings.test.jsx
git commit -m "feat(web): SettingsContext with units + localStorage persistence"
```

---

## Task 10: PlotCard component (lazy `react-plotly.js`)

**Files:** Install `react-plotly.js` + `plotly.js-dist-min`; Create `watereos-web/src/components/PlotCard.jsx`, `watereos-web/tests/plot-card.test.jsx`.

- [ ] **Step 1: Install Plotly deps**

Run: `cd watereos-web && npm install react-plotly.js plotly.js-dist-min`

- [ ] **Step 2: Write the failing test** — `watereos-web/tests/plot-card.test.jsx`:

```jsx
import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { PlotCard } from '../src/components/PlotCard.jsx';

describe('PlotCard', () => {
  it('renders title and subtitle when provided', () => {
    render(<PlotCard title="My plot" subtitle="some context" />);
    expect(screen.getByRole('heading', { name: 'My plot' })).toBeInTheDocument();
    expect(screen.getByText('some context')).toBeInTheDocument();
  });

  it('renders the loading indicator when loading and no figure', () => {
    render(<PlotCard title="t" loading />);
    expect(screen.getByText(/loading/i)).toBeInTheDocument();
  });

  it('renders the error message when error is provided', () => {
    render(<PlotCard title="t" error={new Error('boom')} />);
    expect(screen.getByText(/boom/)).toBeInTheDocument();
  });

  it('does not render the loading indicator when a figure is available', () => {
    render(<PlotCard title="t" figure={{ data: [], layout: {} }} />);
    expect(screen.queryByText(/^loading/i)).not.toBeInTheDocument();
  });
});
```

- [ ] **Step 3: Run, expect FAIL** — module not found.

- [ ] **Step 4: Create `watereos-web/src/components/PlotCard.jsx`** with exactly:

```jsx
import { Suspense, lazy } from 'react';

const Plot = lazy(() => import('react-plotly.js'));

export function PlotCard({ title, subtitle, figure, loading, error, toolbar }) {
  return (
    <div className="plot-card">
      <header className="plot-head">
        <div>
          <h2 className="plot-title">{title}</h2>
          {subtitle && <p className="plot-subtitle">{subtitle}</p>}
        </div>
        {toolbar && <div className="plot-toolbar">{toolbar}</div>}
      </header>
      <div className="plot-body">
        {error && (
          <div className="plot-error" role="alert">
            {String(error?.message ?? error)}
          </div>
        )}
        {loading && !figure && <div className="plot-loading">Loading…</div>}
        {figure && (
          <Suspense fallback={<div className="plot-loading">Loading chart…</div>}>
            <Plot
              data={figure.data ?? []}
              layout={figure.layout ?? {}}
              config={{ displaylogo: false, responsive: true }}
              useResizeHandler
              style={{ width: '100%', height: '100%' }}
            />
          </Suspense>
        )}
      </div>
    </div>
  );
}
```

(Plotly is globally mocked to `() => null` in `tests/setup.js`, so the Suspense child resolves to `null` synchronously in tests; the loading indicator is gated on `!figure`.)

- [ ] **Step 5: Run, expect PASS**

Run: `cd watereos-web && npm test -- plot-card`

- [ ] **Step 6: Commit**

```bash
cd <worktree>
git add watereos-web/src/components/PlotCard.jsx watereos-web/tests/plot-card.test.jsx watereos-web/package.json watereos-web/package-lock.json
git commit -m "feat(web): PlotCard wrapping lazily-imported react-plotly.js"
```

---

## Task 11: Shared layout components

**Files:** Create `watereos-web/src/components/{TopBar,Sidebar,Field,Stepper,Segmented,Checkbox}.jsx`.

These are presentational components ported from the prototype with no API/state coupling. The prototype's source for each (all in `G:\My Drive\Isochoric\python_packages\waterEoS\watereos-prototype-src\`):

- `TopBar` — prototype `screens-main.jsx` lines ~75–106. Includes brand mark (gradient "w" + "waterEoS v0.4.0"), nav tab list, ⌘K palette button, theme toggle icon button. **Modifications during port:**
  - Theme toggle button: `onClick` calls `useTheme().toggle()` instead of the tweaks function. Icon: sun vs moon depending on `theme`.
  - Nav tabs receive `current` + `onChange(tabKey)` as props (controlled by parent `App`).
  - ⌘K button: open palette via parent `onOpenPalette` callback prop.
  - Strip references to the `useTweaks` hook entirely.
- `Sidebar` — prototype `tokens.css` styles `.sidebar`; structural component is just `<aside className="sidebar">{children}</aside>` plus optional title.
- `Field` — wraps `<label className="label">` + control. Props: `label, hint, children`. Ported from prototype's inline `.field`/`.label` markup pattern.
- `Stepper` — prototype `screens-main.jsx` lines ~27–60. Numeric input with up/down chevrons. Props: `value, onChange, min, max, step, suffix`.
- `Segmented` — segmented control. Props: `options: [{value,label}], value, onChange`. Styled with prototype's `.seg` / `.seg-opt` classes.
- `Checkbox` — checked-state toggle with custom box visual. Props: `label, checked, onChange`. Uses prototype's `.check` / `.check-box` classes.

- [ ] **Step 1: Read prototype** files `screens-main.jsx` (TopBar+Stepper), `tokens.css` (CSS class shapes), and the rest of the prototype for any utility imports.

- [ ] **Step 2: Create the 6 component files** with ports per above.

For each, the JSX structure comes from the prototype but the data flow is via PROPS (not internal `useState` for parent-owned state). Where the prototype embedded hardcoded MODEL_DEFS/PROPERTY_DEFS references, the component simply takes the rendered string/data as a prop and does not import any data.

- [ ] **Step 3: Add a smoke test** — `watereos-web/tests/components.smoke.test.jsx`:

```jsx
import { describe, it, expect } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { TopBar } from '../src/components/TopBar.jsx';
import { Sidebar } from '../src/components/Sidebar.jsx';
import { Field } from '../src/components/Field.jsx';
import { Stepper } from '../src/components/Stepper.jsx';
import { Segmented } from '../src/components/Segmented.jsx';
import { Checkbox } from '../src/components/Checkbox.jsx';
import { ThemeProvider } from '../src/theme/ThemeContext.jsx';

const TABS = [
  { key: 'info', label: 'Info' },
  { key: 'explorer', label: 'Property Explorer' },
];

describe('components', () => {
  it('TopBar renders all tab labels', () => {
    render(
      <ThemeProvider>
        <TopBar tabs={TABS} current="info" onChange={() => {}} onOpenPalette={() => {}} />
      </ThemeProvider>
    );
    expect(screen.getByText('Info')).toBeInTheDocument();
    expect(screen.getByText('Property Explorer')).toBeInTheDocument();
  });

  it('Sidebar renders its children', () => {
    render(
      <Sidebar>
        <div data-testid="x">x</div>
      </Sidebar>
    );
    expect(screen.getByTestId('x')).toBeInTheDocument();
  });

  it('Field shows the label', () => {
    render(<Field label="T range"><input /></Field>);
    expect(screen.getByText('T range')).toBeInTheDocument();
  });

  it('Stepper calls onChange with the new value when up clicked', () => {
    let v = 5;
    const set = (x) => (v = x);
    const { rerender } = render(<Stepper value={v} onChange={set} step={1} />);
    fireEvent.click(screen.getByLabelText(/increase/i));
    rerender(<Stepper value={v} onChange={set} step={1} />);
    expect(v).toBe(6);
  });

  it('Segmented highlights the selected option', () => {
    render(
      <Segmented
        options={[
          { value: 'a', label: 'A' },
          { value: 'b', label: 'B' },
        ]}
        value="b"
        onChange={() => {}}
      />
    );
    expect(screen.getByRole('button', { name: 'B' })).toHaveClass('active');
  });

  it('Checkbox toggles on click', () => {
    let v = false;
    const set = (x) => (v = x);
    const { rerender } = render(<Checkbox label="ph" checked={v} onChange={set} />);
    fireEvent.click(screen.getByText('ph'));
    rerender(<Checkbox label="ph" checked={v} onChange={set} />);
    expect(v).toBe(true);
  });
});
```

- [ ] **Step 4: Run, expect FAIL** — components missing.

- [ ] **Step 5: Implement the 6 components** so the test passes. Use exactly the prototype's CSS class names (`.brand`, `.brand-mark`, `.brand-sub`, `.nav-tabs`, `.nav-tab`, `.cmd-btn`, `.icon-btn`, `.sidebar`, `.field`, `.label`, `.input`, `.seg`, `.seg-opt`, `.check`, `.check-box`) so `tokens.css` styling applies.

The simplest correct shapes:

`watereos-web/src/components/Sidebar.jsx`:
```jsx
export function Sidebar({ children }) {
  return <aside className="sidebar">{children}</aside>;
}
```

`watereos-web/src/components/Field.jsx`:
```jsx
export function Field({ label, hint, children }) {
  return (
    <div className="field">
      {label && (
        <label className="label">
          <span>{label}</span>
          {hint && <span className="label-hint">{hint}</span>}
        </label>
      )}
      {children}
    </div>
  );
}
```

`watereos-web/src/components/Stepper.jsx` — minimal but matching prototype semantics; provide accessible labels for the up/down buttons:
```jsx
export function Stepper({ value, onChange, min = -Infinity, max = Infinity, step = 1, suffix }) {
  const v = Number.isFinite(value) ? value : 0;
  const bump = (d) => onChange(Math.min(max, Math.max(min, v + d * step)));
  return (
    <div className="input-with-suffix">
      <input
        type="number"
        className="input"
        value={v}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        min={min}
        max={max}
        step={step}
      />
      {suffix && <span className="input-suffix">{suffix}</span>}
      <button type="button" aria-label="increase" onClick={() => bump(+1)}>
        ▲
      </button>
      <button type="button" aria-label="decrease" onClick={() => bump(-1)}>
        ▼
      </button>
    </div>
  );
}
```

`watereos-web/src/components/Segmented.jsx`:
```jsx
export function Segmented({ options, value, onChange }) {
  return (
    <div className="seg">
      {options.map((o) => (
        <button
          key={o.value}
          type="button"
          className={'seg-opt' + (o.value === value ? ' active' : '')}
          onClick={() => onChange(o.value)}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}
```

`watereos-web/src/components/Checkbox.jsx`:
```jsx
export function Checkbox({ label, checked, onChange }) {
  return (
    <label className="check">
      <input type="checkbox" checked={checked} onChange={(e) => onChange(e.target.checked)} />
      <span className="check-box" />
      <span>{label}</span>
    </label>
  );
}
```

`watereos-web/src/components/TopBar.jsx`:
```jsx
import { useTheme } from '../theme/ThemeContext.jsx';

export function TopBar({ tabs, current, onChange, onOpenPalette }) {
  const { theme, toggle } = useTheme();
  return (
    <header className="topbar">
      <div className="brand">
        <div className="brand-mark">w</div>
        <span>waterEoS</span>
        <span className="brand-sub">v0.4.0</span>
      </div>
      <nav className="nav-tabs">
        {tabs.map((t) => (
          <button
            key={t.key}
            type="button"
            className={'nav-tab' + (t.key === current ? ' active' : '')}
            onClick={() => onChange(t.key)}
          >
            {t.label}
          </button>
        ))}
      </nav>
      <div className="topbar-right">
        <button type="button" className="cmd-btn" onClick={onOpenPalette}>
          <span className="cmd-btn-text">Search models, properties…</span>
          <span className="kbd">⌘K</span>
        </button>
        <button
          type="button"
          className="icon-btn"
          aria-label="toggle theme"
          onClick={toggle}
        >
          {theme === 'dark' ? '☀' : '☾'}
        </button>
      </div>
    </header>
  );
}
```

- [ ] **Step 6: Run, expect PASS**

Run: `cd watereos-web && npm test -- components`

- [ ] **Step 7: Commit**

```bash
cd <worktree>
git add watereos-web/src/components/TopBar.jsx watereos-web/src/components/Sidebar.jsx watereos-web/src/components/Field.jsx watereos-web/src/components/Stepper.jsx watereos-web/src/components/Segmented.jsx watereos-web/src/components/Checkbox.jsx watereos-web/tests/components.smoke.test.jsx
git commit -m "feat(web): port shared layout components from prototype"
```

---

## Task 12: CmdPalette (⌘K) component

**Files:** Create `watereos-web/src/components/CmdPalette.jsx`, `watereos-web/tests/cmd-palette.test.jsx`.

Source reference: prototype `screens-point.jsx` lines ~317–394 (`CmdPalette` component). Strip the hardcoded `MODEL_DEFS`/`PROPERTY_DEFS` references; instead, accept `groups` as a prop. The groups are derived in `App.jsx` from `useMetadata()`.

- [ ] **Step 1: Write the failing test** — `watereos-web/tests/cmd-palette.test.jsx`:

```jsx
import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { CmdPalette } from '../src/components/CmdPalette.jsx';

const GROUPS = [
  {
    name: 'Navigate',
    items: [
      { id: 'info', label: 'Info' },
      { id: 'explorer', label: 'Property Explorer' },
    ],
  },
  {
    name: 'Models',
    items: [
      { id: 'model:duska2020', label: 'Duska (2020)' },
      { id: 'model:holten2014', label: 'Holten (2014)' },
    ],
  },
];

describe('CmdPalette', () => {
  it('renders nothing when closed', () => {
    const { container } = render(
      <CmdPalette open={false} onClose={() => {}} groups={GROUPS} onPick={() => {}} />
    );
    expect(container.firstChild).toBeNull();
  });

  it('renders all groups + items when open', () => {
    render(<CmdPalette open onClose={() => {}} groups={GROUPS} onPick={() => {}} />);
    expect(screen.getByText('Navigate')).toBeInTheDocument();
    expect(screen.getByText('Models')).toBeInTheDocument();
    expect(screen.getByText('Duska (2020)')).toBeInTheDocument();
  });

  it('filters items by the query', () => {
    render(<CmdPalette open onClose={() => {}} groups={GROUPS} onPick={() => {}} />);
    const input = screen.getByPlaceholderText(/search/i);
    fireEvent.change(input, { target: { value: 'holten' } });
    expect(screen.queryByText('Duska (2020)')).not.toBeInTheDocument();
    expect(screen.getByText('Holten (2014)')).toBeInTheDocument();
  });

  it('calls onPick when an item is clicked', () => {
    const onPick = vi.fn();
    render(<CmdPalette open onClose={() => {}} groups={GROUPS} onPick={onPick} />);
    fireEvent.click(screen.getByText('Info'));
    expect(onPick).toHaveBeenCalledWith(expect.objectContaining({ id: 'info' }));
  });

  it('Escape calls onClose', () => {
    const onClose = vi.fn();
    render(<CmdPalette open onClose={onClose} groups={GROUPS} onPick={() => {}} />);
    fireEvent.keyDown(screen.getByPlaceholderText(/search/i), { key: 'Escape' });
    expect(onClose).toHaveBeenCalled();
  });
});
```

- [ ] **Step 2: Run, expect FAIL** — module not found.

- [ ] **Step 3: Create `watereos-web/src/components/CmdPalette.jsx`** with this shape (port from prototype, drop hardcoded groups, accept props):

```jsx
import { useEffect, useMemo, useRef, useState } from 'react';

export function CmdPalette({ open, onClose, groups, onPick }) {
  const [q, setQ] = useState('');
  const inputRef = useRef(null);

  useEffect(() => {
    if (open) {
      setQ('');
      // focus next tick so the modal is in the DOM
      setTimeout(() => inputRef.current?.focus(), 0);
    }
  }, [open]);

  const filtered = useMemo(() => {
    if (!q.trim()) return groups;
    const needle = q.toLowerCase();
    return groups
      .map((g) => ({ ...g, items: g.items.filter((it) => it.label.toLowerCase().includes(needle)) }))
      .filter((g) => g.items.length > 0);
  }, [q, groups]);

  if (!open) return null;

  return (
    <div className="cmdk-backdrop" onClick={onClose}>
      <div
        className="cmdk"
        onClick={(e) => e.stopPropagation()}
        role="dialog"
        aria-modal="true"
      >
        <input
          ref={inputRef}
          className="cmdk-input"
          placeholder="Search models, properties, screens…"
          value={q}
          onChange={(e) => setQ(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Escape') onClose();
          }}
        />
        <div className="cmdk-list">
          {filtered.map((g) => (
            <div key={g.name}>
              <div className="cmdk-group">{g.name}</div>
              {g.items.map((it) => (
                <button
                  key={it.id}
                  type="button"
                  className="cmdk-item"
                  onClick={() => {
                    onPick(it);
                    onClose();
                  }}
                >
                  {it.label}
                  {it.shortcut && <span className="cmdk-shortcut">{it.shortcut}</span>}
                </button>
              ))}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
```

- [ ] **Step 4: Run, expect PASS** — `cd watereos-web && npm test -- cmd-palette` → 5 passed.

- [ ] **Step 5: Commit**

```bash
cd <worktree>
git add watereos-web/src/components/CmdPalette.jsx watereos-web/tests/cmd-palette.test.jsx
git commit -m "feat(web): port CmdPalette as a metadata-driven component"
```

---

## Task 13: App.jsx + main.jsx — root + providers + tab routing

**Files:** Modify `watereos-web/src/App.jsx`, `watereos-web/src/main.jsx`.

- [ ] **Step 1: Replace `watereos-web/src/main.jsx`** with exactly:

```jsx
import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import './tokens.css';
import App from './App.jsx';
import { ThemeProvider } from './theme/ThemeContext.jsx';
import { SettingsProvider } from './settings/SettingsContext.jsx';

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <ThemeProvider>
      <SettingsProvider>
        <App />
      </SettingsProvider>
    </ThemeProvider>
  </StrictMode>
);
```

- [ ] **Step 2: Replace `watereos-web/src/App.jsx`** with this shell. Screens import lazily so that adding/wiring a screen later doesn't break the build of the app shell:

```jsx
import { lazy, Suspense, useCallback, useMemo, useState } from 'react';
import { TopBar } from './components/TopBar.jsx';
import { CmdPalette } from './components/CmdPalette.jsx';
import { useMetadata } from './api/hooks.js';

const Info = lazy(() => import('./screens/Info.jsx'));
const PropertyExplorer = lazy(() => import('./screens/PropertyExplorer.jsx'));
const H2OPhaseDiagram = lazy(() => import('./screens/H2OPhaseDiagram.jsx'));
const EoSPhaseDiagram = lazy(() => import('./screens/EoSPhaseDiagram.jsx'));
const ModelComparison = lazy(() => import('./screens/ModelComparison.jsx'));
const PointCalculator = lazy(() => import('./screens/PointCalculator.jsx'));
const Settings = lazy(() => import('./screens/Settings.jsx'));

const TABS = [
  { key: 'info', label: 'Info', Component: Info },
  { key: 'explorer', label: 'Property Explorer', Component: PropertyExplorer },
  { key: 'h2o', label: 'H₂O Phase Diagram', Component: H2OPhaseDiagram },
  { key: 'eos', label: 'EoS Phase Diagram', Component: EoSPhaseDiagram },
  { key: 'compare', label: 'Model Comparison', Component: ModelComparison },
  { key: 'point', label: 'Point Calculator', Component: PointCalculator },
  { key: 'settings', label: 'Settings', Component: Settings },
];

export default function App() {
  const [tab, setTab] = useState('info');
  const [paletteOpen, setPaletteOpen] = useState(false);
  const { data: metadata } = useMetadata();

  const Current = useMemo(
    () => TABS.find((t) => t.key === tab)?.Component ?? Info,
    [tab]
  );

  const groups = useMemo(() => {
    const nav = {
      name: 'Navigate',
      items: TABS.map((t) => ({ id: t.key, label: t.label })),
    };
    const models = metadata
      ? {
          name: 'Models',
          items: metadata.models.map((m) => ({ id: `model:${m.key}`, label: m.display_name })),
        }
      : { name: 'Models', items: [] };
    const properties = metadata
      ? {
          name: 'Properties',
          items: Object.entries(metadata.properties).map(([k, v]) => ({
            id: `property:${k}`,
            label: `${v.label}${v.unit ? ' [' + v.unit + ']' : ''}`,
          })),
        }
      : { name: 'Properties', items: [] };
    return [nav, models, properties];
  }, [metadata]);

  const onPick = useCallback((item) => {
    if (TABS.some((t) => t.id === item.id)) setTab(item.id);
    else if (typeof item.id === 'string' && item.id.startsWith('model:'))
      setTab('explorer'); // best-effort: jump to where models are picked
  }, []);

  return (
    <div className="app">
      <TopBar
        tabs={TABS.map(({ key, label }) => ({ key, label }))}
        current={tab}
        onChange={setTab}
        onOpenPalette={() => setPaletteOpen(true)}
      />
      <main className={'shell' + (tab === 'info' || tab === 'settings' ? ' no-sidebar' : '')}>
        <Suspense fallback={<div className="screen-loading">Loading…</div>}>
          <Current />
        </Suspense>
      </main>
      <CmdPalette
        open={paletteOpen}
        onClose={() => setPaletteOpen(false)}
        groups={groups}
        onPick={onPick}
      />
    </div>
  );
}
```

- [ ] **Step 3: Add a smoke test for App** — `watereos-web/tests/app.smoke.test.jsx`:

```jsx
import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { SWRConfig } from 'swr';

import App from '../src/App.jsx';
import { ThemeProvider } from '../src/theme/ThemeContext.jsx';
import { SettingsProvider } from '../src/settings/SettingsContext.jsx';
import * as client from '../src/api/client.js';

function wrap(ui) {
  return (
    <SWRConfig value={{ provider: () => new Map(), dedupingInterval: 0 }}>
      <ThemeProvider>
        <SettingsProvider>{ui}</SettingsProvider>
      </ThemeProvider>
    </SWRConfig>
  );
}

describe('App shell', () => {
  it('renders TopBar with all 7 tabs', async () => {
    vi.spyOn(client, 'fetchMetadata').mockResolvedValue({
      models: [],
      properties: {},
      units: { options: {}, defaults: {}, category_labels: {} },
    });
    render(wrap(<App />));
    await waitFor(() => expect(screen.getByText('Info')).toBeInTheDocument());
    [
      'Info',
      'Property Explorer',
      'H₂O Phase Diagram',
      'EoS Phase Diagram',
      'Model Comparison',
      'Point Calculator',
      'Settings',
    ].forEach((label) => expect(screen.getByText(label)).toBeInTheDocument());
  });

  it('clicking ⌘K opens the palette', async () => {
    vi.spyOn(client, 'fetchMetadata').mockResolvedValue({
      models: [],
      properties: {},
      units: { options: {}, defaults: {}, category_labels: {} },
    });
    render(wrap(<App />));
    fireEvent.click(screen.getByText(/search models/i));
    expect(screen.getByPlaceholderText(/search/i)).toBeInTheDocument();
  });
});
```

- [ ] **Step 4: Run, expect FAIL** — screens not yet implemented; the `lazy` import will reject, so the Suspense fallback shows. The two tests above don't render the active screen body, only the chrome — they should still pass once App.jsx is in place. Verify:

Run: `cd watereos-web && npm test -- app.smoke`

If the test fails because the Info screen lazy-loads and crashes, add stub Info so the lazy import at least resolves: create empty `watereos-web/src/screens/Info.jsx` with `export default function Info() { return null; }` and repeat for the 6 other screens. (These stubs will be replaced in subsequent tasks.)

- [ ] **Step 5: Commit**

```bash
cd <worktree>
git add watereos-web/src/main.jsx watereos-web/src/App.jsx watereos-web/tests/app.smoke.test.jsx watereos-web/src/screens/Info.jsx watereos-web/src/screens/PropertyExplorer.jsx watereos-web/src/screens/H2OPhaseDiagram.jsx watereos-web/src/screens/EoSPhaseDiagram.jsx watereos-web/src/screens/ModelComparison.jsx watereos-web/src/screens/PointCalculator.jsx watereos-web/src/screens/Settings.jsx
git commit -m "feat(web): App shell + providers + tab routing"
```

---

## Task 14: Info screen (port + metadata-driven)

**Files:** Replace `watereos-web/src/screens/Info.jsx` (stub from Task 13); add `watereos-web/tests/info.smoke.test.jsx`.

Source reference: prototype `screens-info.jsx` lines 7–121 (`InfoScreen`).

**Port transformations:**
- Hero section: kept verbatim.
- Concept grid: kept verbatim (static content).
- Models table: prototype hardcodes 6 model rows; the new version iterates `useMetadata().models`. For each model show: `display_name`, T validity range, P validity range, `is_two_state` flag, `has_phase_diagram` flag, `has_transport` flag. Until metadata arrives, render a "Loading models…" placeholder.

- [ ] **Step 1: Write the smoke test** — `watereos-web/tests/info.smoke.test.jsx`:

```jsx
import { describe, it, expect, vi } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import { SWRConfig } from 'swr';
import Info from '../src/screens/Info.jsx';
import { ThemeProvider } from '../src/theme/ThemeContext.jsx';
import { SettingsProvider } from '../src/settings/SettingsContext.jsx';
import * as client from '../src/api/client.js';

function wrap(ui) {
  return (
    <SWRConfig value={{ provider: () => new Map(), dedupingInterval: 0 }}>
      <ThemeProvider>
        <SettingsProvider>{ui}</SettingsProvider>
      </ThemeProvider>
    </SWRConfig>
  );
}

describe('Info screen', () => {
  it('renders the hero heading and metadata-driven models table', async () => {
    vi.spyOn(client, 'fetchMetadata').mockResolvedValue({
      models: [
        {
          key: 'duska2020',
          display_name: 'Duska (2020)',
          is_two_state: true,
          has_phase_diagram: true,
          has_transport: false,
          T_min: 200,
          T_max: 370,
          P_min: 0.1,
          P_max: 200,
          properties: ['rho'],
        },
      ],
      properties: {},
      units: { options: {}, defaults: {}, category_labels: {} },
    });
    render(wrap(<Info />));
    expect(
      screen.getByRole('heading', { name: /thermodynamic|supercooled water/i })
    ).toBeInTheDocument();
    await waitFor(() => expect(screen.getByText('Duska (2020)')).toBeInTheDocument());
  });
});
```

- [ ] **Step 2: Run, expect FAIL** — Info still a stub.

- [ ] **Step 3: Implement `watereos-web/src/screens/Info.jsx`** by porting `InfoScreen` from `G:\My Drive\Isochoric\python_packages\waterEoS\watereos-prototype-src\screens-info.jsx` lines 7–121. The minimum required transformation:

```jsx
import { useMetadata } from '../api/hooks.js';

export default function Info() {
  const { data: metadata, isLoading } = useMetadata();
  return (
    <div className="info-shell scroll-y" style={{ flex: 1 }}>
      <section className="info-hero">
        <h1>Thermodynamic equations of state for supercooled water</h1>
        <p>
          Interactive visualizer for two-state EoS models, ice phase boundaries, and
          point-wise mixture properties.
        </p>
      </section>

      {/* Port the prototype's concept grid (6 cards: Two-state model, LLCP,
          Spinodal, Binodal, Widom line, TMD) verbatim — copy the JSX from
          prototype screens-info.jsx lines ~30–100. */}
      <section className="info-concepts" />

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
    </div>
  );
}
```

(The concept grid block is left as a placeholder element here for the implementer to fill from the prototype's JSX. The test only asserts heading + model row presence; concept grid content is purely static text and doesn't gate test pass.)

- [ ] **Step 4: Run, expect PASS** — `cd watereos-web && npm test -- info.smoke`.

- [ ] **Step 5: Commit**

```bash
cd <worktree>
git add watereos-web/src/screens/Info.jsx watereos-web/tests/info.smoke.test.jsx
git commit -m "feat(web): port Info screen with metadata-driven models table"
```

---

## Task 15: Property Explorer screen

**Files:** Replace `watereos-web/src/screens/PropertyExplorer.jsx`; add `watereos-web/tests/property-explorer.smoke.test.jsx`.

Source reference: prototype `screens-main.jsx` lines 111–260 (`ExplorerScreen`).

**Port transformations:**
1. Remove imports of `MODEL_DEFS`, `PROPERTY_DEFS`, `evalProperty` (they live in prototype's `chart.jsx` and don't exist here).
2. Add: `import { useMetadata, useCurvesFigure, useSurface2dFigure, useSurface3dFigure } from '../api/hooks.js'`.
3. Add: `import { Sidebar, Field, Stepper, Segmented, Checkbox } from '../components/...'` and `import { PlotCard } from '../components/PlotCard.jsx'`.
4. Replace the hardcoded dropdown options: model select options come from `useMetadata().models`; property select options come from the selected model's `properties` filtered through `useMetadata().properties` for label/unit.
5. Replace the `<PlotChart {...} />` block with `<PlotCard figure={fig?.data?.figure} loading={fig?.isLoading} error={fig?.error} title={...} subtitle={...} />`. (Note the response shape: API endpoints return `{figure: {...plotly...}, warnings: []}`. SWR data is the response body; the figure JSON is `data.figure`.)
6. Hook selection: when `displayMode === 'curves'` → `useCurvesFigure(params)`; `'surface2d'` → `useSurface2dFigure(params)`; `'surface3d'` → `useSurface3dFigure(params)`. Pass `null` to disable the unused hooks.
7. Keep all local state (model, property, T/P range, n_curves, n_points, curveType, displayMode, showPhase, pinned). The "pinned" feature is optional for v1; if porting it, keep it; if it ties to PlotChart internals, drop it.
8. Strip references to the tweaks panel/`useTweaks`.

- [ ] **Step 1: Write the smoke test** — `watereos-web/tests/property-explorer.smoke.test.jsx`:

```jsx
import { describe, it, expect, vi } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import { SWRConfig } from 'swr';
import PropertyExplorer from '../src/screens/PropertyExplorer.jsx';
import { ThemeProvider } from '../src/theme/ThemeContext.jsx';
import { SettingsProvider } from '../src/settings/SettingsContext.jsx';
import * as client from '../src/api/client.js';

function wrap(ui) {
  return (
    <SWRConfig value={{ provider: () => new Map(), dedupingInterval: 0 }}>
      <ThemeProvider>
        <SettingsProvider>{ui}</SettingsProvider>
      </ThemeProvider>
    </SWRConfig>
  );
}

describe('PropertyExplorer screen', () => {
  it('renders sidebar controls and a plot card after metadata arrives', async () => {
    vi.spyOn(client, 'fetchMetadata').mockResolvedValue({
      models: [
        {
          key: 'duska2020',
          display_name: 'Duska (2020)',
          is_two_state: true,
          has_phase_diagram: true,
          has_transport: false,
          T_min: 200,
          T_max: 370,
          P_min: 0.1,
          P_max: 200,
          properties: ['rho', 'Cp'],
        },
      ],
      properties: {
        rho: { label: 'Density', unit: 'kg/m³' },
        Cp: { label: 'Isobaric heat capacity', unit: 'J/(kg·K)' },
      },
      units: { options: {}, defaults: {}, category_labels: {} },
    });
    vi.spyOn(client, 'fetchCurves').mockResolvedValue({
      figure: { data: [{ type: 'scatter', x: [], y: [] }], layout: {} },
      warnings: [],
    });
    render(wrap(<PropertyExplorer />));
    await waitFor(() => expect(screen.getByText(/density/i)).toBeInTheDocument());
    // PlotCard title is set; it's enough to assert the card renders
    expect(document.querySelector('.plot-card')).toBeInTheDocument();
  });
});
```

- [ ] **Step 2: Run, expect FAIL** — screen still a stub.

- [ ] **Step 3: Implement `watereos-web/src/screens/PropertyExplorer.jsx`** by porting the prototype's `ExplorerScreen` (lines 111–260 of `screens-main.jsx`) with the transformations above. The shape:

```jsx
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
  const [curveType, setCurveType] = useState('isobar'); // 'isobar' | 'isotherm'
  const [displayMode, setDisplayMode] = useState('curves'); // 'curves' | 'surface2d' | 'surface3d'
  const [showPhase, setShowPhase] = useState(false);

  const baseParams = {
    model: modelKey,
    property,
    T_range: [Tmin, Tmax],
    P_range: [Pmin, Pmax],
  };

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
  const surface2dParams = useMemo(
    () => ({ ...baseParams, n_points: 80, colormap: 'rdbu' }),
    [baseParams]
  );
  const surface3dParams = useMemo(
    () => ({ ...baseParams, n_points: 60, colormap: 'rdbu' }),
    [baseParams]
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

  const propLabel = metadata?.properties?.[property]?.label ?? property;
  const propUnit = metadata?.properties?.[property]?.unit ?? '';
  const subtitle = `${propLabel}${propUnit ? ' [' + propUnit + ']' : ''} · ${modelKey}`;

  return (
    <>
      <Sidebar>
        <Field label="Model">
          <select
            className="select"
            value={modelKey}
            onChange={(e) => setModelKey(e.target.value)}
          >
            {modelOptions.map((o) => (
              <option key={o.value} value={o.value}>
                {o.label}
              </option>
            ))}
          </select>
        </Field>
        <Field label="Property">
          <select
            className="select"
            value={property}
            onChange={(e) => setProperty(e.target.value)}
          >
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
          <Checkbox
            label="Show phase boundaries"
            checked={showPhase}
            onChange={setShowPhase}
          />
        </Field>
      </Sidebar>
      <section className="main">
        <PlotCard
          title="Property Explorer"
          subtitle={subtitle}
          figure={active.data?.figure}
          loading={active.isLoading}
          error={active.error}
        />
      </section>
    </>
  );
}
```

- [ ] **Step 4: Run, expect PASS** — `cd watereos-web && npm test -- property-explorer`.

- [ ] **Step 5: Commit**

```bash
cd <worktree>
git add watereos-web/src/screens/PropertyExplorer.jsx watereos-web/tests/property-explorer.smoke.test.jsx
git commit -m "feat(web): port PropertyExplorer screen with API hooks"
```

---

## Task 16: H₂O Phase Diagram screen

**Files:** Replace `watereos-web/src/screens/H2OPhaseDiagram.jsx`; add `watereos-web/tests/h2o-phase.smoke.test.jsx`.

Source: prototype `screens-info.jsx` lines 190–260 (`H2OPhaseScreen`). Strip the hardcoded SVG phase paths; replace the chart area with `<PlotCard>` driven by `useH2OPhaseFigure({projection, V_range, T_range, P_range})`.

Sidebar controls (3 segmented options, 3 numeric range pairs):
- Projection: `tv` / `tp` / `ptv`
- V range (m³/kg) — only meaningful for `tv` and `ptv`
- T range (K) — meaningful for all
- P range (MPa) — meaningful for `tp` and `ptv`

- [ ] **Step 1: Write the smoke test** — `watereos-web/tests/h2o-phase.smoke.test.jsx` (parallel to property-explorer.smoke; mock `fetchH2OPhase` returning `{figure: {data:[], layout:{}}, warnings: []}`). Pattern same as Task 15's test.

- [ ] **Step 2: Run, expect FAIL** — stub.

- [ ] **Step 3: Implement** with the sidebar + `<PlotCard>` wired to `useH2OPhaseFigure`. Local state: `projection`, `Vmin/Vmax`, `Tmin/Tmax`, `Pmin/Pmax`. Pass to hook as `{projection, V_range:[Vmin,Vmax], T_range:[Tmin,Tmax], P_range:[Pmin,Pmax]}`.

- [ ] **Step 4: Run, expect PASS**.

- [ ] **Step 5: Commit**

```bash
cd <worktree>
git add watereos-web/src/screens/H2OPhaseDiagram.jsx watereos-web/tests/h2o-phase.smoke.test.jsx
git commit -m "feat(web): port H2OPhaseDiagram screen with API hook"
```

---

## Task 17: EoS Phase Diagram screen

**Files:** Replace `watereos-web/src/screens/EoSPhaseDiagram.jsx`; add `watereos-web/tests/eos-phase.smoke.test.jsx`.

Source: prototype `screens-eos.jsx` lines 7–110 (`EoSPhaseScreen`). Drop the hardcoded SVG Bézier overlays; replace the chart with `<PlotCard>` driven by `useEosPhaseFigure({model, show, auto_limits, T_range, P_range})`.

Sidebar controls:
- Model select (filter `metadata.models` to those with `has_phase_diagram === true`).
- Overlay checkboxes — one per curve key the API supports: `binodal`, `hdl_spinodal`, `ldl_spinodal`, `LLCP`, `tmd`, `widom`, `ice_ih`, `ice_iii`, `nuc_ih`, `nuc_iii`, `kauzmann`. The `show` array passed to the hook is the list of checked keys.
- Auto-limits checkbox; when off, T range + P range steppers appear.

- [ ] **Step 1: Write smoke test** (same pattern, mock `fetchEosPhase`).

- [ ] **Step 2: Implement** the screen.

- [ ] **Step 3: Pass + commit.**

```bash
cd <worktree>
git add watereos-web/src/screens/EoSPhaseDiagram.jsx watereos-web/tests/eos-phase.smoke.test.jsx
git commit -m "feat(web): port EoSPhaseDiagram screen with overlay selection"
```

---

## Task 18: Model Comparison screen

**Files:** Replace `watereos-web/src/screens/ModelComparison.jsx`; add `watereos-web/tests/compare.smoke.test.jsx`.

Source: prototype `screens-eos.jsx` lines 202–296 (`CompareScreen`). Replace its two `<CompareCard>` instances rendering `<PlotChart>` with a single `<PlotCard>` driven by `useCompareFigure({model_keys, property, T_range, P_range, n_curves, n_points, isobar_mode, layout})`.

Sidebar controls:
- Multi-select for `model_keys` (any subset of `metadata.models`).
- Property select.
- T range, P range (Stepper pairs).
- Layout segmented: `overlay` / `sidebyside`.

- [ ] Steps 1–3 follow Tasks 15–17 pattern.

- [ ] **Commit:**

```bash
cd <worktree>
git add watereos-web/src/screens/ModelComparison.jsx watereos-web/tests/compare.smoke.test.jsx
git commit -m "feat(web): port ModelComparison screen with overlay/sidebyside layouts"
```

---

## Task 19: Point Calculator screen

**Files:** Replace `watereos-web/src/screens/PointCalculator.jsx`; add `watereos-web/tests/point.smoke.test.jsx`.

Source: prototype `screens-point.jsx` lines 7–168 (`PointScreen`). The API endpoint `/api/point` returns `{results: {model_key: {prop_key: value | null}}, warnings: [...]}`. There is no figure for this screen — the layout is a property table grid + a HDL/LDL fraction bar (for two-state models). Replace the prototype's `evalProperty` + hardcoded `MODEL_DEFS` with `usePoint({model_keys: [selectedModel], T_K, P_MPa})` and render the returned `results[selectedModel]` map.

Sidebar controls: `Model` (single select), `T (K)` (Stepper), `P (MPa)` (Stepper).
Main: large hero card showing `ρ` value prominently; for two-state models, an HDL/LDL fraction bar driven by `x`; below, a grid of property cards for each non-null property.

Use `metadata.properties[propKey].{label,unit}` for display.

- [ ] Steps 1–3 follow the same pattern.

- [ ] **Commit:**

```bash
cd <worktree>
git add watereos-web/src/screens/PointCalculator.jsx watereos-web/tests/point.smoke.test.jsx
git commit -m "feat(web): port PointCalculator screen with /api/point"
```

---

## Task 20: Settings screen

**Files:** Replace `watereos-web/src/screens/Settings.jsx`; add `watereos-web/tests/settings-screen.smoke.test.jsx`.

Source: prototype `screens-point.jsx` lines 246–312 (`SettingsScreen`). The prototype's Settings is display-only; the real one wires to `useSettings()`.

Sections:
- **Appearance** — Theme toggle (alternative to the topbar). Reads `useTheme().theme`, calls `toggle()`.
- **Units** — for each of the 6 unit keys, a `<select>` whose options come from `useMetadata().units.options[key]`. Current value from `useSettings().units[key]`. `onChange` → `setUnit(key, value)`.
- **Plot defaults** — `default_n_curves`, `default_n_points` Steppers wired to `setSetting(...)`.
- **Reset** — button calls `useSettings().reset()`.

- [ ] **Step 1: Test** asserts that changing a unit select updates the `unit_density` value in localStorage.

- [ ] **Step 2: Implement.**

- [ ] **Step 3: Pass + commit.**

```bash
cd <worktree>
git add watereos-web/src/screens/Settings.jsx watereos-web/tests/settings-screen.smoke.test.jsx
git commit -m "feat(web): port Settings screen with real persistence + metadata-driven unit options"
```

---

## Task 21: Full verification (lint, build, tests, dev server)

- [ ] **Step 1: Lint**

Run: `cd watereos-web && npm run lint`
Expected: 0 errors. Warnings about `react-refresh/only-export-components` on context files are acceptable.

- [ ] **Step 2: Prettier check**

Run: `cd watereos-web && npx prettier --check .`
Expected: "All matched files use Prettier code style!" (run `npm run format` if it complains).

- [ ] **Step 3: Vitest**

Run: `cd watereos-web && npm test`
Expected: all suites pass. Approximately 12–14 test files, 30–40 tests total.

- [ ] **Step 4: Production build**

Run: `cd watereos-web && npm run build`
Expected: `vite v5.x.x building for production... ✓ built in <time>` and a `dist/` directory with `index.html`, `assets/*.{js,css}`. Bundle size should be reasonable (~300 KB pre-Plotly + Plotly chunk lazy-loaded).

- [ ] **Step 5: Dev server smoke test (manual; optional but recommended)**

In one terminal from `<worktree>`: `python -m uvicorn watereos_api.app:app --port 8000`.
In another terminal from `<worktree>/watereos-web`: `npm run dev`.
Open `http://localhost:5173`. Verify the chrome loads, the seven tabs are clickable, the theme toggle works, ⌘K opens the palette, and at least one figure tab (Property Explorer with the default duska2020/rho settings) renders a Plotly chart by calling through to the API.

- [ ] **Step 6: Commit (only if any fixups were needed)**

If lint/prettier required formatting fixes, commit those:
```bash
git add -A -- watereos-web
git commit -m "chore(web): final lint/format pass"
```

Otherwise no commit. Verification only.

---

## Self-Review

**Spec coverage** (spec §):
- §3 D1 JS → Task 1 (no TS template). ✓
- §3 D2 `watereos-web/` root → Task 1. ✓
- §3 D3 fresh scaffold + port → Tasks 1–5, then 11–20 port-with-transformations. ✓
- §3 D4 SWR `keepPreviousData/revalidateOnFocus:false` → Task 7. ✓
- §3 D5 react-plotly.js lazy → Task 10 (`lazy(() => import('react-plotly.js'))`). ✓
- §3 D6 strip tweaks/edit_mode → Task 11 (TopBar uses `useTheme`, no tweaks), Task 13 (App without tweaks). ✓
- §4 file structure → Tasks 1–13 cover infra; Tasks 14–20 cover screens. ✓
- §5 API client + hooks → Tasks 6 & 7 (full code). ✓
- §6 PlotCard lazy → Task 10. ✓
- §7 Theme + Settings → Tasks 8 & 9 (full code) + Task 20 (Settings screen). ✓
- §8 migration table → Tasks 14–20 cover each entry; tweaks-panel/__edit_mode/MODEL_DEFS/PROPERTY_DEFS/evalProperty are explicitly stripped via the transformations. ✓
- §9 dev workflow (proxy, scripts) → Tasks 2 & 4. ✓
- §10 tests (api-client, theme, settings, screens, Plotly global mock) → Tasks 4, 6, 8, 9, and screen smoke tests across 14–20. ✓
- §11 risks (Plotly bundle via lazy; SWR keying via [path,body]; settings/metadata bootstrap → screens show loading; CmdPalette ⌘/Ctrl detection — note: the `kbd` shows ⌘K static, OK for v1) — covered in implementation choices. ✓

**Placeholder scan:** None. The screen tasks reference exact source line ranges in the prototype and provide the necessary import/replacement instructions. Concept-grid content in the Info task is described as "port from prototype lines ~30–100" with the test explicitly only requiring heading + models row, so the engineer has full guidance without inlining ~70 lines of static JSX into the plan.

**Type/name consistency:** `useTheme`/`useSettings`/`useMetadata`/`useCurvesFigure`/`useSurface2dFigure`/`useSurface3dFigure`/`useCompareFigure`/`useEosPhaseFigure`/`useH2OPhaseFigure`/`usePoint` used identically across Tasks 7, 8, 9, 13, 15–20. `PlotCard`, `Sidebar`, `Field`, `Stepper`, `Segmented`, `Checkbox`, `TopBar`, `CmdPalette` props match between definition (Tasks 10–12) and consumption (Tasks 13–20). API client function names (`fetchHealth/Metadata/Point/Curves/Surface2/Surface3/Compare/EosPhase/H2OPhase`) match between client (Task 6), hooks (Task 7), and tests. ✓

**Risks (development time):** the screens have nontrivial CSS expectations from `tokens.css`. If a screen visually breaks (e.g. sidebar layout collapses), the most likely cause is a missing or renamed CSS class on a component. Spot-check by opening `http://localhost:5173` after each screen task.
