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

  const Current = useMemo(() => TABS.find((t) => t.key === tab)?.Component ?? Info, [tab]);

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
    if (TABS.some((t) => t.key === item.id)) setTab(item.id);
    else if (typeof item.id === 'string' && item.id.startsWith('model:')) setTab('explorer'); // best-effort: jump to where models are picked
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
          <Current setTab={setTab} />
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
