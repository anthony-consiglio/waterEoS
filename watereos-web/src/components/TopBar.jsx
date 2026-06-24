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
        <button type="button" className="icon-btn" aria-label="toggle theme" onClick={toggle}>
          {theme === 'dark' ? '☀' : '☾'}
        </button>
      </div>
    </header>
  );
}
