import { useEffect, useMemo, useRef, useState } from 'react';

function CmdPaletteInner({ onClose, groups, onPick }) {
  const [q, setQ] = useState('');
  const inputRef = useRef(null);

  useEffect(() => {
    // focus next tick so the modal is in the DOM
    setTimeout(() => inputRef.current?.focus(), 0);
  }, []);

  const filtered = useMemo(() => {
    if (!q.trim()) return groups;
    const needle = q.toLowerCase();
    return groups
      .map((g) => ({
        ...g,
        items: g.items.filter((it) => it.label.toLowerCase().includes(needle)),
      }))
      .filter((g) => g.items.length > 0);
  }, [q, groups]);

  return (
    <div className="cmdk-backdrop" onClick={onClose}>
      <div className="cmdk" onClick={(e) => e.stopPropagation()} role="dialog" aria-modal="true">
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

export function CmdPalette({ open, onClose, groups, onPick }) {
  if (!open) return null;
  return <CmdPaletteInner onClose={onClose} groups={groups} onPick={onPick} />;
}
