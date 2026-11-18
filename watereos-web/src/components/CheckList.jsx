import { useMemo, useState } from 'react';

// A list of checkboxes with counts, "all"/"none" shortcuts and an optional
// search box. `items` is [{ value, label, count }]; `checked(value)` says
// whether a box is ticked; `onToggle(value, next)` changes one box and
// `onSet(values, next)` changes many at once.
export function CheckList({
  items,
  checked,
  onToggle,
  onSet,
  search = false,
  maxHeight,
  emptyText,
}) {
  const [q, setQ] = useState('');
  const shown = useMemo(() => {
    if (!q.trim()) return items;
    const s = q.trim().toLowerCase();
    return items.filter((it) => `${it.label} ${it.value}`.toLowerCase().includes(s));
  }, [items, q]);
  return (
    <div className="md-checklist">
      {search && (
        <input
          className="input md-checklist-search"
          type="search"
          placeholder="Search…"
          value={q}
          onChange={(e) => setQ(e.target.value)}
        />
      )}
      {onSet && shown.length > 1 && (
        <div className="md-checklist-links">
          <button
            type="button"
            className="md-link"
            onClick={() =>
              onSet(
                shown.map((i) => i.value),
                true
              )
            }
          >
            all
          </button>
          <span>·</span>
          <button
            type="button"
            className="md-link"
            onClick={() =>
              onSet(
                shown.map((i) => i.value),
                false
              )
            }
          >
            none
          </button>
        </div>
      )}
      <div
        className="md-checklist-items"
        style={maxHeight ? { maxHeight, overflowY: 'auto' } : undefined}
      >
        {shown.length === 0 && <div className="md-muted">{emptyText ?? 'Nothing to choose.'}</div>}
        {shown.map((it) => (
          <label key={String(it.value)} className="md-check-row" title={it.title ?? it.label}>
            <input
              type="checkbox"
              checked={checked(it.value)}
              onChange={(e) => onToggle(it.value, e.target.checked)}
            />
            <span className="check-box" />
            <span className="md-check-label">{it.label}</span>
            {it.count !== undefined && <span className="md-check-count">{it.count}</span>}
          </label>
        ))}
      </div>
    </div>
  );
}
