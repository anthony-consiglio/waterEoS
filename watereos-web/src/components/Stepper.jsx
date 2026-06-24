export function Stepper({ value, onChange, min = -Infinity, max = Infinity, step = 1, suffix }) {
  const v = Number.isFinite(value) ? value : 0;
  const bump = (d) => onChange(Math.min(max, Math.max(min, v + d * step)));
  return (
    <div className="stepper">
      <input
        type="number"
        className="input stepper-input"
        value={v}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        min={min}
        max={max}
        step={step}
      />
      {suffix && <span className="stepper-suffix">{suffix}</span>}
      <div className="stepper-spin" aria-hidden="false">
        <button type="button" aria-label="increase" onClick={() => bump(+1)} className="stepper-btn">
          ▲
        </button>
        <button type="button" aria-label="decrease" onClick={() => bump(-1)} className="stepper-btn">
          ▼
        </button>
      </div>
    </div>
  );
}
