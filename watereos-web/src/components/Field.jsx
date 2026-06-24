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
