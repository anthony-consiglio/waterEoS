// Labels and number formatting for the measured-data views.

export const UNIT_PRETTY = {
  'kg/m3': 'kg/m³',
  '1/MPa': 'MPa⁻¹',
  '1/K': 'K⁻¹',
  'J/(kg K)': 'J kg⁻¹ K⁻¹',
  'J/(mol K)': 'J mol⁻¹ K⁻¹',
  'J/mol': 'J/mol',
  'm/s': 'm/s',
  K: 'K',
  MPa: 'MPa',
  '1/(m3 s)': 'm⁻³ s⁻¹',
  s: 's',
  'cm3/mol': 'cm³/mol',
  1: 'fraction',
  'm2/s': 'm²/s',
  'Pa s': 'Pa·s',
};

export const unitPretty = (u) => UNIT_PRETTY[u] ?? u ?? '';

export function axisTitle(label, unit) {
  return unit && unit !== '1' ? `${label} (${unitPretty(unit)})` : label;
}

export const FIT_TOKENS = [
  'Holten2012',
  'Holten2014',
  'Caupin2019',
  'Duska2020',
  'IAPWS95',
  'MishimaSumita2023',
];
export const FIT_LABELS = {
  Holten2012: 'Holten 2012',
  Holten2014: 'Holten 2014',
  Caupin2019: 'Caupin 2019',
  Duska2020: 'Duška 2020',
  IAPWS95: 'IAPWS-95',
  MishimaSumita2023: 'Mishima–Sumita 2023',
};

export function fmtNum(x, sig = 6) {
  if (x === null || x === undefined || !Number.isFinite(x)) return '';
  const ax = Math.abs(x);
  if (ax !== 0 && (ax < 1e-3 || ax >= 1e6)) return x.toPrecision(4).replace(/\.?0+e/, 'e');
  return String(Number(x.toPrecision(sig)));
}

export function fitsLabel(s) {
  const set = new Set((s || '').split(';').filter(Boolean));
  const out = FIT_TOKENS.filter((t) => set.has(t)).map((t) => FIT_LABELS[t]);
  return out.length ? out.join(', ') : 'none';
}

export const blankLabel = (v) => (v === '' || v === null || v === undefined ? '(none)' : String(v));

const esc = (s) =>
  String(s ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');

function wrap(s, width = 70, maxLines = 2) {
  const words = String(s || '').split(/\s+/);
  const lines = [];
  let cur = '';
  for (const w of words) {
    if ((cur + ' ' + w).trim().length > width && cur) {
      lines.push(cur);
      cur = w;
    } else cur = (cur + ' ' + w).trim();
  }
  if (cur) lines.push(cur);
  if (lines.length > maxLines) {
    lines.length = maxLines;
    lines[maxLines - 1] += ' …';
  }
  return lines.map(esc).join('<br>');
}

// Hover text of one point (Plotly's subset of HTML).
export function hoverText(c, i, cites, propLabel, unit) {
  const lines = [];
  const cite = cites?.[c.source_key[i]] || c.source_key[i];
  lines.push(`<b>${esc(cite)}</b>`);
  const v = c.value[i];
  const u = c.u_std[i];
  const val = `${esc(propLabel)} = ${fmtNum(v)}${u != null ? ' ± ' + fmtNum(u) : ''} ${esc(unitPretty(unit))}`;
  lines.push(val);
  const tp = [];
  if (c.T_K[i] != null) tp.push(`T = ${fmtNum(c.T_K[i])} K`);
  if (c.P_MPa[i] != null) tp.push(`P = ${fmtNum(c.P_MPa[i])} MPa`);
  if (tp.length) lines.push(tp.join(', '));
  lines.push(
    [c.data_type[i], c.extraction[i], c.substance[i], c.state[i] || null, c.sample[i] || null]
      .filter(Boolean)
      .map(esc)
      .join(' · ')
  );
  if (c.definition[i]) lines.push(wrap('Definition: ' + c.definition[i], 70, 1));
  if (c.location[i]) lines.push(wrap('Location: ' + c.location[i]));
  lines.push(`QC ${esc(c.qc_flag[i])} · fits: ${esc(fitsLabel(c.used_in_fits[i]))}`);
  lines.push('<i>click for every column</i>');
  return lines.join('<br>');
}
