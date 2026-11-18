import { describe, it, expect } from 'vitest';
import {
  UNITS,
  comp,
  compText,
  decode,
  fileOf,
  geometry,
  inUnit,
  openingStep,
  shortTitle,
  uFor,
} from '../src/h2oAtlas/atlas.js';

const NACL = { M: 58.443, rho: 2.165 };

describe('composition units', () => {
  it('maps wt % to mol %, molality and vol %', () => {
    expect(UNITS.wt.f(10, NACL)).toBe(10);
    expect(UNITS.mol.f(10, NACL)).toBeCloseTo(3.3116, 3);
    expect(UNITS.molal.f(10, NACL)).toBeCloseTo(1.9012, 3);
    expect(UNITS.vol.f(10, NACL)).toBeCloseTo(4.868, 2);
  });

  it('falls back to wt % for vol % without a density', () => {
    expect(uFor({ M: 50 }, 'vol')).toBe('wt');
    expect(uFor(NACL, 'vol')).toBe('vol');
    expect(comp(10, { M: 50 }, 'vol')).toBe('10.0 wt %');
  });

  it('formats compositions and rewrites text in the unit chosen', () => {
    expect(comp(10, NACL, 'wt')).toBe('10.0 wt %');
    expect(comp(10, NACL, 'mol')).toBe('3.31 mol %');
    expect(compText('x = 10 wt %', UNITS.molal, NACL)).toBe('m = 1.90 mol/kg');
    expect(compText('x = %{x:.1f} wt %', UNITS.mol, NACL)).toBe('x = %{x:.2f} mol %');
    expect(compText('x (wt %)', UNITS.vol, NACL)).toBe('x (vol %)');
    expect(compText('no composition here', UNITS.mol, NACL)).toBe('no composition here');
  });
});

describe('figures', () => {
  it('decodes Plotly binary arrays', () => {
    const bytes = new Uint8Array(new Float32Array([1, 2.5]).buffer);
    const bdata = btoa(String.fromCharCode(...bytes));
    expect(Array.from(decode({ dtype: 'f4', bdata }))).toEqual([1, 2.5]);
    expect(decode([1, 2])).toEqual([1, 2]);
  });

  it('redraws a 2-D diagram in mol %: axis, title and a refined tie-line', () => {
    const fig = {
      data: [{ type: 'scatter', mode: 'lines', name: 'L', x: [0, 20], y: [1, 2] }],
      layout: {
        meta: { comp: { axis: 'x' } },
        xaxis: { range: [0, 20], title: { text: 'x (wt %)' } },
        yaxis: {},
      },
    };
    expect(inUnit(fig, 'wt', NACL)).toBe(fig);
    const out = inUnit(fig, 'mol', NACL);
    const top = UNITS.mol.f(20, NACL);
    expect(out.layout.xaxis.range[1]).toBeCloseTo(top, 6);
    expect(out.layout.xaxis.title.text).toBe('x (mol %)');
    const x = out.data[0].x;
    expect(x.length).toBeGreaterThan(2); // the straight line in wt % is a curve in mol %
    expect(x[x.length - 1]).toBeCloseTo(top, 6);
    expect(fig.layout.xaxis.title.text).toBe('x (wt %)'); // the stored figure is untouched
  });
});

describe('views', () => {
  it('names files and pills', () => {
    expect(fileOf({ id: 'd-T-xv', log: 'log v' }, true)).toBe('d-T-xv-log');
    expect(fileOf({ id: 'd-T-xv', log: 'log v' }, false)).toBe('d-T-xv');
    expect(fileOf({ id: 's3-T', log: null }, true)).toBe('s3-T');
    expect(shortTitle('Isothermal · x – v')).toBe('Isothermal x – v');
    expect(shortTitle('Isothermal surface  p(x, v)')).toBe('Isothermal p(x, v)');
  });

  it('fits a 2-D diagram with a square plot and a 3-D view as a square', () => {
    const d = geometry('d-T-xv', 800, 600);
    expect(d.width - d.margin.l - d.margin.r).toBe(d.height - d.margin.t - d.margin.b);
    expect(d.height).toBeLessThanOrEqual(600);
    const s = geometry('s3-T', 800, 600);
    expect(s.width).toBe(s.height);
    expect(s.width).toBe(592);
  });

  it('opens a cold-to-warm temperature slider at -10 °C, others at their own step', () => {
    const steps = (labels) => labels.map((label) => ({ label }));
    const T = [
      '-42 °C',
      '-38 °C',
      '-34 °C',
      '-30 °C',
      '-26 °C',
      '-22 °C',
      '-18 °C',
      '-14 °C',
      '-10 °C',
      '-6 °C',
    ];
    expect(
      openingStep({ sliders: [{ active: 0, currentvalue: { prefix: 'T = ' }, steps: steps(T) }] })
    ).toBe(8);
    const hull = ['-4 °C', '-12 °C', '-20 °C'];
    expect(
      openingStep({
        sliders: [{ active: 0, currentvalue: { prefix: 'T = ' }, steps: steps(hull) }],
      })
    ).toBe(0);
    const p = ['0.1 MPa', '25 MPa'];
    expect(
      openingStep({ sliders: [{ active: 1, currentvalue: { prefix: 'p = ' }, steps: steps(p) }] })
    ).toBe(1);
    expect(openingStep({})).toBe(0);
  });
});
