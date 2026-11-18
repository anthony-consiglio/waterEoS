// Layer keys of watereos.eos_phase_diagram.LAYERS, grouped for the sidebar.
export const OVERLAY_GROUPS = [
  {
    label: 'Liquid–liquid transition',
    items: [
      { key: 'binodal', label: 'Coexistence line' },
      { key: 'LLCP', label: 'Critical point' },
      { key: 'hdl_spinodal', label: 'HDL stability limit' },
      { key: 'ldl_spinodal', label: 'LDL stability limit' },
      { key: 'widom', label: 'Widom line' },
    ],
  },
  {
    label: 'Boundaries',
    items: [
      { key: 'lv_spinodal', label: 'Liquid–vapor stability limit' },
      { key: 'vapor_liquid', label: 'Vapor–liquid coexistence' },
      { key: 'nucleation', label: 'Homogeneous nucleation' },
      { key: 'ice_ih', label: 'Ice Ih melting' },
      { key: 'ice_iii', label: 'Ice III melting' },
      { key: 'ice_v', label: 'Ice V melting' },
      { key: 'kauzmann', label: 'Kauzmann line' },
    ],
  },
  {
    label: 'Response extrema',
    items: [
      { key: 'density', label: 'Density' },
      { key: 'compressibility', label: 'Compressibility' },
      { key: 'sound_speed', label: 'Sound speed' },
      { key: 'heat_capacity', label: 'Heat capacity' },
    ],
  },
  {
    label: 'Measured points',
    items: [
      { key: 'tmd_data', label: 'Density maxima' },
      { key: 'warm_extrema', label: 'Warm extrema' },
      { key: 'cp_minima', label: 'Troncoso Cp minima' },
    ],
  },
];

// Everything but the Kauzmann line, as in watereos.eos_phase_diagram.DEFAULT_LAYERS.
export const DEFAULT_SHOW = OVERLAY_GROUPS.flatMap((g) => g.items.map((o) => o.key)).filter(
  (k) => k !== 'kauzmann'
);

// Standard window of the diagram; curves reach 100 K and 400 MPa where the model has them.
export const DEFAULT_RANGES = { Tmin: 100, Tmax: 375, Pmin: -400, Pmax: 420 };
