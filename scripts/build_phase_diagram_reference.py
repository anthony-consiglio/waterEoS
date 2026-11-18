"""Build watereos/data/phase_diagram_reference.json, the measured points of the EoS phase diagram.

    python scripts/build_phase_diagram_reference.py --source <path to fitting_data>

The source is the curated water-property database that also feeds the
Measured Data tab (``python -m watereos_api.measured.load --source ...``).
Its melting and density-maximum rows are filtered as in the EoS fitting
project; the warm response extrema, the 1 atm extrema and Troncoso's
heat-capacity minima are literature values written here.
"""

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from watereos.eos_phase_diagram import nucleation_temperature  # noqa: E402

OUT = ROOT / 'watereos' / 'data' / 'phase_diagram_reference.json'

# (T in K, P in MPa). Negative pressures: digitized from Holten et al. (2017), Fig. 2, warm
# branches of the compressibility minimum and sound-speed maximum. Positive pressures: Fine and
# Millero (1973), Eq. 9, after converting gauge to absolute pressure, and El Hawary and Meier
# (2023), Eq. 3 and Table 3, each sampled inside its published pressure range.
WARM_EXTREMA = {
    'compressibility_min': {
        'property': 'compressibility',
        'T_K': [299.6, 303.7, 307.6, 310.7, 313.3, 315.6, 317.7, 321.2, 322.5, 323.6, 324.3, 324.6],
        'p_MPa': [-135, -120, -100, -80, -60, -40, -20, 20, 40, 60, 80, 100],
        'sources': ['holten2017compressibility', 'fine1973compressibility'],
    },
    'sound_speed_max': {
        'property': 'sound_speed',
        'T_K': [322.4, 327.0, 331.3, 335.4, 339.4, 343.3, 350.9, 354.1, 356.9, 359.4, 361.7],
        'p_MPa': [-120, -100, -80, -60, -40, -20, 20, 40, 60, 80, 100],
        'sources': ['holten2017compressibility', 'hawary2023highly'],
    },
}
# Minimum of Kell (1975), Eq. 20, and maximum of Marczak (1997), at 1 atm.
AMBIENT_EXTREMA = {
    'compressibility_min': {'property': 'compressibility', 'T_K': 319.665, 'p_MPa': 0.101325,
                            'sources': ['kell1975density']},
    'sound_speed_max': {'property': 'sound_speed', 'T_K': 347.324, 'p_MPa': 0.101325,
                        'sources': ['marczak1997water']},
}
# Troncoso (2017), Eq. 10 with the water coefficients, as plotted by Caupin and Anisimov (2019),
# Figs. 12-13. The coldest isotherm is below the measured temperature range.
HEAT_CAPACITY_MINIMA = {
    'T_K': [253.15, 263.15, 273.15, 283.15],
    'p_MPa': [261.50, 290.74, 351.42, 392.34],
    'in_measured_range': [False, True, True, True],
    'sources': ['troncoso2017isobaric'],
}
# Citations not in the database's sources table.
EXTRA_SOURCES = {
    'fine1973compressibility': ('R. A. Fine, F. J. Millero (1973). Compressibility of water as a function of '
                                'temperature and pressure. J. Chem. Phys. 59, 5529.', '10.1063/1.1679903'),
    'marczak1997water': ('W. Marczak (1997). Water as a standard in the measurements of speed of sound in '
                         'liquids. J. Acoust. Soc. Am. 102, 2776.', '10.1121/1.420332'),
}
HIGH_PRESSURE_SOURCES = ('mishima2021liquidphase', 'bridgman1912water')


def read(path):
    with open(path, newline='') as fp:
        return list(csv.DictReader(fp))


def points(rows):
    rows = sorted(rows, key=lambda r: (r['source_key'], float(r['T_K']), float(r['P_MPa'])))
    return {'T_K': [round(float(r['T_K']), 4) for r in rows],
            'p_MPa': [round(float(r['P_MPa']), 4) for r in rows],
            'source_key': [r['source_key'] for r in rows]}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--source', required=True, type=Path, help='the fitting_data folder')
    args = parser.parse_args()

    melting = read(args.source / 'phase_equilibrium' / 'melting.csv')
    ih = [r for r in melting
          if r['substance'] == 'H2O' and r['ice_phase'] == 'Ih' and r['data_type'] in ('measured', 'derived')
          and r['qc_flag'] != 'outlier' and -140.0 <= float(r['P_MPa']) <= 400.0]
    # Melting points below the homogeneous nucleation line are not plotted.
    t_h = nucleation_temperature([float(r['P_MPa']) for r in ih])
    ih = [r for r, th in zip(ih, t_h) if float(r['T_K']) >= th]
    high = {name: [r for r in melting
                   if r['substance'] == 'H2O' and r['ice_phase'] == name and r['data_type'] == 'measured'
                   and r['source_key'] in HIGH_PRESSURE_SOURCES and r['qc_flag'] != 'outlier'
                   and float(r['P_MPa']) <= 420.0]
            for name in ('III', 'V')}
    tmd = [r for r in read(args.source / 'liquid' / 'tmd.csv')
           if r['substance'] == 'H2O' and r['data_type'] in ('measured', 'derived')]

    data = {
        'melting': {'Ih': points(ih), 'III': points(high['III']), 'V': points(high['V'])},
        'density_maxima': points(tmd),
        'warm_extrema': WARM_EXTREMA,
        'ambient_extrema': AMBIENT_EXTREMA,
        'heat_capacity_minima': HEAT_CAPACITY_MINIMA,
    }
    used = set()
    for group in (data['melting'].values(), [data['density_maxima']]):
        for entry in group:
            used.update(entry['source_key'])
    for entry in [*WARM_EXTREMA.values(), *AMBIENT_EXTREMA.values(), HEAT_CAPACITY_MINIMA]:
        used.update(entry['sources'])
    table = {r['source_key']: (r['citation'], r['doi']) for r in read(args.source / 'sources.csv')}
    table.update(EXTRA_SOURCES)
    missing = sorted(used - set(table))
    if missing:
        raise SystemExit(f'no citation for {missing}')
    data['sources'] = {key: {'citation': table[key][0], 'doi': table[key][1] or None} for key in sorted(used)}
    data['about'] = ('Measured points for watereos.eos_phase_diagram, built by '
                     'scripts/build_phase_diagram_reference.py from the curated water-property database. '
                     'Ice Ih: measured or derived H2O melting points from -140 to 400 MPa, at or above the '
                     'homogeneous nucleation line. Ices III and V: measured melting points of Mishima (2021) and '
                     'Bridgman (1912) up to 420 MPa. Density maxima: measured or derived H2O rows. '
                     'T in K, p in MPa.')

    OUT.write_text(json.dumps(data, indent=1, ensure_ascii=False) + '\n')
    counts = {name: len(entry['T_K']) for name, entry in data['melting'].items()}
    print(f'wrote {OUT.relative_to(ROOT)}: melting {counts}, density maxima {len(tmd)}, '
          f'{len(data["sources"])} sources', flush=True)


if __name__ == '__main__':
    main()
