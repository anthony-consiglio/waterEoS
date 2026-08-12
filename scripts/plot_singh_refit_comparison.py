"""
Comparison figure for the Singh (2017) transport model refits.

Three panels (eta, D, tau_r vs pressure) showing the experimental data along
selected isotherms together with the fitted model curves for the three
two-state backbones (holten2014 solid, caupin2019 dashed, duska2020 dotted).

Output: references/data/singh_refit_comparison.png
"""

import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import refit_singh_transport as R
from singh_viscosity.core import compute_batch

plt.rcParams['font.family'] = ['Helvetica', 'Arial', 'sans-serif']

BACKBONES = ['holten2014', 'caupin2019', 'duska2020']
STYLES = {'holten2014': '-', 'caupin2019': '--', 'duska2020': ':'}
PROP_KEY = {'eta': 'eta', 'D': 'D', 'tau_r': 'tau_r'}
SCALE = {'eta': 1e3, 'D': 1e9, 'tau_r': 1e12}
YLABEL = {'eta': r'$\eta$ (mPa s)', 'D': r'$D$ (10$^{-9}$ m$^2$/s)',
          'tau_r': r'$\tau_r$ (ps)'}
TITLE = {'eta': 'Viscosity', 'D': 'Self-diffusion', 'tau_r': 'Rotational time'}
ISOTHERMS = {'eta': [297.8, 272.8, 252.8, 244.3],
             'D': [273.0, 248.0, 223.0, 208.0],
             'tau_r': [273.0, 258.0, 248.0, 238.0]}
T_TOL = 0.35   # K, matching tolerance data <-> isotherm


def main():
    data = {'eta': list(map(np.asarray, R.load_eta())),
            'D': list(map(np.asarray, R.load_D())),
            'tau_r': list(map(np.asarray, R.load_tau_r()))}

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6))
    cmap = plt.get_cmap('viridis')

    for ax, prop in zip(axes, ('eta', 'D', 'tau_r')):
        T, P, y, s = data[prop]
        temps = ISOTHERMS[prop]
        colors = [cmap(v) for v in np.linspace(0.05, 0.8, len(temps))]

        for Ti, ci in zip(temps, colors):
            m = np.abs(T - Ti) < T_TOL
            ax.errorbar(P[m], y[m] * SCALE[prop], yerr=s[m] * SCALE[prop],
                        fmt='o', ms=3.5, lw=0, elinewidth=0.8, capsize=1.5,
                        color=ci, zorder=3)
            if not m.any():
                continue
            # Draw model curves only over the data span of the isotherm:
            # at low T the backbones cross their liquid-liquid transition
            # at low P (a region with no liquid data), where the fraction
            # -- and hence the model curve -- jumps branch unphysically.
            Pg = np.linspace(max(0.1, P[m].min() - 15.0), P[m].max() + 15.0, 120)
            Tg = np.full_like(Pg, Ti)
            for b in BACKBONES:
                out = compute_batch(Tg, Pg, backbone=b)
                ax.plot(Pg, out[PROP_KEY[prop]] * SCALE[prop], STYLES[b],
                        color=ci, lw=1.1, zorder=2)
            ax.annotate(f'{Ti:g} K', xy=(1.01, (y[m] * SCALE[prop])[np.argmax(P[m])]
                                          if m.any() else 1),
                        xycoords=('axes fraction', 'data'),
                        fontsize=7, color=ci, va='center')

        ax.set_yscale('log')
        ax.set_xlabel('Pressure (MPa)')
        ax.set_ylabel(YLABEL[prop])
        ax.set_title(TITLE[prop], fontsize=11)
        ax.grid(alpha=0.2)
        ax.set_box_aspect(1)

    handles = [plt.Line2D([], [], color='k', ls=STYLES[b], lw=1.2,
                          label={'holten2014': 'Holten 2014 (published params)',
                                 'caupin2019': 'Caupin 2019 (refit)',
                                 'duska2020': 'Duska 2020 (refit)'}[b])
               for b in BACKBONES]
    handles.append(plt.Line2D([], [], color='0.4', marker='o', ls='', ms=4,
                              label='experimental data'))
    fig.legend(handles=handles, loc='lower center', ncol=4, fontsize=8.5,
               frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle('Singh et al. (2017) two-state transport model: '
                 'backbone comparison', fontsize=12)
    fig.tight_layout(rect=(0, 0.02, 1, 1))

    out_path = os.path.join(R.DATA_DIR, 'singh_refit_comparison.png')
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    print('written', out_path)


if __name__ == '__main__':
    main()
