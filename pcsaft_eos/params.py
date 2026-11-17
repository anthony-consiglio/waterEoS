"""Pure-water PC-SAFT parameters. Lengths are Å and energies divided by kB are K."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Parameters:
    m: float
    sigma: float
    epsilon: float
    h_bond: tuple
    l_bond: tuple = (0.0, 0.0)
    dh: float = 0.0
    ds: float = 0.0
    dp: float = 0.0
    cross_bond: tuple = None


FITS = {
    # Novak et al. (2024), Table II, reference fit of Diamantonis and Economou.
    'pcsaft_water': Parameters(2.1945, 2.229, 141.66, (1804.2, 0.2039)),
    # Novak et al. (2024), Table II. dp is in mol K/m³.
    'novak2024_pcsaft_ts_tv': Parameters(
        2.1145, 2.266, 122.33, (2373.7, 0.0910), (5723.2, 0.00009538),
        -928.9, -5.79352, 278.36e5),
    # Novak et al. (2025), Table 1, temperature-only fit.
    'novak2024_pcsaft_ts_t': Parameters(
        2.2235, 2.207, 130.81, (1811.2, 0.2324), (4045.9, 0.1030),
        -2615.2, -10.769),
    # Novak et al. (2025), Table 3. The unlike bonding parameters are fitted.
    'novak2025_pcsaft_ts_caf': Parameters(
        2.3147, 2.182, 143.03, (1746.0, 0.2388), (3736.7, 0.06212),
        -2967.0, -12.037, cross_bond=(2453.6, 0.4560)),
}
