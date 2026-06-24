"""Extract SeaFreeze GLBF spline coefficients into a Rust-friendly binary.

Run once per SeaFreeze upgrade to regenerate the bundled spline file.
The output is consumed at runtime by watereos_rs::seafreeze.

Binary layout (all little-endian):
    HEADER
      magic        4 bytes   b"WSF1"
      n_phases     u16
    PER PHASE
      name_len     u8        bytes for UTF-8 phase name
      name         name_len bytes
      order_P      u8        polynomial order in P (e.g. 6 means quintic)
      order_T      u8        polynomial order in T
      n_knots_P    u32       knot count along P
      n_knots_T    u32       knot count along T
      n_basis_P    u32       basis-function count along P
      n_basis_T    u32       basis-function count along T
      has_shear    u8        1 if shear_mod params follow, else 0
      knots_P      n_knots_P x f64
      knots_T      n_knots_T x f64
      coefs        n_basis_P x n_basis_T x f64 (row-major, P-axis first)
      shear_mod    6 x f64   (only if has_shear == 1)

Usage:
    python watereos/data/extract_seafreeze_splines.py
    -> writes watereos/data/seafreeze_splines.bin
"""

import struct
import warnings
from pathlib import Path

warnings.simplefilter("ignore")

# Phases waterEoS uses (subset of SeaFreeze's full catalogue).
PHASES_TO_EXTRACT = [
    "water1",
    "water_IAPWS95",
    "Ih",
    "II",
    "III",
    "V",
    "VI",
    "VII_X_French",
]


def main():
    from mlbspline.load import loadSpline
    from seafreeze.seafreeze import defpath, phases as sf_phases

    out_path = Path(__file__).parent / "seafreeze_splines.bin"
    buf = bytearray()

    buf.extend(b"WSF1")
    buf.extend(struct.pack("<H", len(PHASES_TO_EXTRACT)))

    for name in PHASES_TO_EXTRACT:
        descriptor = sf_phases[name]
        sp = loadSpline(defpath, descriptor.sp_name)

        assert sp["number"].size == 2, f"{name}: not a 2D spline"
        n_basis_P, n_basis_T = int(sp["number"][0]), int(sp["number"][1])
        order_P, order_T = int(sp["order"][0]), int(sp["order"][1])
        knots_P = sp["knots"][0].astype("<f8")
        knots_T = sp["knots"][1].astype("<f8")
        coefs = sp["coefs"].astype("<f8")
        assert knots_P.size == n_basis_P + order_P
        assert knots_T.size == n_basis_T + order_T
        assert coefs.shape == (n_basis_P, n_basis_T)

        has_shear = 1 if descriptor.shear_mod_parms else 0

        name_bytes = name.encode("utf-8")
        assert len(name_bytes) < 256

        buf.append(len(name_bytes))
        buf.extend(name_bytes)
        buf.append(order_P)
        buf.append(order_T)
        buf.extend(struct.pack("<I", knots_P.size))
        buf.extend(struct.pack("<I", knots_T.size))
        buf.extend(struct.pack("<I", n_basis_P))
        buf.extend(struct.pack("<I", n_basis_T))
        buf.append(has_shear)
        buf.extend(knots_P.tobytes(order="C"))
        buf.extend(knots_T.tobytes(order="C"))
        buf.extend(coefs.tobytes(order="C"))
        if has_shear:
            shear = [float(x) for x in descriptor.shear_mod_parms]
            assert len(shear) == 6
            buf.extend(struct.pack("<6d", *shear))

        print(f"  {name:<15} order=({order_P},{order_T}) "
              f"basis=({n_basis_P},{n_basis_T}) "
              f"knots=({knots_P.size},{knots_T.size}) "
              f"shear={'yes' if has_shear else 'no'}")

    out_path.write_bytes(bytes(buf))
    print(f"\nWrote {out_path} ({len(buf):,} bytes)")


if __name__ == "__main__":
    main()
