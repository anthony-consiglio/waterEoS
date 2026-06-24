"""Precompute phase diagram data for all two-state models.

Generates JSON files in watereos/data/ that are loaded at runtime
instead of computing on the fly.

Usage:
    python scripts/precompute_phase_diagrams.py
"""

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from watereos.computation import compute_phase_diagram_data

MODELS = ['holten2014', 'caupin2019', 'duska2020']
DATA_DIR = Path(__file__).resolve().parent.parent / 'watereos' / 'data'


def _serialize(obj):
    """Convert numpy types to JSON-serializable Python types."""
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    return obj


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    warnings.filterwarnings('ignore')

    for model_key in MODELS:
        print(f'Computing {model_key}...', end=' ', flush=True)
        t0 = time.time()
        try:
            data = compute_phase_diagram_data(model_key)
            serialized = _serialize(data)

            out_path = DATA_DIR / f'{model_key}_phase_diagram.json'
            with open(out_path, 'w') as f:
                json.dump(serialized, f, separators=(',', ':'))

            size_kb = out_path.stat().st_size / 1024
            elapsed = time.time() - t0
            print(f'done ({elapsed:.1f}s, {size_kb:.0f} KB)')
        except Exception as e:
            print(f'FAILED: {e}')
            raise


if __name__ == '__main__':
    main()
