"""
Minimal mzML parsing utilities with lazy pyteomics import so that tests can skip
when unavailable.
"""
from typing import Tuple, Dict, Any
import numpy as np

try:
    from pyteomics import mzml  # type: ignore
    _PYTEOMICS_AVAILABLE = True
except Exception:
    mzml = None  # type: ignore
    _PYTEOMICS_AVAILABLE = False


def parse_mzml_build_matrix(path: str, bin_size: float = 0.1, mz_min: float | None = None,
                             mz_max: float | None = None, max_spectra: int | None = None,
                             ms_level: int = 1) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    if not _PYTEOMICS_AVAILABLE:
        raise ImportError("pyteomics not available")

    rts = []
    spectra = []
    with mzml.MzML(path) as reader:  # type: ignore
        for i, spec in enumerate(reader):
            if max_spectra is not None and i >= max_spectra:
                break
            if ms_level is not None and int(spec.get('ms level', 1)) != ms_level:
                continue
            mz = np.asarray(spec.get('m/z array', []), dtype=float)
            inten = np.asarray(spec.get('intensity array', []), dtype=float)
            if mz_min is None:
                mz_min = float(mz.min()) if mz.size else 0.0
            if mz_max is None:
                mz_max = float(mz.max()) if mz.size else 2000.0
            if mz.size == 0:
                continue
            rts.append(float(spec.get('scanList', {}).get('scan', [{}])[0].get('scan start time', 0.0)))
            spectra.append((mz, inten))
    if not spectra:
        return np.zeros((0, 0)), np.asarray(rts), np.zeros(0), {}

    # build bins
    edges = np.arange(mz_min, mz_max + bin_size, bin_size)
    bins = 0.5 * (edges[:-1] + edges[1:])
    X = np.zeros((len(spectra), len(bins)), dtype=float)
    for i, (mz, inten) in enumerate(spectra):
        idx = np.clip(np.digitize(mz, edges) - 1, 0, len(bins) - 1)
        np.add.at(X[i], idx, inten)
        s = X[i].sum()
        if s > 0:
            X[i] /= s
    meta = {
        'mz_bin_edges': edges,
        'frame_indices': np.arange(len(spectra)),
    }
    return X, np.asarray(rts), bins, meta
