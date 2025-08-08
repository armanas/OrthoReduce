"""
ms_data.py - mzML parsing, binning, and TIC normalization utilities.

Phase 1: Data preparation for mass spectrometry frames.

Functions:
- parse_mzml_build_matrix: Parse mzML to a (frames, bins) intensity matrix with fixed m/z binning and TIC normalization.

Notes:
- Uses pyteomics.mzml.MzML iterator. Only MS1 spectra are considered by default.
- Supports limiting the number of spectra via max_spectra for testing/performance.
- Returns retention times in minutes when available; otherwise seconds are converted to minutes.
"""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from pyteomics import mzml
    _PYTEOMICS_AVAILABLE = True
except Exception:
    _PYTEOMICS_AVAILABLE = False


def _extract_rt_minutes(spectrum: dict) -> Optional[float]:
    """Extract retention time (in minutes) from a pyteomics spectrum dict.

    Attempts multiple keys and unit annotations commonly present in mzML.
    """
    # Common location
    try:
        scan = spectrum.get('scanList', {}).get('scan', [{}])[0]
        rt = scan.get('scan start time', None)
        if rt is None:
            # Alternate CV param id
            rt = scan.get('MS:1000016', None)
        # Unit handling
        unit = None
        if isinstance(rt, dict):
            # Sometimes encoded as {'value': x, 'unitName': 'minute'}
            unit = rt.get('unitName') or rt.get('unit')
            val = rt.get('value')
        else:
            val = rt
        if val is None:
            return None
        if unit is None:
            # Heuristic: values > 200 likely seconds; typical LC gradients are < 120 min
            # Many mzML via pyteomics already convert to minutes; keep as-is if small.
            if val > 200:
                return float(val) / 60.0
            return float(val)
        unit = str(unit).lower()
        if 'min' in unit:
            return float(val)
        if 'sec' in unit or 's' == unit:
            return float(val) / 60.0
        return float(val)
    except Exception:
        return None


def parse_mzml_build_matrix(
    file_path: str,
    bin_size: float = 0.1,
    mz_min: Optional[float] = None,
    mz_max: Optional[float] = None,
    max_spectra: Optional[int] = None,
    ms_level: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """Parse an mzML file and construct a binned, TIC-normalized intensity matrix.

    Parameters
    ----------
    file_path : str
        Path to the mzML file.
    bin_size : float
        Fixed m/z bin width. Default 0.1 Th.
    mz_min, mz_max : Optional[float]
        If provided, use this m/z window; otherwise inferred from data.
    max_spectra : Optional[int]
        If provided, stop after reading this many spectra (useful for tests).
    ms_level : int
        MS level to include (default 1).

    Returns
    -------
    X : np.ndarray
        Intensity matrix of shape (frames, bins), TIC-normalized (row sums ≈ 1).
    rts : np.ndarray
        Retention times (minutes), length = frames.
    mz_bin_centers : np.ndarray
        Centers of the m/z bins.
    meta : dict
        Additional metadata: {'frame_indices', 'mz_bin_edges'}
    """
    if not _PYTEOMICS_AVAILABLE:
        raise ImportError("pyteomics is required to parse mzML files. Please install pyteomics.")

    spectra_cache: List[Tuple[np.ndarray, np.ndarray, Optional[float]]] = []
    inferred_min, inferred_max = np.inf, -np.inf

    count = 0
    with mzml.MzML(file_path) as reader:
        for spec in reader:
            if max_spectra is not None and count >= max_spectra:
                break
            # Filter by MS level if present
            if ms_level is not None:
                try:
                    lvl = spec.get('ms level') or spec.get('msLevel')
                    if lvl is not None and int(lvl) != int(ms_level):
                        continue
                except Exception:
                    pass
            mz_arr = spec.get('m/z array')
            int_arr = spec.get('intensity array')
            if mz_arr is None or int_arr is None:
                continue
            if len(mz_arr) == 0 or len(int_arr) == 0:
                continue
            # Ensure numpy arrays
            mz_arr = np.asarray(mz_arr, dtype=float)
            int_arr = np.asarray(int_arr, dtype=float)
            # Retention time in minutes
            rt_min = _extract_rt_minutes(spec)

            spectra_cache.append((mz_arr, int_arr, rt_min))
            # Update inferred m/z range (ignore NaNs/Infs)
            local_min = np.nanmin(mz_arr)
            local_max = np.nanmax(mz_arr)
            if np.isfinite(local_min):
                inferred_min = min(inferred_min, local_min)
            if np.isfinite(local_max):
                inferred_max = max(inferred_max, local_max)
            count += 1

    if count == 0:
        raise ValueError("No spectra read from mzML. Check file path and ms_level.")

    # Establish m/z range
    if mz_min is None:
        mz_min = float(np.floor(inferred_min))
    if mz_max is None:
        mz_max = float(np.ceil(inferred_max))
    if mz_max <= mz_min:
        raise ValueError(f"Invalid m/z range: [{mz_min}, {mz_max}]")

    # Build bin edges and centers
    n_bins = int(np.ceil((mz_max - mz_min) / bin_size))
    mz_edges = mz_min + np.arange(n_bins + 1) * bin_size
    mz_centers = mz_edges[:-1] + 0.5 * bin_size

    n_frames = len(spectra_cache)
    X = np.zeros((n_frames, n_bins), dtype=float)
    rts = np.full(n_frames, np.nan, dtype=float)

    eps = 1e-12
    for i, (mz_arr, int_arr, rt_min) in enumerate(spectra_cache):
        # Histogram with weights for intensities
        hist, _ = np.histogram(mz_arr, bins=mz_edges, weights=int_arr)
        # TIC normalization
        s = float(hist.sum())
        if s > 0:
            X[i, :] = hist / (s + eps)
        else:
            X[i, :] = 0.0
        if rt_min is not None:
            rts[i] = rt_min

    # Replace NaNs/Infs if any
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # If any RTs missing, attempt simple imputation by monotonic fill
    if np.any(~np.isfinite(rts)):
        # Use linear spacing as fallback
        valid = np.isfinite(rts)
        if valid.any():
            # Interpolate missing
            idx = np.arange(n_frames)
            rts = np.interp(idx, idx[valid], rts[valid])
        else:
            rts = np.linspace(0, n_frames * 0.01, n_frames)  # 0.6 sec per frame ≈ 0.01 min

    meta = {
        'frame_indices': np.arange(n_frames),
        'mz_bin_edges': mz_edges,
    }
    return X, rts, mz_centers, meta
