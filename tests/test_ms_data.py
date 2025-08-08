import os
import numpy as np
import pytest

from orthogonal_projection import __name__  # ensure package importable
from orthogonal_projection import dimensionality_reduction as dr  # reuse utilities

# Import the module without requiring pyteomics at import time
import orthogonal_projection.ms_data as msd


def _pyteomics_available():
    return getattr(msd, "_PYTEOMICS_AVAILABLE", False)


data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', '20150731_QEp7_MiBa_SA_SKOV3-1.mzML')


@pytest.mark.skipif(not _pyteomics_available(), reason="pyteomics not installed")
@pytest.mark.skipif(not os.path.exists(data_path), reason="mzML test data not found")
def test_build_matrix_shape_and_normalization():
    # Use a small number of spectra to keep the test fast
    X, rts, mz_bins, meta = msd.parse_mzml_build_matrix(data_path, bin_size=0.1, max_spectra=10, ms_level=1)

    # Shape checks
    assert X.ndim == 2
    assert X.shape[0] == len(rts)
    assert X.shape[1] == len(mz_bins)
    assert X.shape[0] > 0 and X.shape[1] > 0

    # Normalization: rows should sum to ~1 where there is signal
    row_sums = X.sum(axis=1)
    nonzero = row_sums > 0
    if np.any(nonzero):
        assert np.allclose(row_sums[nonzero], 1.0, rtol=1e-6, atol=1e-6)

    # No NaNs or infs
    assert not np.isnan(X).any()
    assert not np.isinf(X).any()

    # Retention times should be monotonic or at least finite
    assert np.all(np.isfinite(rts))


@pytest.mark.skipif(not _pyteomics_available(), reason="pyteomics not installed")
@pytest.mark.skipif(not os.path.exists(data_path), reason="mzML test data not found")
def test_bin_range_and_metadata():
    # Explicit m/z range
    X, rts, mz_bins, meta = msd.parse_mzml_build_matrix(data_path, bin_size=0.2, mz_min=300, mz_max=1200, max_spectra=5)
    assert X.shape[1] == len(mz_bins)
    assert meta and 'mz_bin_edges' in meta and 'frame_indices' in meta
    edges = meta['mz_bin_edges']
    assert np.isclose(mz_bins[0], (edges[0] + edges[1]) / 2.0)
