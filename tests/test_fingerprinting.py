import numpy as np
from orthogonal_projection.fingerprinting import (
    compute_fingerprints,
    cosine_similarity_matrix,
    match_frames_to_peptides,
)


def test_compute_fingerprints_shapes():
    rng = np.random.default_rng(0)
    n_per = 10
    d = 8
    centers = {
        'PEP1': rng.standard_normal(d),
        'PEP2': rng.standard_normal(d),
        'PEP3': rng.standard_normal(d),
    }
    embeddings = []
    labels = []
    rts = []
    rt_by_pep = {'PEP1': 10.0, 'PEP2': 15.0, 'PEP3': 20.0}
    for pid, c in centers.items():
        for _ in range(n_per):
            embeddings.append(c + 0.05 * rng.standard_normal(d))
            labels.append(pid)
            rts.append(rt_by_pep[pid] + 0.05 * rng.standard_normal())
    X = np.vstack(embeddings)
    rts = np.asarray(rts)

    F, uniq = compute_fingerprints(X, labels)
    assert F.shape == (len(centers), d)
    assert set(uniq) == set(centers.keys())

    # Cosine similarities should be higher on-diagonal for fingerprints
    S = cosine_similarity_matrix(F, F)
    offdiag = (np.eye(len(uniq)) == 0)
    assert np.all(np.diag(S) >= S[offdiag].reshape(len(uniq), -1).max(axis=1) - 1e-6)

    res = match_frames_to_peptides(X, labels, rts, rt_by_pep, rt_tolerance=0.5)
    assert 'accuracy' in res and 'time_only_accuracy' in res
    # Because data are well-separated and RTs consistent, expect high accuracy
    assert res['accuracy'] >= 0.7
