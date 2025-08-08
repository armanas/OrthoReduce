import numpy as np
from typing import Dict, List, Tuple


def compute_fingerprints(X: np.ndarray, labels: List[str]) -> Tuple[np.ndarray, List[str]]:
    labels = np.asarray(labels)
    uniq = sorted(set(labels.tolist()))
    F = []
    for u in uniq:
        F.append(X[labels == u].mean(axis=0))
    return np.vstack(F), uniq


def cosine_similarity_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    A_norm = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    B_norm = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
    return A_norm @ B_norm.T


def match_frames_to_peptides(X: np.ndarray, labels: List[str], rts: np.ndarray,
                             rt_by_pep: Dict[str, float], rt_tolerance: float = 0.5) -> Dict[str, float]:
    labels = np.asarray(labels)
    uniq = sorted(rt_by_pep.keys())
    F, uniq_labels = compute_fingerprints(X, labels)
    # map uniq order to rt_by_pep order
    order = [uniq_labels.index(u) for u in uniq]
    F = F[order]
    S = cosine_similarity_matrix(X, F)  # n x p
    pred_idx = S.argmax(axis=1)
    pred_labels = [uniq[i] for i in pred_idx]

    # accuracy with RT prior
    correct = 0
    correct_time_only = 0
    for i, (pl, true_l) in enumerate(zip(pred_labels, labels)):
        # RT gating
        if abs(rts[i] - rt_by_pep[pl]) <= rt_tolerance:
            if pl == true_l:
                correct += 1
        # time-only correctness: choose peptide with closest RT
        closest = min(uniq, key=lambda u: abs(rts[i] - rt_by_pep[u]))
        if closest == true_l:
            correct_time_only += 1
    return {
        'accuracy': correct / len(X),
        'time_only_accuracy': correct_time_only / len(X),
    }
