"""
fingerprinting.py - Peptide fingerprinting and frame matching utilities.

Phase 4: Compute peptide fingerprints and assign frames via cosine similarity.

Key functions:
- compute_fingerprints(embeddings, peptide_ids): per-peptide average embedding.
- cosine_similarity_matrix(A, B): pairwise cosine similarities.
- match_frames_to_peptides(embeddings, peptide_ids, rts, rt_by_peptide, rt_tolerance=0.5):
    returns assignments and accuracy metrics.

Notes:
- Embeddings are expected to be the convex embeddings (e.g., after convex hull projection),
  but the functions are agnostic and will work with any vector representation.
- If peptide_ids is None or missing, the caller can provide synthetic labels; this module
  does not infer labels from mzML.
"""
from __future__ import annotations
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


def compute_fingerprints(embeddings: np.ndarray, peptide_ids: Iterable) -> Tuple[np.ndarray, List]:
    """Compute per-peptide fingerprints by averaging embeddings across frames.

    Parameters
    ----------
    embeddings : (n_frames, d) array
    peptide_ids : iterable of length n_frames (hashable peptide identifiers)

    Returns
    -------
    F : (n_peptides, d) array of per-peptide fingerprints
    labels : list of peptide labels corresponding to rows in F
    """
    embeddings = np.asarray(embeddings, dtype=float)
    peptide_ids = list(peptide_ids)
    assert embeddings.shape[0] == len(peptide_ids), "embeddings and peptide_ids must align"

    # Group by peptide id
    uniq = []
    idx_by_label: Dict = {}
    for i, pid in enumerate(peptide_ids):
        if pid not in idx_by_label:
            idx_by_label[pid] = []
            uniq.append(pid)
        idx_by_label[pid].append(i)

    F = np.zeros((len(uniq), embeddings.shape[1]), dtype=float)
    for j, pid in enumerate(uniq):
        idxs = idx_by_label[pid]
        F[j] = embeddings[idxs].mean(axis=0)

    # Normalize fingerprints to unit length for cosine similarity robustness
    norms = np.linalg.norm(F, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    F = F / norms
    return F, uniq


def cosine_similarity_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Compute pairwise cosine similarities between rows of A and B."""
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    A_norm = A / np.clip(np.linalg.norm(A, axis=1, keepdims=True), 1e-12, None)
    B_norm = B / np.clip(np.linalg.norm(B, axis=1, keepdims=True), 1e-12, None)
    return A_norm @ B_norm.T


def match_frames_to_peptides(
    embeddings: np.ndarray,
    peptide_ids: Iterable,
    rts: np.ndarray,
    rt_by_peptide: Dict,
    rt_tolerance: float = 0.5,
) -> Dict[str, object]:
    """Match each frame to the closest peptide fingerprint and evaluate accuracy.

    Rules:
    - Top-1 cosine similarity assignment to a peptide fingerprint.
    - A frame is considered correctly assigned if the top-matching peptide's
      retention time is within ±rt_tolerance minutes of the ground-truth peptide's
      retention time, AND the predicted label equals the ground-truth label.
      We also report a relaxed time-only accuracy for reference.

    Parameters
    ----------
    embeddings : (n_frames, d)
    peptide_ids : iterable of true peptide ids per frame
    rts : (n_frames,) retention times (minutes)
    rt_by_peptide : dict mapping peptide id -> canonical retention time (minutes)
    rt_tolerance : float, minutes

    Returns
    -------
    dict with keys:
      - 'predicted_ids': list
      - 'scores': np.ndarray of cosine similarities
      - 'top_indices': np.ndarray of predicted fingerprint indices
      - 'labels': list of fingerprint labels in order
      - 'accuracy': float (strict: label match AND time-within-window)
      - 'time_only_accuracy': float (predicted peptide's RT within tolerance of true peptide RT)
    """
    embeddings = np.asarray(embeddings, dtype=float)
    rts = np.asarray(rts, dtype=float)
    peptide_ids = list(peptide_ids)

    F, labels = compute_fingerprints(embeddings, peptide_ids)
    S = cosine_similarity_matrix(embeddings, F)
    top_idx = np.argmax(S, axis=1)
    scores = S[np.arange(S.shape[0]), top_idx]
    preds = [labels[i] for i in top_idx]

    # Accuracies
    time_ok = []
    strict_ok = []
    for i, true_pid in enumerate(peptide_ids):
        pred_pid = preds[i]
        true_rt = rt_by_peptide.get(true_pid, np.nan)
        pred_rt = rt_by_peptide.get(pred_pid, np.nan)
        within = np.isfinite(true_rt) and np.isfinite(pred_rt) and (abs(pred_rt - true_rt) <= rt_tolerance)
        time_ok.append(within)
        strict_ok.append(within and (pred_pid == true_pid))

    time_only_acc = float(np.mean(time_ok)) if len(time_ok) else 0.0
    strict_acc = float(np.mean(strict_ok)) if len(strict_ok) else 0.0

    return {
        'predicted_ids': preds,
        'scores': scores,
        'top_indices': top_idx,
        'labels': labels,
        'accuracy': strict_acc,
        'time_only_accuracy': time_only_acc,
    }
