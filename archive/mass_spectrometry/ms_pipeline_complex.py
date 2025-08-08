#!/usr/bin/env python3
"""
ms_pipeline.py - End-to-end MS data pipeline (Phases 1–5)

Phase 0: Assumes environment is prepared and dataset exists under /data.
Phase 1: Parse mzML, fixed m/z binning (0.1 Th), TIC normalization.
Phase 2: JL target dimension from ceil(4 ln n / eps^2), capped; apply GaussianRandomProjection.
Phase 3: Convex hull projection via constrained optimization.
Phase 4: Peptide fingerprinting & matching (if labels provided) with cosine similarity.
Phase 5: Results table and plots saved to /results.

Usage:
  python ms_pipeline.py --mzml data/20150731_QEp7_MiBa_SA_SKOV3-1.mzML --epsilon 0.3 --max_spectra 500 --labels_csv labels.csv

If --labels_csv is provided, it must contain columns: frame_index, peptide_id, peptide_rt (minutes).
"""
import os
import argparse
import json
from typing import Optional, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.random_projection import GaussianRandomProjection
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances

from orthogonal_projection.ms_data import parse_mzml_build_matrix
from orthogonal_projection.projection import jll_dimension
from orthogonal_projection.evaluation import compute_distortion
from orthogonal_projection.dimensionality_reduction import rank_corr
from orthogonal_projection.convex_optimized import project_onto_convex_hull_qp
from orthogonal_projection.fingerprinting import match_frames_to_peptides, compute_fingerprints, cosine_similarity_matrix


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def plot_distortion_vs_epsilons(plots_dir: str, X: np.ndarray, epsilons):
    ensure_dir(plots_dir)
    means = []
    ks = []
    for eps in epsilons:
        k = min(jll_dimension(X.shape[0], eps), X.shape[1])
        grp = GaussianRandomProjection(n_components=int(k), random_state=0)
        Y = grp.fit_transform(X)
        m, _, _, _ = compute_distortion(X, Y)
        means.append(m)
        ks.append(int(k))
    plt.figure(figsize=(6, 4))
    plt.plot(epsilons, means, marker='o')
    plt.xlabel('epsilon')
    plt.ylabel('mean distortion')
    plt.title('Distortion vs epsilon')
    out = os.path.join(plots_dir, 'distortion_vs_epsilon.png')
    plt.tight_layout()
    plt.savefig(out, dpi=300)
    plt.close()


def plot_cosine_similarity_histogram(plots_dir: str, scores: np.ndarray):
    ensure_dir(plots_dir)
    plt.figure(figsize=(6, 4))
    plt.hist(scores, bins=30, color='steelblue', edgecolor='white')
    plt.xlabel('Top-1 cosine similarity')
    plt.ylabel('Count')
    plt.title('Histogram of cosine similarities (top-1)')
    out = os.path.join(plots_dir, 'cosine_similarity_hist.png')
    plt.tight_layout()
    plt.savefig(out, dpi=300)
    plt.close()


def compute_spearman_from_mats(D_orig_sq: np.ndarray, D_red_sq: np.ndarray) -> float:
    # Convert squared distances to distances
    D1 = np.sqrt(np.maximum(D_orig_sq, 0.0))
    D2 = np.sqrt(np.maximum(D_red_sq, 0.0))
    return rank_corr(D1, D2)


def maybe_load_labels(labels_csv: Optional[str], n_frames: int) -> Optional[Dict]:
    if not labels_csv:
        return None
    df = pd.read_csv(labels_csv)
    required = {"frame_index", "peptide_id", "peptide_rt"}
    if not required.issubset(df.columns):
        raise ValueError(f"labels_csv must contain columns: {required}")
    # Build per-frame lists
    peptide_ids = [None] * n_frames
    for _, row in df.iterrows():
        idx = int(row["frame_index"])
        if 0 <= idx < n_frames:
            peptide_ids[idx] = row["peptide_id"]
    # Build rt_by_peptide
    rt_by_peptide = (
        df.groupby("peptide_id")["peptide_rt"].median().to_dict()
    )
    if any(pid is None for pid in peptide_ids):
        # Fill missing with a dummy label to keep lengths aligned
        for i in range(n_frames):
            if peptide_ids[i] is None:
                peptide_ids[i] = f"UNK_{i}"
    return {
        "peptide_ids": peptide_ids,
        "rt_by_peptide": rt_by_peptide,
    }


def plot_overlay_profiles(plots_dir: str, X: np.ndarray, X_recon: np.ndarray, peptide_ids=None):
    # Plot a few profiles: either per top peptides or random frames
    ensure_dir(plots_dir)
    plt.figure(figsize=(10, 6))
    n = X.shape[0]
    idxs = np.linspace(0, n - 1, num=min(3, n), dtype=int)
    for i in idxs:
        raw = X[i]
        rec = X_recon[i]
        plt.plot(raw, alpha=0.7, label=f"raw[{i}]")
        plt.plot(rec, alpha=0.7, linestyle='--', label=f"recon[{i}]")
    plt.title("Overlay of raw vs reconstructed intensity profiles (sample frames)")
    plt.xlabel("m/z bin index")
    plt.ylabel("normalized intensity")
    plt.legend()
    out = os.path.join(plots_dir, "overlay_profiles.png")
    plt.tight_layout()
    plt.savefig(out, dpi=300)
    plt.close()


def plot_embeddings_scatter(plots_dir: str, Y: np.ndarray, labels=None, rts=None):
    # Reduce to 2D for scatter visualization
    ensure_dir(plots_dir)
    Z = PCA(n_components=2, random_state=0).fit_transform(Y)
    plt.figure(figsize=(8, 6))
    if labels is not None:
        # Map labels to integers
        uniq = {lab: i for i, lab in enumerate(sorted(set(labels)))}
        c = [uniq[l] for l in labels]
        sc = plt.scatter(Z[:, 0], Z[:, 1], c=c, cmap='tab20', s=8)
        plt.colorbar(sc, label='peptide id index')
    elif rts is not None:
        sc = plt.scatter(Z[:, 0], Z[:, 1], c=rts, cmap='viridis', s=8)
        plt.colorbar(sc, label='retention time (min)')
    else:
        plt.scatter(Z[:, 0], Z[:, 1], s=8)
    plt.title("Embeddings scatter (PCA 2D)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    out = os.path.join(plots_dir, "embedding_scatter.png")
    plt.tight_layout()
    plt.savefig(out, dpi=300)
    plt.close()


def write_report_tex(metrics: Dict, plots_dir: str, project_root: str):
    # Build LaTeX report content dynamically based on available figures
    title_status = "N/A"
    acc = metrics.get("accuracy")
    if isinstance(acc, (int, float)):
        title_status = "met" if acc >= 0.7 else "not met"
    # Figures to include
    figures = []
    def add_fig(filename, caption):
        path_rel = os.path.join("results", "plots", filename)
        if os.path.exists(os.path.join(project_root, path_rel)):
            figures.append((path_rel.replace("\\", "/"), caption))
    add_fig("embedding_scatter.png", "Scatter plot of embeddings colored by peptide ID (or RT if labels unavailable).")
    add_fig("overlay_profiles.png", "Overlay of reconstructed vs. raw intensity profiles for sample frames.")
    add_fig("distortion_vs_epsilon.png", "Mean distortion as a function of epsilon.")
    add_fig("cosine_similarity_hist.png", "Histogram of top-1 cosine similarities across frames.")

    tex = []
    tex.append("\\documentclass[11pt,a4paper]{article}")
    tex.append("\\usepackage[margin=1in]{geometry}")
    tex.append("\\usepackage{graphicx}")
    tex.append("\\usepackage{amsmath}")
    tex.append("\\usepackage{booktabs}")
    tex.append("")
    tex.append("\\title{Fast Peptide Identification via JLL + Convex Hull Embeddings}")
    tex.append("\\author{Auto-generated by OrthoReduce}")
    tex.append("\\date{\\today}")
    tex.append("")
    tex.append("\\begin{document}")
    tex.append("\\maketitle")
    tex.append("")
    tex.append("\\begin{abstract}")
    tex.append("This project aims to accelerate peptide identification from mass spectrometry data by projecting high-dimensional spectra into a reduced space via the Johnson--Lindenstrauss lemma and applying convex hull embedding. The first sprint implemented and tested the pipeline on a public mzML dataset. The acceptance criterion of $\\geq 70\\%$ correct assignments within $\\pm 0.5$ minutes RT was \\textbf{" + title_status + "}. Key metrics and visual results are presented.")
    tex.append("\\end{abstract}")
    tex.append("")
    tex.append("\\section{Methods}")
    tex.append("Data were parsed from mzML with TIC normalization and fixed m/z binning. The JL target dimension $k=\\lceil 4\\ln(n)/\\epsilon^2 \\rceil$ was used with GaussianRandomProjection, followed by convex hull projection via constrained optimization. Matching used cosine similarity of per-peptide fingerprints (if labels provided). Evaluation includes distortion, Spearman correlation, and accuracy within \\pm 0.5 minutes RT.")
    tex.append("\\section{Results}")
    tex.append("\\begin{table}[h]")
    tex.append("\\centering")
    tex.append("\\begin{tabular}{lr}")
    tex.append("\\toprule")
    tex.append("Metric & Value \\\\")
    tex.append("\\midrule")
    def fmt(x):
        return ("{:.4f}".format(x) if isinstance(x, (int, float)) else ("N/A" if x is None else str(x)))
    rows = [
        ("Accuracy (\\%)", fmt(metrics.get("accuracy_percent"))),
        ("Time-only Accuracy (\\%)", fmt(metrics.get("time_only_accuracy_percent"))),
        ("Mean Distortion", fmt(metrics.get("mean_distortion"))),
        ("Max Distortion", fmt(metrics.get("max_distortion"))),
        ("Spearman Correlation", fmt(metrics.get("spearman_correlation"))),
        ("Peptides Tested", fmt(metrics.get("n_peptides_tested"))),
        ("Frames (n)", fmt(metrics.get("n_frames"))),
        ("Bins (d)", fmt(metrics.get("n_bins"))),
        ("k", fmt(metrics.get("k"))),
    ]
    for k_, v_ in rows:
        tex.append("{} & {} \\\\".format(k_, v_))
    tex.append("\\bottomrule")
    tex.append("\\end{tabular}")
    tex.append("\\caption{Summary of key metrics.}")
    tex.append("\\end{table}")
    for path_rel, cap in figures:
        tex.append("\\begin{figure}[h]")
        tex.append("\\centering")
        tex.append("\\includegraphics[width=0.8\\textwidth]{%s}" % path_rel)
        tex.append("\\caption{%s}" % cap)
        tex.append("\\end{figure}")
    tex.append("\\end{document}")

    tex_path = os.path.join(project_root, 'report.tex')
    with open(tex_path, 'w') as f:
        f.write("\n".join(tex))


def main():
    parser = argparse.ArgumentParser(description="MS data pipeline (Phases 1–5)")
    parser.add_argument("--mzml", type=str, default=os.path.join("data", "20150731_QEp7_MiBa_SA_SKOV3-1.mzML"))
    parser.add_argument("--epsilon", type=float, default=0.3)
    parser.add_argument("--bin_size", type=float, default=0.1)
    parser.add_argument("--mz_min", type=float, default=None)
    parser.add_argument("--mz_max", type=float, default=None)
    parser.add_argument("--max_spectra", type=int, default=1000)
    parser.add_argument("--results", type=str, default="results")
    parser.add_argument("--labels_csv", type=str, default=None, help="CSV with frame_index,peptide_id,peptide_rt (min)")
    parser.add_argument("--epsilons", type=str, default=None, help="Comma-separated epsilons to plot distortion vs epsilon (e.g., 0.2,0.3,0.4)")

    args = parser.parse_args()
    ensure_dir(args.results)
    plots_dir = os.path.join(args.results, "plots")
    ensure_dir(plots_dir)

    # Phase 1: Parse mzML and build matrix
    X, rts, mz_bins, meta = parse_mzml_build_matrix(
        args.mzml, bin_size=args.bin_size, mz_min=args.mz_min, mz_max=args.mz_max, max_spectra=args.max_spectra
    )
    n, d = X.shape

    # Phase 2: JL dimension and GRP projection
    k = min(jll_dimension(n, args.epsilon), d)
    grp = GaussianRandomProjection(n_components=k, random_state=0)
    Y = grp.fit_transform(X)

    # Compute distortion and Spearman correlation
    mean_dist, max_dist, D_orig_sq, D_red_sq = compute_distortion(X, Y)
    rho = compute_spearman_from_mats(D_orig_sq, D_red_sq)

    # Phase 3: Convex hull projection
    Y_conv, alphas, V = project_onto_convex_hull_qp(Y, tol=1e-6, maxiter=200)
    sums = alphas.sum(axis=1)
    nonneg = (alphas >= -1e-6).all(axis=1)
    close_sums = np.isclose(sums, 1.0, atol=1e-3)
    frac_ok = float((nonneg & close_sums).mean())

    # Approximate reconstruction of X using GRP components for visualization only
    # components_: (k, d); Y = X @ components_.T => X_hat ≈ Y @ components_
    X_recon = (Y @ grp.components_)
    X_recon = np.clip(X_recon, 0.0, None)
    # Row-normalize for comparability
    rs = X_recon.sum(axis=1, keepdims=True) + 1e-12
    X_recon = X_recon / rs

    # Phase 4: Fingerprinting & matching if labels available
    acc = None
    time_acc = None
    labels = None
    labels_data = maybe_load_labels(args.labels_csv, n)
    if labels_data is not None:
        labels = labels_data["peptide_ids"]
        rt_by_peptide = labels_data["rt_by_peptide"]
        match_res = match_frames_to_peptides(Y_conv, labels, rts, rt_by_peptide, rt_tolerance=0.5)
        acc = match_res["accuracy"]
        time_acc = match_res["time_only_accuracy"]

    # Phase 5: Save results
    metrics = {
        "n_frames": n,
        "n_bins": d,
        "k": int(k),
        "epsilon": float(args.epsilon),
        "mean_distortion": float(mean_dist),
        "max_distortion": float(max_dist),
        "spearman_correlation": float(rho),
        "convex_constraint_satisfaction": float(frac_ok),
        "accuracy": (None if acc is None else float(acc)),
        "time_only_accuracy": (None if time_acc is None else float(time_acc)),
    }

    # Save JSON and CSV
    with open(os.path.join(args.results, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    pd.DataFrame([metrics]).to_csv(os.path.join(args.results, "metrics.csv"), index=False)

    # Plots
    plot_overlay_profiles(plots_dir, X, X_recon, peptide_ids=labels)
    plot_embeddings_scatter(plots_dir, Y_conv, labels=labels, rts=rts)
    # Distortion vs epsilon if provided
    if args.epsilons:
        try:
            eps_list = [float(x) for x in args.epsilons.split(',') if x.strip()]
            if eps_list:
                plot_distortion_vs_epsilons(plots_dir, X, eps_list)
        except Exception:
            pass
    # Cosine similarity histogram if labels available
    if labels_data is not None:
        try:
            # Reuse scores from match_res if available; otherwise recompute
            if 'match_res' in locals() and isinstance(match_res, dict) and 'scores' in match_res:
                scores = match_res['scores']
            else:
                # Compute fingerprints to get similarities
                F, _ = compute_fingerprints(Y_conv, labels)
                S = cosine_similarity_matrix(Y_conv, F)
                scores = S[np.arange(S.shape[0]), np.argmax(S, axis=1)]
            plot_cosine_similarity_histogram(plots_dir, np.asarray(scores))
        except Exception:
            pass

    # Write LaTeX report
    # Augment metrics with percentages and peptide count for report
    metrics['accuracy_percent'] = (None if metrics['accuracy'] is None else 100.0 * metrics['accuracy'])
    metrics['time_only_accuracy_percent'] = (None if metrics['time_only_accuracy'] is None else 100.0 * metrics['time_only_accuracy'])
    metrics['n_peptides_tested'] = (None if labels_data is None else int(len(rt_by_peptide)))
    write_report_tex(metrics, plots_dir, project_root=os.path.dirname(os.path.abspath(__file__)))

    print("Pipeline complete. Metrics saved to:", os.path.join(args.results, "metrics.json"))


if __name__ == "__main__":
    main()
