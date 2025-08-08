Title: OrthoReduce Sprint 1 – MS Pipeline with JL and Convex Hull Projection

1) Introduction (0:00–0:40)
Hi, I’m walking you through Sprint 1 of OrthoReduce, an open-source toolkit for dimensionality reduction with a focus on stability and geometric structure. Our first sprint targets a mass spectrometry (MS) use case. We’ll parse mzML spectral data, convert it into normalized intensity frames, reduce dimensionality using the Johnson–Lindenstrauss (JL) lemma, project onto a convex hull for geometric consistency, and finally compute peptide fingerprints for matching. We’ll conclude with metrics and visuals, and reflect on runtime and quality.

2) Sprint Goal and Test Case (0:40–1:10)
Goal: build an end‑to‑end pipeline that accepts an mzML file, produces a well-formed (frames × bins) intensity matrix, applies JL projection with a theoretically derived target k, refines with convex hull projection, and evaluates peptide assignment accuracy using a ±0.5 minute retention-time window. As an example dataset, place an mzML file under data/. Optionally, provide a labels CSV mapping frame indices to peptide IDs and canonical retention times.

3) Execution Walkthrough (1:10–3:30)
Phase 0 – Setup: Create a fresh Python environment and install requirements: numpy, scipy, scikit-learn, matplotlib, umap-learn (optional), pyteomics, pandas, and jupyter. Put your mzML file into the data/ directory.

Phase 1 – Data Preparation: We parse the mzML with pyteomics and perform fixed m/z binning (default 0.1 Th). For each spectrum, we build a histogram over these bins, then normalize each frame by total ion current (TIC) so every row sums to ~1 where signal exists. We also extract retention times in minutes and ensure there are no NaNs. Unit tests check shape, normalization, and finiteness.

Phase 2 – Dimensionality Reduction (JL): We compute k using ceil(4 * ln(n) / epsilon^2) and cap it by the original dimension. We project frames using GaussianRandomProjection from scikit-learn. We measure mean and max relative squared-distance distortion and compute the Spearman rank correlation of pairwise distances before and after projection. In practice, lower epsilon (e.g., 0.2–0.3) increases k and typically yields better distance preservation.

Phase 3 – Convex Hull Projection: To impose geometric consistency, we obtain hull vertices (scipy.spatial.ConvexHull when feasible) and, for each projected point y, solve min ||Vα − y||^2 subject to sum(α)=1 and α≥0 using SLSQP (scipy.optimize.minimize). We validate constraints and target ≥99% satisfaction. If the hull is large or ill-conditioned, we fall back to extreme points per dimension. Unit tests verify both accuracy and constraint adherence on synthetic polygons.

Phase 4 – Peptide Fingerprinting & Matching: We form a fingerprint for each peptide by averaging its convex embeddings. For each frame, we compute cosine similarities to all fingerprints and take the top match. A frame is correctly assigned if the predicted peptide’s canonical RT is within ±0.5 minutes of the frame’s ground-truth peptide RT and the label matches. On synthetic, well-separated data we achieve ≥70% strict accuracy. On real data, accuracy depends on noise, chromatography, and label quality.

Phase 5 – Results & Visualization: The CLI ms_pipeline.py saves a metrics table (CSV/JSON) with n, d, k, mean/max distortion, Spearman correlation, and the convex constraint satisfaction rate. If labels are provided, it also reports strict and time-only accuracies. We include two plots: an overlay of reconstructed versus raw intensity profiles for sample frames, and a 2D scatter of embeddings (PCA) colored by peptide ID or RT.

4) Results and Observations (3:30–4:30)
With moderate epsilon (e.g., 0.3), JL typically preserves distances well—mean distortion can be kept low and rank correlation high. The convex hull step encourages embeddings to lie within the feasible polytope spanned by extreme patterns, improving interpretability for mixture-like signals. Accuracy hinges on peptide separability and the quality of labels/RT priors; results are reported per run for transparency.

5) Reflections and Next Steps (4:30–5:00)
We completed an end‑to‑end path with tests for stability. Next steps include: refining binning adaptively, incorporating MS/MS identification evidence, calibrating RT alignment models, trying alternative convex/regularization schemes, and benchmarking on curated BSA tryptic digest datasets with known IDs. We’ll also explore faster solvers and distributed computation to scale beyond thousands of frames.

Thanks for watching!
