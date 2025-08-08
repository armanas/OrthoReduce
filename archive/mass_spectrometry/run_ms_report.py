#!/usr/bin/env python3
"""
run_ms_report.py - One-shot script to run the MS pipeline and build the PDF report.

This script will:
  1) Run ms_pipeline.py on the given mzML file to generate results/metrics and plots
  2) Optionally compile report.tex into report.pdf using pdflatex (if available)

Example:
  python run_ms_report.py \
    --mzml data/20150731_QEp7_MiBa_SA_SKOV3-1.mzML \
    --epsilon 0.3 \
    --max_spectra 1000 \
    --epsilons 0.2,0.3,0.4

Notes:
- Ensure dependencies are installed: pip install -r requirements.txt
- For PDF compilation, a LaTeX distribution providing `pdflatex` must be installed.
"""
import os
import sys
import argparse
import subprocess
import shutil


def main():
    parser = argparse.ArgumentParser(description="Run MS pipeline and build PDF report")
    parser.add_argument("--mzml", type=str, default=os.path.join("data", "20150731_QEp7_MiBa_SA_SKOV3-1.mzML"), help="Path to mzML file")
    parser.add_argument("--epsilon", type=float, default=0.3, help="JL epsilon")
    parser.add_argument("--bin_size", type=float, default=0.1, help="m/z bin size")
    parser.add_argument("--mz_min", type=float, default=None, help="Min m/z")
    parser.add_argument("--mz_max", type=float, default=None, help="Max m/z")
    parser.add_argument("--max_spectra", type=int, default=1000, help="Max spectra to parse")
    parser.add_argument("--results", type=str, default="results", help="Results directory")
    parser.add_argument("--labels_csv", type=str, default=None, help="CSV with frame_index,peptide_id,peptide_rt (min)")
    parser.add_argument("--epsilons", type=str, default=None, help="Comma-separated epsilons for distortion vs epsilon plot")
    parser.add_argument("--no-pdf", action="store_true", help="Skip PDF compilation of report.tex")
    parser.add_argument("--python", type=str, default=sys.executable, help="Python interpreter to use")
    args = parser.parse_args()

    # Ensure results directory exists
    os.makedirs(args.results, exist_ok=True)

    # Build ms_pipeline command
    cmd = [
        args.python,
        os.path.join(os.path.dirname(__file__), "ms_pipeline.py"),
        "--mzml", args.mzml,
        "--epsilon", str(args.epsilon),
        "--bin_size", str(args.bin_size),
        "--max_spectra", str(args.max_spectra),
        "--results", args.results,
    ]
    if args.mz_min is not None:
        cmd.extend(["--mz_min", str(args.mz_min)])
    if args.mz_max is not None:
        cmd.extend(["--mz_max", str(args.mz_max)])
    if args.labels_csv:
        cmd.extend(["--labels_csv", args.labels_csv])
    if args.epsilons:
        cmd.extend(["--epsilons", args.epsilons])

    print("[run_ms_report] Running pipeline:")
    print(" ", " ".join(cmd))
    subprocess.run(cmd, check=True)

    metrics_json = os.path.join(args.results, "metrics.json")
    if os.path.exists(metrics_json):
        print(f"[run_ms_report] Metrics written to: {metrics_json}")
    else:
        print(f"[run_ms_report] Warning: metrics.json not found at {metrics_json}")

    # Compile report.tex to PDF if requested and available
    if not args.__dict__.get("no_pdf"):
        pdflatex = shutil.which("pdflatex")
        tex_path = os.path.join(os.path.dirname(__file__), "report.tex")
        if pdflatex and os.path.exists(tex_path):
            print("[run_ms_report] Compiling report.tex with pdflatex...")
            try:
                # Run twice for stable references
                subprocess.run([pdflatex, "-interaction=nonstopmode", "-halt-on-error", "report.tex"], check=True, cwd=os.path.dirname(__file__))
                subprocess.run([pdflatex, "-interaction=nonstopmode", "-halt-on-error", "report.tex"], check=True, cwd=os.path.dirname(__file__))
                pdf_path = os.path.join(os.path.dirname(__file__), "report.pdf")
                if os.path.exists(pdf_path):
                    print(f"[run_ms_report] PDF built: {pdf_path}")
                else:
                    print("[run_ms_report] pdflatex ran but report.pdf not found")
            except subprocess.CalledProcessError as e:
                print("[run_ms_report] pdflatex failed:", e)
        else:
            if not pdflatex:
                print("[run_ms_report] Skipping PDF build: 'pdflatex' not found on PATH")
            if not os.path.exists(tex_path):
                print("[run_ms_report] Skipping PDF build: report.tex not found (pipeline may have failed)")
    else:
        print("[run_ms_report] --no-pdf set; skipping PDF compilation")

    print("[run_ms_report] Done.")


if __name__ == "__main__":
    main()
