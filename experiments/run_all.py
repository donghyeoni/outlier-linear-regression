"""Regenerate every committed artifact under ``results/`` in one command.

This is the single reproducible entry point for the project: it runs the
baseline optimizer benchmark, the outlier study, the pruning diagnostic of
``ours_v1`` and its MSE-vs-weight-error analysis, the ablation of
the two fixes, the held-out evaluation of the ratio-based method, the
ratio-vs-threshold and re-admission pruning comparisons, the confirmatory
test of the final method and its descriptive values, the final evaluation
on untouched seeds, and the README concept figure. All data is generated
in-code from fixed seeds, so no download or external dataset is required.

Usage
-----
    python experiments/run_all.py
"""

from __future__ import annotations

import analyze_mse_vs_weight
import diagnose_pruning
import plot_concept
import run_ablation
import run_baseline
import run_confirmatory
import run_eval_extra
import run_evaluation
import run_final
import run_outlier
import run_readmit
import run_threshold

STEPS = [
    ("baseline optimizer benchmark", run_baseline),
    ("outlier / mixture study", run_outlier),
    ("pruning diagnostic (ours_v1)", diagnose_pruning),
    ("MSE vs weight error analysis (ours_v1)", analyze_mse_vs_weight),
    ("ablation: stopping rule / per-cycle convergence", run_ablation),
    ("ratio-based method on held-out datasets", run_final),
    ("ratio vs threshold pruning", run_threshold),
    ("re-admission pruning (fresh seeds 21-40)", run_readmit),
    ("pre-registered confirmatory test (seeds 41-140)", run_confirmatory),
    ("descriptive values on the confirmatory seeds", run_eval_extra),
    ("final evaluation on untouched seeds 141-240", run_evaluation),
    ("README concept figure", plot_concept),
]


def main():
    for i, (name, module) in enumerate(STEPS, 1):
        print(f"\n########## {i}/{len(STEPS)}  {name} ##########")
        module.main()
    print("\nAll results regenerated under results/.")


if __name__ == "__main__":
    main()
