"""Evaluate re-admission pruning against ratio pruning (log E5).

``readmit`` recomputes the inlier set from all samples after every converged
cycle (``|r| <= 3 * sigma_MAD``), so samples wrongly removed by the
contaminated first fit can return. Settings (``k = 3``, ``tol = 1e-5``, at
most 5 cycles) were fixed before this evaluation.

Seeds 1-20 were already used to compare ratio vs threshold pruning, so the
primary evaluation uses fresh seeds 21-40. Seeds 1-20 are reported as a
secondary check.

Also records paired tests (readmit - ratio) per seed set, the inlier-set
trace on seed 0, and a 20-cycle diagnostic for runs that hit the 5-cycle cap.

Writes ``results/ablation/step4_readmit.csv`` and
``results/ablation/step4_readmit_summary.json``.

Usage
-----
    python experiments/run_readmit.py
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from outlier_regression.data import generate_mixture_data
from outlier_regression.outlier_removal import ours_v2
from outlier_regression.regression import closed_form_solution, weight_error
from outlier_regression.stats import sign_flip_p, sign_test_p

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(REPO_ROOT, "results", "ablation")

PRIMARY_SEEDS = range(21, 41)
SECONDARY_SEEDS = range(1, 21)
RULES = ("ratio", "readmit")
LEARNING_RATES = (0.01, 0.1, 0.5)
CONFIG = {"converge_tol": 1e-5, "stop_k": 3.0}


def evaluate(seed):
    X, y, z, w1, _ = generate_mixture_data(N=1000, D=4, p=0.9, seed=seed)
    clean = z == 1
    noise = y - np.where(clean, 1.0, -1.0) * (X @ w1)
    row = {
        "seed": seed,
        "n_outliers": int(np.sum(z == 2)),
        "Oracle": weight_error(closed_form_solution(X[clean], y[clean]), w1),
        "Naive": weight_error(closed_form_solution(X, y), w1),
    }
    for rule in RULES:
        for lr in LEARNING_RATES:
            _, _, w_hist, info = ours_v2(
                X, y, w1, learning_rate=lr, prune_rule=rule, eval_X=X,
                eval_y=y, eval_mask=clean, seed=0, **CONFIG)
            kept = info["kept_idx"]
            removed_clean = np.setdiff1d(np.flatnonzero(clean), kept)
            key = f"{rule}_lr{lr}"
            row[key] = w_hist[-1]
            row[f"{key}_clean_kept"] = int(np.sum(z[kept] == 1))
            row[f"{key}_outliers_left"] = int(np.sum(z[kept] == 2))
            row[f"{key}_cycles"] = len(info["cycle_epochs"])
            row[f"{key}_stopped"] = info["stopped_at_cycle"] is not None
            row[f"{key}_removed_clean_pos"] = int(np.sum(noise[removed_clean] > 0))
            row[f"{key}_removed_clean_neg"] = int(np.sum(noise[removed_clean] < 0))
    return row


def paired_tests(diff, n_flips=100_000):
    """Two-sided sign test and sign-flip permutation test for mean(diff)."""
    return {"n": len(diff), "mean_diff": float(diff.mean()),
            "readmit_better": int(np.sum(diff < 0)),
            "p_sign_test": sign_test_p(diff),
            "p_sign_flip": sign_flip_p(diff, n_flips,
                                       np.random.default_rng(0))}


def readmit_trace(seed, num_cycles):
    """Inlier-set sizes (and outliers in it) after each readmit cycle."""
    X, y, z, w1, _ = generate_mixture_data(N=1000, D=4, p=0.9, seed=seed)
    clean = z == 1
    trace = []
    _, _, w_hist, info = ours_v2(
        X, y, w1, learning_rate=0.1, prune_rule="readmit",
        eval_X=X, eval_y=y, eval_mask=clean, seed=0,
        **{**CONFIG, "num_cycles": num_cycles},
        on_prune=lambda ep, rem, kept: trace.append(
            {"kept": int(len(kept)), "outliers": int(np.sum(z[kept] == 2))}))
    return {"seed": seed, "num_cycles_cap": num_cycles,
            "stopped_at_cycle": info["stopped_at_cycle"],
            "cycles_run": len(info["cycle_epochs"]),
            "inlier_sets": trace, "weight_error": w_hist[-1]}


def summarize(df, label):
    out = {"seeds": [int(df.seed.min()), int(df.seed.max())], "methods": {}}
    print(f"\n{label} (seeds {df.seed.min()}-{df.seed.max()}):")
    for c in ["Oracle", "Naive"] + [f"{r}_lr{lr}" for r in RULES
                                    for lr in LEARNING_RATES]:
        v = df[c].to_numpy()
        m = {"weight_error_mean": float(v.mean()),
             "weight_error_std": float(v.std())}
        if c not in ("Oracle", "Naive"):
            m.update({
                "clean_kept_mean": float(df[f"{c}_clean_kept"].mean()),
                "outliers_left_total": int(df[f"{c}_outliers_left"].sum()),
                "max_cycles": int(df[f"{c}_cycles"].max()),
                "not_stopped": int((~df[f"{c}_stopped"]).sum()),
                "removed_clean_pos_total": int(df[f"{c}_removed_clean_pos"].sum()),
                "removed_clean_neg_total": int(df[f"{c}_removed_clean_neg"].sum()),
            })
        out["methods"][c] = m
        extra = ""
        if "clean_kept_mean" in m:
            extra = (f"  clean kept {m['clean_kept_mean']:.1f}  outliers left "
                     f"{m['outliers_left_total']}  removed clean +/- "
                     f"{m['removed_clean_pos_total']}/"
                     f"{m['removed_clean_neg_total']}  not stopped "
                     f"{m['not_stopped']}")
        print(f"  {c:14s} {v.mean():.4f} ± {v.std():.4f}{extra}")
    a, b = df["readmit_lr0.1"], df["ratio_lr0.1"]
    out["readmit_beats_ratio"] = int(np.sum(a < b))
    out["readmit_minus_ratio_mean"] = float((a - b).mean())
    out["readmit_minus_oracle_mean"] = float((a - df["Oracle"]).mean())
    print(f"  readmit < ratio on {out['readmit_beats_ratio']}/{len(df)}; "
          f"mean diff {out['readmit_minus_ratio_mean']:+.4f}; "
          f"readmit - Oracle {out['readmit_minus_oracle_mean']:+.4f}")
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.DataFrame([evaluate(s) for s in [0, *PRIMARY_SEEDS,
                                             *SECONDARY_SEEDS]])
    df.to_csv(os.path.join(OUT_DIR, "step4_readmit.csv"), index=False)
    primary = df[df.seed.isin(PRIMARY_SEEDS)]
    secondary = df[df.seed.isin(SECONDARY_SEEDS)]
    diff = lambda d: (d["readmit_lr0.1"] - d["ratio_lr0.1"]).to_numpy()
    tests = {"seeds_21_40": paired_tests(diff(primary)),
             "seeds_1_20": paired_tests(diff(secondary)),
             "seeds_1_40": paired_tests(np.concatenate([diff(secondary),
                                                        diff(primary)]))}
    for name, t in tests.items():
        print(f"{name}: mean {t['mean_diff']:+.4f}, better "
              f"{t['readmit_better']}/{t['n']}, sign p {t['p_sign_test']:.3f}, "
              f"sign-flip p {t['p_sign_flip']:.4f}")
    capped = df[~df["readmit_lr0.1_stopped"] & (df.seed > 0)].seed.tolist()
    summary = {
        "config": CONFIG,
        "paired_tests_lr0.1": tests,
        "seed0_trace": readmit_trace(0, 5),
        "capped_runs_with_20_cycles": [readmit_trace(int(s), 20)
                                       for s in capped],
        "development_seed0": {k: (v.item() if hasattr(v, "item") else v)
                              for k, v in df[df.seed == 0].iloc[0].items()},
        "primary": summarize(primary, "primary"),
        "secondary": summarize(secondary, "secondary"),
    }
    with open(os.path.join(OUT_DIR, "step4_readmit_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nArtifacts saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
