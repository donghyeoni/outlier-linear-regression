"""Regenerate every committed artifact under ``results/`` in one command.

This is the single reproducible entry point for the project: it runs the
baseline optimizer benchmark (notebook 1) followed by the outlier study
(notebook 2). All data is generated in-code from fixed seeds, so no download
or external dataset is required.

Usage
-----
    python experiments/run_all.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import experiments.run_baseline as run_baseline
import experiments.run_outlier as run_outlier


def main():
    print("########## 1/2  baseline optimizer benchmark ##########")
    run_baseline.main()
    print("\n########## 2/2  outlier / mixture study ##########")
    run_outlier.main()
    print("\nAll results regenerated under results/.")


if __name__ == "__main__":
    main()
