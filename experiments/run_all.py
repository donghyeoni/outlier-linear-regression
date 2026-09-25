"""Run every experiment in order and regenerate ``results/`` and the
concept figure in ``docs/images/``.

    python experiments/run_all.py
"""

from __future__ import annotations

import dev0_baselines
import dev1_v1
import dev2_converge
import dev3_stop
import dev4_rules
import dev5_final
import dev_summary
import exp1_clean
import plot_concept
import t1_final

STEPS = [
    ("E1  optimizer grid on clean data", exp1_clean),
    ("D0  Oracle, Naive and the grid on mixture data", dev0_baselines),
    ("D1  ours_v1", dev1_v1),
    ("D2  per-cycle convergence", dev2_converge),
    ("D3  stopping rule", dev3_stop),
    ("D4  pruning rule", dev4_rules),
    ("D5  final settings", dev5_final),
    ("    development summary", dev_summary),
    ("T1  final evaluation", t1_final),
    ("    concept figure", plot_concept),
]


def main():
    for i, (name, module) in enumerate(STEPS, 1):
        print(f"\n##### {i}/{len(STEPS)} {name}")
        module.main()


if __name__ == "__main__":
    main()
