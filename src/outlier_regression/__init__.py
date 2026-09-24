"""Pure-NumPy study of gradient-based optimizers on synthetic linear
regression, and the impact of outliers.

Modules
-------
data            : synthetic data generators (clean & two-population mixture)
optimizers      : hand-coded GD / AdaGrad / RMSProp / Adam update steps
regression      : closed-form normal-equation solver
train           : unified trainer + experiment grid runner
outlier_removal : ``ours`` (the final method), ``ours_v2`` (the general
                  implementation with the stopping rule, per-cycle
                  convergence and the ratio / threshold / readmit pruning
                  rules) and ``ours_v1`` (the first version)
stats           : paired tests (sign, sign-flip, bootstrap CI, Wilcoxon)
plots           : convergence plotting helpers
"""
