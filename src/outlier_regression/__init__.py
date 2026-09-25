"""Pure-NumPy study of gradient-based optimizers on synthetic linear
regression, and the impact of outliers.

Modules
-------
data            : synthetic data generators (clean and two-population mixture)
optimizers      : hand-coded GD / AdaGrad / RMSProp / Adam update steps
regression      : closed-form least squares and the weight error
train           : trainer and optimizer-grid runner
outlier_removal : ``ours_v1`` (starting method), ``ours_v2`` (general
                  implementation used in development) and ``ours`` (final)
stats           : paired sign-flip test and bootstrap CI
diagnostics     : descriptions of a sample selection using the true labels
plots           : learning curves and per-dataset strip plots
"""
