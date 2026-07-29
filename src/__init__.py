"""Pure-NumPy study of gradient-based optimizers on synthetic linear
regression, and the impact of outliers.

Modules
-------
data            : synthetic data generators (clean & two-population mixture)
optimizers      : hand-coded GD / AdaGrad / RMSProp / Adam update steps
regression      : closed-form normal-equation solver
train           : unified trainer + experiment grid runner
outlier_removal : the custom periodic residual-cutoff + re-init method ("ours")
plots           : convergence plotting helpers
"""
