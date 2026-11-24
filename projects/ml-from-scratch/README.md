# Machine learning from scratch

**Question:** how do k-nearest neighbors, logistic regression, and a decision tree turn examples into predictions?

Run `python projects/ml-from-scratch/demo.py` with Python 3.10+. It uses the standard library only, a fixed seed, 180 training records, and 60 held-out records. The synthetic boundary is `x1 + 0.4*x2 > 0`. All fitting, including tree threshold selection, uses training data only.

The script implements Euclidean 3-nearest-neighbor classification, logistic regression with averaged gradients and an intercept, and a depth-one decision tree. It reports held-out accuracy for this simple synthetic task, not a business result. The tree is deliberately a stump, not a full CART implementation. There is no model tuning on the test set.

The original notebook is preserved in `legacy/`; it includes additional coursework using datasets and local paths unavailable in this repository. The new demo is an independent, reproducible companion, not a recreation of those reported experiments.

[Website article](https://placenciohid.github.io/2023/07/12/MLScratch.html)
