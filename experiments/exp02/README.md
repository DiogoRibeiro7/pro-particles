**Experiment exp02**
Linear regression (D.4, Appendix D.4).

**Run**
```bash
python experiments/exp02/run.py
```

**Expected output**
- `experiments/exp02/results.json` created.
- `samples_method.npz` saved.
- Metrics (`elpd`, `mmd2_test_mean`, `crps_mean`) finite.

**Runtime**
- Depends on `K=4000`, `p=16`, and Monte Carlo `m=32`; expect minutes on CPU.

**Notes**
- Covariate distribution `z_i` is not specified in the paper; this script uses standard normal.
- FUSE schedule uses `r_eps=0.001` (paper does not specify this parameter).
