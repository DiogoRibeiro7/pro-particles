**Experiment exp01**
Normal location (D.1, Appendix D.1).

**Run**
```bash
python experiments/exp01/run.py
```

**Fast mode**
```bash
python experiments/exp01/run.py experiments/exp01/config_fast.json
```
Fast config reduces `n`, `p`, `K`, and `m_mmd`, and disables FUSE (uses constant `dt`) for speed.

**Expected output**
- `experiments/exp01/results.json` created.
- `samples_method.npz` saved.
- Metrics (`elpd`, posterior mean/std) finite.

**Runtime**
- Depends on `K=4000` and `p=32`; expect minutes on CPU.

**Notes**
- FUSE schedule uses `r_eps=0.001` (paper does not specify this parameter).
