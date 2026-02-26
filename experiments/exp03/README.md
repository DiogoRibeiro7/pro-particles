**Experiment exp03**
Binary classification (D.5, Section 5.1 + Appendix D.5).

**Run**
```bash
python experiments/exp03/run.py
```

**Fast mode**
```bash
python experiments/exp03/run.py experiments/exp03/config_fast.json
```
Fast config reduces `n`, `p`, and `K`, and disables FUSE (uses constant `dt`) for speed.

**Expected output**
- `experiments/exp03/results.json` created.
- `samples_method.npz` saved.
- Metrics (`elpd`) finite.

**Runtime**
- Depends on `K=10000`, `p=64`; expect minutes on CPU.

**Notes**
- FUSE schedule uses `r_eps=0.001` (paper does not specify this parameter).
