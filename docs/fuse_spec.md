# FUSE Spec (Exact Transcription from PDF)

**Source PDF**
- Sharrock, L. and C. Nemeth (2025). *Tuning-Free Sampling via Optimization on the Space of Probability Measures*. arXiv:2510.25315. (paper/2510.25315.pdf)

**Equation: ideal (mean-field) Fuse schedule**
- Equation (117): (p. 28)

```math
\eta_t =
\left[
\frac{\max\left\{ r_\varepsilon,\ \max_{1\le s\le t} W_2(\mu^1, \mu^{s-1})^2 \right\}}
{\sum_{s=1}^t \int_{\mathbb{R}^d} \| \zeta_s(x) \|^2\, d\mu_s(x)}
\right]^{1/2}.
```

**Update condition**
- The schedule is defined for \(t \ge 1\); the text states \( (\eta_t)_{t\ge 1} \) and requires a small initialization parameter \(r_\varepsilon\). (p. 28)
- Algorithm 1 updates the step size only if \(t \ge 1\). (Algorithm 1, line 3–4, p. 38)

**Discretisation step where \(\eta_t\) appears (ULA × Fuse)**
- Algorithm 1 (ULA × Fuse), p. 38:
  - Half-step (drift step), Equation (181):

```math
x_i^{t+1/2} = x_i^t - \hat{\eta}_t^n \nabla U(x_i^t).
```

  - Noise step, Equation (182):

```math
x_i^{t+1} = x_i^{t+1/2} + \sqrt{2\hat{\eta}_t^n}\, z_i^t,\quad z_i^t \sim \mathcal{N}(0, I_d).
```

**Practical particle-based schedule (used in Algorithm 1)**
- Algorithm 1, Equation (180): (p. 38)

```math
\hat{\eta}_t^n =
\left[
\frac{\max\left\{ r_\varepsilon,\ \max_{1\le s\le t} \frac{1}{n}\sum_{i=1}^n \|x_i^1 - x_i^{s-1}\|^2 \right\}}
{\sum_{s=1}^t \frac{1}{n}\sum_{i=1}^n \|\nabla U(x_i^s)\|^2}
\right]^{1/2}.
```

**Symbols**
- Numerator: \(\max\{ r_\varepsilon,\ \max_{1\le s\le t} \frac{1}{n}\sum_{i=1}^n \|x_i^1 - x_i^{s-1}\|^2 \}\). (Eq. 180, p. 38)
- Denominator: \(\sum_{s=1}^t \frac{1}{n}\sum_{i=1}^n \|\nabla U(x_i^s)\|^2\). (Eq. 180, p. 38)
- Half-step: \(x_i^{t+1/2}\) is defined by the drift step before noise. (Eq. 181, p. 38)
- Noise step: \(x_i^{t+1}\) adds \(\sqrt{2\hat{\eta}_t^n}\, z_i^t\). (Eq. 182, p. 38)
