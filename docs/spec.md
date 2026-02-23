# Spec: PrO Posterior Sampling (2510.01915v2)

This file pins down the exact dynamics, estimators, and experimental hyperparameters as written in the paper.

**SDE and Particle Recursion**
- Wasserstein gradient of the objective:

```math
\nabla_W L(Q)[\vartheta] = \lambda_n\, W(Q)[\vartheta] - \nabla_{\vartheta}\log d\Pi(\vartheta) + \nabla_{\vartheta}\log dQ(\vartheta).
```
(Section 6, p. 32)

- Mean-field Langevin SDE for the WGF:

```math
 d\vartheta_t = -\{\lambda_n W(Q_t)[\vartheta_t] - \nabla_{\vartheta}\log d\Pi(\vartheta_t)\}\,dt + \sqrt{2}\, dB_t, \quad \vartheta_t \sim Q_t.
```
(Section 6, p. 32)

- Interacting particle system (with independent Brownian motions):

```math
 d\vartheta_t^{(j)} = -\{\lambda_n W(Q_t)[\vartheta_t^{(j)}] - \nabla_{\vartheta}\log d\Pi(\vartheta_t^{(j)})\}\,dt + \sqrt{2}\, dB_t^{(j)},
 \quad j=1,\dots,p.
```
(Section 6, p. 32)

- Particle approximation used inside W(Q_t):

```math
Q_t(\theta) \approx \frac{1}{p}\sum_{\ell=1,\ \ell\neq j}^p \delta_{\vartheta_t^{(\ell)}}.
```
(Section 6, p. 32)

**W(Q)[theta] by Scoring Rule**
- Squared MMD:

```math
L_{\mathrm{MMD}}(\vartheta,\theta;x) = \langle \mu(P_{\vartheta}) - \mu(\delta_x),\, \mu(P_{\theta}) - \mu(\delta_x) \rangle_{\mathcal{H}},
```

```math
W(Q)[\vartheta] = \frac{1}{n}\sum_{i=1}^n \int \nabla_1 L_{\mathrm{MMD}}(\vartheta,\theta;x_i)\, dQ(\theta).
```
(Appendix E.1, p. 71)

- Logarithmic score:

```math
W(Q)[\vartheta] = \frac{1}{n}\sum_{i=1}^n \frac{\nabla_{\vartheta} dP_{\vartheta}(x_i)}{\int dP_{\theta}(x_i)\, dQ(\theta)}.
```
(Appendix E.1, p. 71)

- Other scores: CRPS is used as an evaluation metric in the linear regression example, but the paper does not derive a corresponding W(Q) for CRPS. (Appendix D.4, pp. 67–68)

**Discrete-Time Estimator of Q_n (Time-Averaged Empirical Measure)**
Given time grid t1, t2, ..., after burn-in \tau, the PrO posterior is approximated as:

```math
Q_n \approx \frac{1}{|\{t_i : t_i > \tau\}|} \sum_{t_i > \tau} \widehat{Q}[t_i],
\quad
\widehat{Q}[t_i] = \frac{1}{p}\sum_{j=1}^p \delta_{\vartheta_{t_i}^{(j)}}.
```
(Section 6, p. 32)

Burn-in is explicitly part of the estimator via \tau, but no numerical \tau is reported in the experimental settings. (Section 6, p. 32; Appendix D.1–D.8, pp. 63–70)

**Experimental Hyperparameters**
The paper reports the following hyperparameters for experiments (PrO/WGF and MALA). When a field is not stated in the paper, it is marked as “not specified.”

| Experiment (Appendix) | p (particles) | dt / step-size schedule | \lambda_n | Burn-in / warmup B | Total steps K | Thinning |
| --- | --- | --- | --- | --- | --- | --- |
| D.1 Normal location illustrations | p = 32 | not specified (WGF) | \lambda_n = n | not specified | 4,000 WGF iterations | not specified |
| D.2 Palmer penguins | p = 32 | not specified (WGF) | \lambda_n = 10^3 | not specified | 10,000 WGF iterations | not specified |
| D.4 Linear regression example | p = 16 (WGF) | MALA: dt = 10^-4 | \lambda_n = n^{1/2} | MALA warmup = 96,000 iterations | WGF: 4,000 iterations; MALA: 4,000 steps | not specified |
| D.4.1 Sensitivity (linear regression) | p = {2^1, 2^2, ..., 2^6} | dt in [0.1, 0.0001]; FUSE adaptive schedule used in all experiments | \lambda_n = n^{1/2} | not specified | 10,000 WGF iterations | not specified |
| D.5 Binary classification | p = 64 (WGF) | MALA: dt = 10^-4 | \lambda_n = n^{1/2} | not specified | WGF: 10,000 iterations; MALA: 4,000 steps | not specified |
| D.6 River water flow | p = 50 (WGF) | MALA: dt = 10^-6 | \lambda_n = n^{1/2} | MALA warmup = 46,000 iterations | WGF: 10,000 iterations; MALA: 4,000 steps | not specified |
| D.8 Conditional auto-regression | p = 16 (WGF, MMD) and p = 128 (WGF, log score) | MALA: dt = 10^-6 | \lambda_n = n^{1/2} | MALA warmup = 96,000 iterations | WGF: 2,000 (MMD) and 4,000 (log score) iterations; MALA: 4,000 steps | not specified |

Sources: Appendix D.1 (p. 63), D.2 (p. 65), D.4–D.4.1 (pp. 67–68), D.5 (p. 69), D.6 (p. 69), D.8 (p. 70).
