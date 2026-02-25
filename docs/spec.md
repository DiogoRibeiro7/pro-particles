# Spec: PrO Particle System (2510.01915v2)

This spec is a contract for an unambiguous implementation. Every item includes a page reference to the paper.

**1) Notation and Shapes**
- Parameters: \(\vartheta \in \mathbb{R}^d\). (p. 32)
- Particles: \(\vartheta_t^{(j)}\) for \(j = 1,\dots,p\). (p. 32)
- Time: continuous \(t\) for the SDE; discrete times \(t_1,t_2,\dots\) for simulation. (p. 32)
- Empirical particle measure (leave-one-out inside W): \(Q_t(\theta) \approx \frac{1}{p}\sum_{\ell=1,\ \ell\neq j}^p \delta_{\vartheta_t^{(\ell)}}\). (p. 32)
- Quote (leave-one-out scaling): “\(Q_t(\theta) \approx p^{-1}\sum_{\ell=1,\ell\neq j}^p \delta_{\vartheta_t^{(\ell)}}\)”. (p. 32)
- In Appendix E.1, the evolution equations use \((p-1)^{-1}\sum_{\ell\neq j}\) (see drift formulas there). (p. 71)

**2) Discrete Update Step (Euler–Maruyama from the stated SDE)**
- Wasserstein gradient of the objective:

```math
\nabla_W L(Q)[\vartheta] = \lambda_n\, W(Q)[\vartheta] - \nabla_{\vartheta}\log d\Pi(\vartheta) + \nabla_{\vartheta}\log dQ(\vartheta).
```
(p. 32)

- Mean-field Langevin SDE:

```math
d\vartheta_t = -\{\lambda_n W(Q_t)[\vartheta_t] - \nabla_{\vartheta}\log d\Pi(\vartheta_t)\}\,dt + \sqrt{2}\, dB_t.
```
(p. 32)

- Interacting particle system:

```math
d\vartheta_t^{(j)} = -\{\lambda_n W(Q_t)[\vartheta_t^{(j)}] - \nabla_{\vartheta}\log d\Pi(\vartheta_t^{(j)})\}\,dt + \sqrt{2}\, dB_t^{(j)},
\quad j=1,\dots,p.
```
(p. 32)

- Discretisation statement (paper wording, short quote): “one discretises it into time steps \(t_1, t_2, \dots\)”. (p. 32)

- **Implementation contract (Euler–Maruyama)** derived directly from the SDE above, fixing signs and constants:

```math
\vartheta_{k+1}^{(j)} =
\vartheta_k^{(j)}
 - \{\lambda_n W(Q_{t_k})[\vartheta_k^{(j)}] - \nabla_{\vartheta}\log d\Pi(\vartheta_k^{(j)})\}\, \Delta t_k
 + \sqrt{2\,\Delta t_k}\, \xi_k^{(j)},
```

with \(\xi_k^{(j)} \sim \mathcal{N}(0, I_d)\) i.i.d. across \(k,j\). (SDE on p. 32; discretisation sentence on p. 32)
- The paper does **not** name “Euler–Maruyama” explicitly; the discrete update above is the direct EM interpretation of the stated SDE and time-step sentence. (p. 32)

**3) Construction of \(Q_n\) from the Trajectory**
- Time-averaged empirical measure after burn-in:

```math
Q_n \approx \frac{1}{|\{t_i : t_i > \tau\}|} \sum_{t_i>\tau} \widehat{Q}[t_i],
\quad
\widehat{Q}[t_i] = \frac{1}{p}\sum_{j=1}^p \delta_{\vartheta_{t_i}^{(j)}}.
```
(p. 32)

- Quote (burn-in): “past an initial burn-in \(\tau\)”. (p. 32)
- Burn-in is explicit in the construction via \(\tau\); no numeric \(\tau\) is specified in experiments. (p. 32; pp. 63–70)

- Quote (time averaging): “one averages over all time steps past an initial burn-in \(\tau\)”. (p. 32)
- The approximation uses time-averaging of the particle measures \(\widehat{Q}[t_i]\). (p. 32)

**4) Hyperparameters Used in Experiments**
The paper reports the following values; fields not stated are “not specified.” (pp. 63–70)

| Experiment (Appendix) | p (particles) | dt / step-size schedule | \(\lambda_n\) | Burn-in / warmup B | Total steps K | Thinning |
| --- | --- | --- | --- | --- | --- | --- |
| D.1 Normal location | \(p=32\) | not specified (WGF) | \(\lambda_n=n\) | not specified | 4,000 WGF iterations | not specified |
| D.2 Palmer penguins | \(p=32\) | not specified (WGF) | \(\lambda_n=10^3\) | not specified | 10,000 WGF iterations | not specified |
| D.4 Linear regression | \(p=16\) (WGF) | MALA: \(dt=10^{-4}\) | \(\lambda_n=n^{1/2}\) | MALA warmup = 96,000 | WGF: 4,000; MALA: 4,000 | not specified |
| D.4.1 Sensitivity | \(p=\{2^1,2^2,\dots,2^6\}\) | \(dt \in [0.1, 0.0001]\); FUSE adaptive schedule used in all experiments | \(\lambda_n=n^{1/2}\) | not specified | 10,000 WGF iterations | not specified |
| D.5 Binary classification | \(p=64\) (WGF) | MALA: \(dt=10^{-4}\) | \(\lambda_n=n^{1/2}\) | not specified | WGF: 10,000; MALA: 4,000 | not specified |
| D.6 River water flow | \(p=50\) (WGF) | MALA: \(dt=10^{-6}\) | \(\lambda_n=n^{1/2}\) | MALA warmup = 46,000 | WGF: 10,000; MALA: 4,000 | not specified |
| D.8 Conditional auto-regression | \(p=16\) (MMD), \(p=128\) (log score) | MALA: \(dt=10^{-6}\) | \(\lambda_n=n^{1/2}\) | MALA warmup = 96,000 | WGF: 2,000 (MMD), 4,000 (log) | not specified |

Sources: Appendix D.1 (p. 63), D.2 (p. 65), D.4–D.4.1 (pp. 67–68), D.5 (p. 69), D.6 (p. 69), D.8 (p. 70).

FUSE schedule definition source: see `docs/fuse_sources.md` (Sharrock & Nemeth, 2025). (p. 72)
