# Scoring Rules and W(Q) Drift (2510.01915v2)

Only formulas stated in the PDF are recorded here; each item includes a page reference.

**Logarithmic Scoring Rule (log score)**
- Score (paper’s equivalent to loss): \(S(P, x) = -\log dP(x)\). (p. 5)
- W(Q) expression used in the drift:

```math
W(Q)[\vartheta] = \frac{1}{n}\sum_{i=1}^n \frac{\nabla_{\vartheta} dP_{\vartheta}(x_i)}{\int dP_{\theta}(x_i)\, dQ(\theta)}.
```
(p. 71)

- Particle evolution form (shows leave-one-out averaging and drift use):

```math
d\vartheta_t^{(j)} = -\left\{\lambda_n \frac{1}{n}\sum_{i=1}^n
\frac{\nabla_{\vartheta} dP_{\vartheta_t^{(j)}}(x_i)}
\frac{1}{p-1}\sum_{\ell\neq j} dP_{\vartheta_t^{(\ell)}}(x_i)}
 - \nabla_{\vartheta}\log d\Pi(\vartheta_t^{(j)})\right\} dt + \sqrt{2}\, dB_t^{(j)}.
```
(p. 71)

- Approximation: none specified for the log-score W(Q) formula in the paper. (p. 71)

**Squared MMD**
- Score definition (MMD² with Dirac at x):

```math
\mathrm{MMD}^2(P,\delta_x) :=
\mathbb{E}_{X\sim P,X'\sim P}[k(X,X')]
 + k(x,x) - 2\,\mathbb{E}_{X\sim P}[k(X,x)].
```
(p. 5)

- Loss used for W(Q) in Appendix E.1:

```math
L_{\mathrm{MMD}}(\vartheta,\theta; x)
= \langle \mu(P_{\vartheta}) - \mu(\delta_x),\; \mu(P_{\theta}) - \mu(\delta_x) \rangle_{\mathcal{H}}.
```
(p. 71)

- Gradient w.r.t. the first argument enters W(Q) as:

```math
W(Q)[\vartheta] = \frac{1}{n}\sum_{i=1}^n \int \nabla_1 L_{\mathrm{MMD}}(\vartheta,\theta; x_i)\, dQ(\theta).
```
(p. 71)

- Particle evolution form (shows leave-one-out averaging and drift use):

```math
d\vartheta_t^{(j)} = -\left\{ \frac{\lambda_n}{n}\sum_{i=1}^n
\frac{1}{p-1}\sum_{\ell\neq j}\nabla_1 L_{\mathrm{MMD}}(\vartheta_t^{(j)},\vartheta_t^{(\ell)}; x_i)
 - \nabla_{\vartheta}\log d\Pi(\vartheta_t^{(j)}) \right\} dt + \sqrt{2}\, dB_t^{(j)}.
```
(p. 71)

- Approximation (Monte Carlo) used in regression example for MMD-based losses:

```math
\ell^\dagger(\theta; X,y) = \mathbb{E}_{Y,Y'\sim P_\theta(\cdot|x)}\{\kappa(Y,Y') - 2\kappa(Y,y)\},
```

```math
\ell^\dagger(\theta; X,y) \approx \frac{1}{m(m-1)}\sum_{i\neq j}\kappa(Y_i,Y'_j) - \frac{2}{m}\sum_i \kappa(Y_i,y).
```
(pp. 66–67)

```math
\ell(\vartheta,\theta; X,y) = \mathbb{E}_{Y\sim P_\vartheta(\cdot|x),\, Y'\sim P_\theta(\cdot|x)}
\{\kappa(Y,Y') - \kappa(Y,y) - \kappa(Y',y)\},
```

```math
\ell(\vartheta,\theta; X,y) \approx \frac{1}{m^2}\sum_{i=1}^m\sum_{j=1}^m\kappa(Y_i,Y'_j)
 - \frac{1}{m}\sum_i\kappa(Y_i,y) - \frac{1}{m}\sum_i\kappa(Y'_i,y).
```
(pp. 66–67)

**Other Scoring Rules Mentioned**
- Spherical score, CRPS, energy score are listed as possible scoring rules; no W(Q) formula is given for these. (p. 5)
- CRPS is used as an evaluation metric in experiments; no W(Q) or L(θ,θ';x) drift formula is provided. (p. 68)
