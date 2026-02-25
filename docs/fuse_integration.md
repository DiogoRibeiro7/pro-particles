# FUSE Integration Mapping

**Function signature**
- `update_eta(state, *, t, particles_t, particles_prev, grad_t) -> float`

**State carried across iterations**
- `r_eps`: initial movement parameter \(r_\varepsilon\). (docs/fuse_spec.md, Eq. 180)
- `x1`: particles at time \(t=1\) (used in \(\|x_i^1 - x_i^{s-1}\|^2\)). (Eq. 180)
- `max_movement`: running max of \(\frac{1}{n}\sum_i \|x_i^1 - x_i^{s-1}\|^2\). (Eq. 180)
- `sum_grad_norm_sq`: running sum of \(\frac{1}{n}\sum_i \|\nabla U(x_i^s)\|^2\). (Eq. 180)
- `eta`: latest \(\hat{\eta}_t^n\). (Eq. 180)

**Where in the loop**
- Initialize with \(\eta_0 = r_\varepsilon\) and particles \(x^0\). (Algorithm 1, line 1, p. 38)
- Iterate `t = 0, 1, ..., T-1`. (Algorithm 1, line 2, p. 38)
- If `t >= 1`, update step size using Eq. 180 *before* the drift and noise steps. (Algorithm 1, lines 3–4, p. 38)
- Apply half-step using \(\eta_t\): \(x^{t+1/2} = x^t - \eta_t \nabla U(x^t)\). (Eq. 181, p. 38)
- Apply noise step using the same \(\eta_t\): \(x^{t+1} = x^{t+1/2} + \sqrt{2\eta_t}\, z^t\). (Eq. 182, p. 38)

**Mapping to arrays**
- `particles_t`: \(x^t\).
- `particles_prev`: \(x^{t-1}\) (used to update the numerator max).
- `x1`: \(x^1\) stored after the first full update (t = 0).
- `grad_t`: \(\nabla U(x_i^t)\) per particle, used for denominator update.
