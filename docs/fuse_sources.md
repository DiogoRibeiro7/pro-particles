# FUSE Sources (PDF-only)

**FUSE reference paper (bibliography entry in 2510.01915v2.pdf)**
- Sharrock, L. and C. Nemeth (2025, October). *Tuning-Free Sampling via Optimization on the Space of Probability Measures*. arXiv:2510.25315 [stat]. (p. 69, p. 72 in 2510.01915v2.pdf)

**Where FUSE is mentioned in 2510.01915v2.pdf**
- Appendix E.2 “Practicalities and Implementation”: “Sharrock and Nemeth (2025) developed an adaptive step size schedule … which they call FUSE.” (p. 72)
- No FUSE equations/algorithms are given in 2510.01915v2.pdf. (p. 72)

**FUSE definition in Sharrock & Nemeth (2025) PDF (paper/2510.25315.pdf)**

**Mean-field (ideal) Fuse step size schedule**
- Section 4.2.2 “Deterministic Case: Adaptive Step Size”. (p. 28)
- Equation (117):
  - \( \eta_t = \left[ \max\{ r_\varepsilon, \max_{1\le s\le t} W_2(\mu^1, \mu^{s-1})^2 \} \Big/ \sum_{s=1}^t \int_{\mathbb{R}^d} \|\zeta_s(x)\|^2 d\mu_s(x) \right]^{1/2} \). (p. 28)
  - Variable names: \( \eta_t, r_\varepsilon, \mu^1, \mu^{s-1}, W_2(\cdot,\cdot), \zeta_s, \mu_s \). (p. 28)
- “Fuse: Functional Upper-Bound Step-Size Estimator.” (Equation (118), p. 28)

**Practical particle-based schedule and update steps (ULA × Fuse)**
- Algorithm 1 “ULA x Fuse”. (p. 38)
- Equation (180) (step size update):
  - \( \hat{\eta}_t^n = \left[ \max\{ r_\varepsilon, \max_{1\le s\le t} \tfrac{1}{n}\sum_{i=1}^n \|x_i^1 - x_i^{s-1}\|^2 \} \Big/ \sum_{s=1}^t \tfrac{1}{n}\sum_{i=1}^n \|\nabla U(x_i^s)\|^2 \right]^{1/2} \). (p. 38)
  - Variable names: \( \hat{\eta}_t^n, r_\varepsilon, x_i^1, x_i^{s-1}, n, \nabla U(x_i^s) \). (p. 38)
- Equation (181) (half-step / drift step):
  - \( x_i^{t+1/2} = x_i^t - \hat{\eta}_t^n \nabla U(x_i^t) \). (p. 38)
- Equation (182) (noise step):
  - \( x_i^{t+1} = x_i^{t+1/2} + \sqrt{2\hat{\eta}_t^n}\, z_i^t \), \( z_i^t \sim \mathcal{N}(0, I_d) \). (p. 38)

