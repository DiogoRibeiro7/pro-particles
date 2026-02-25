# PDF Index: 2510.01915v2

**Sections (titles as written) + page numbers**
- 1     Background — p. 4
- 1.1    Key Ideas — p. 5
- 1.2    Related Work — p. 8
- 2     A Motivating Example — p. 9
- 2.1    Numerical Results — p. 11
- 2.2    A Taxonomy of Model Misspecification — p. 12
- 3        Formal Guarantees — p. 13
- 3.1    Regularity conditions — p. 14
- 3.2    Key Results — p. 15
- 4     Parameter Uncertainty of the PrO Posterior — p. 17
- 4.1    The Origin of PrO Posterior Uncertainty — p. 18
- 4.2    Key Comparisons — p. 21
- 5      Illustrations — p. 26
- 5.1      Binary classification — p. 26
- 5.2    River flow data — p. 27
- 5.3    Boston housing data — p. 29
- 6     Computation via Wasserstein Gradient Flows — p. 31
- 7     Discussion — p. 34
- 8     Detailed Theoretical Developments — p. 42
- 8.1     A Preliminary Result — p. 42
- 8.2    Generic Misspecification — p. 43
- 8.3    Non-Trivial Misspecification — p. 44
- 8.4     Misspecification with Convex Recovery — p. 45
- Appendix — p. 49
- A Notations                                                                             50 — p. 49
- B Lemmas                                                                                51 — p. 49
- B.1 Existence and uniqueness of the minimiser . . . . . . . . . . . . . . . . .     54 — p. 49
- C Proofs of Main results                                                                56 — p. 49
- D Further experimental details                                                          64 — p. 49
- D.1 Normal location illustrations . . . . . . . . . . . . . . . . . . . . . . . .   64 — p. 49
- D.2 Palmer penguins example . . . . . . . . . . . . . . . . . . . . . . . . . .     64 — p. 49
- D.3 Regression with the MMD . . . . . . . . . . . . . . . . . . . . . . . . . .          66 — p. 50
- D.4 Linear regression example . . . . . . . . . . . . . . . . . . . . . . . . . .        67 — p. 50
- D.5 Binary classification example . . . . . . . . . . . . . . . . . . . . . . . .        69 — p. 50
- D.6 River water flow example . . . . . . . . . . . . . . . . . . . . . . . . . .         69 — p. 50
- D.7 Time series cross-validation . . . . . . . . . . . . . . . . . . . . . . . . .       70 — p. 50
- D.8 Conditional auto-regression example . . . . . . . . . . . . . . . . . . . .          70 — p. 50
- E Computation                                                                               71 — p. 50
- E.1 Examples: MMD and logarithmic score . . . . . . . . . . . . . . . . . .              71 — p. 50
- E.2 Practicalities and Implementation . . . . . . . . . . . . . . . . . . . . . .        72 — p. 50
- A      Notations — p. 50

**Algorithms / Theorems / Propositions / Definitions (labels as written) + page numbers**
- Definition 1 (Scoring Rule). Let S : P(X ) × X → R ∪ {∞} be an extended real-valued — p. 5
- Definition 2. We say that MΘ is well-specified if P0 ∈ MΘ , and misspecified if P0 ∈ — p. 12
- Theorem 1. Assumption 1 is satisfied, and Assumption 2 or 2′ holds. Under prior regu- — p. 16
- Corollary 1. Assumption 1 and Assumption 2 or 2′ holds. Under convex recovery and — p. 17
- Lemma 1. Assumption 1 is satisfied. If Assumption 2 or Assumption 2′ holds, then — p. 43
- Theorem 2. Under Assumptions 1-3, E[DS (PQn , P0 ) − DS (Pθ⋆ , P0 )] ≲ νn . Furthermore, — p. 44
- Theorem 3. Under Assumptions 1-4 and for n large enough but finite, it holds that — p. 45
- Theorem 4. Under Assumptions 1-5, and for n sufficiently large but finite, E [DS (PQn , P0 )] < — p. 46
- Corollary 2. If Assumptions 1-2 and 5(ii) hold for J = 1 and θ1⋆ = θ⋆ , then E [DS (PQn , P0 )]− — p. 47
- Corollary 3. If Assumptions 1-2, and 5-6 hold, then E[d1 (Qn , Q⋆ )] ≲ νn . — p. 48
- Lemma 2. For any kernel scoring rule, and for all x ∈ X : (i) the score Pθ 7→ S(Pθ , x) is — p. 51
- Lemma 3. Under Assumptions 1-3, for any λn ≥ dS /r0 , we have — p. 52
- Lemma 4. Suppose that Assumption 1 holds and further that the score is bounded from — p. 54

**Figures / Tables (labels as written) + page numbers**
- Figure 1: The Gibbs and PrO posterior distributions under different forms of model misspecifica- — p. 10
- Figure 2: Mean and standard error of Gibbs posterior’s predictive loss relative to PrO posteriors — p. 11
- Figure 3: Illustration of key ideas introduced in Definition 2. Panel (a) depicts trivial model mis- — p. 13
- Figure 4: Palmer penguins example. We plot both the kernel density estimate (KDE) and the — p. 19
- Figure 5: Golf putting data. Points represent the proportion of successful putts made by professional — p. 21
- Figure 6: Top row : the mean (solid line) and one standard error intervals (shaded area) of the — p. 23
- Figure 7: The distribution of PrO posterior log-odds for the golf putting data. The black dots — p. 26
- Figure 8: Top row: Bayes and PrO posterior predictive distributions overlaid with the raw data. — p. 28
- Figure 9: Constructing Bayes and PrO posteriors using n observations, we plot their negative log — p. 28
- Figure 10: River flow data. In the left column we show the water flow (log scaled) in the winter — p. 30
- Figure 11: Top left: illustration of the Boston housing data set. Nodes correspond to census tracts, — p. 32
- Figure 12: In the left-most column we show the particle trajectories from a Wasserstein gradient flow — p. 65
- Figure 13: A comparison of PrO and Gibbs posterior predictive distributions in a well-specified — p. 66
- Figure 14: Synthetic linear regression example. — p. 69
- Figure 15: The empirical data distribution of the median housing value by census tract. — p. 71

**Notation Map (symbol -> first page found)**
- λ_n — not found
- λn — p. 6
- Q_n — not found
- Qn — p. 6
- W(Q) — p. 32
- p — p. 1
- dt — p. 33
- B — p. 1
- K — p. 1

**Keyword Hits (keyword -> pages)**
- Algorithm — not found
- Definition — p. 5, 12, 13, 26, 44, 45
- Proposition — p. 55, 56
- Theorem — p. 7, 16, 19, 27, 36, 42, 44, 45, 46, 47, 56, 59, 60, 61, 62, 63, 73
- Appendix — p. 7, 11, 15, 19, 29, 30, 33, 49, 73
- Figure — p. 10, 11, 12, 13, 18, 19, 21, 22, 23, 25, 26, 27, 28, 29, 30, 31, 32, 64, 65, 66, 67, 68, 69, 70, 71, 72
- Table — p. 49
- Proof — p. 49, 51, 52, 55, 56, 59, 60, 61, 63
- Euler–Maruyama — not found
- Euler-Maruyama — not found
- particle system — p. 32, 33, 34
- McKean — p. 32
- SDE — p. 33, 72
- time average — not found
- burn-in — p. 33