# time-emergence
Experiments about "TIME" itself
## Background & Rationale  

### Why “Time” in Information Fields?  
Classical physics frames time as an external, ever-advancing axis;  
information theory largely sidesteps the issue by treating time as a  
sequence index.  Yet many complex systems—neural networks, social  
graphs, distributed ledgers—evolve on *graphs of information exchange*,  
not on a simple line.  This has sparked two speculative claims:

1. **Entropy-Arrow Claim** – in any self-updating information field,  
   global entropy should increase monotonically, mimicking an  
   intrinsic “arrow of time.”  
2. **Speed-of-Time Claim** – the mean magnitude of state change  
   \\(|\Delta S|\\) should control, or at least correlate with, the  
   *rate* at which that arrow is perceived.

Both claims are attractive: they promise a bridge between Shannon’s  
static entropy and dynamical “flow of time” concepts.  They also echo  
folk intuitions (“more happening ⇒ time feels faster”) and some  
popular-science accounts of the thermodynamic arrow.

## Formal Conjectures

| Symbol | Meaning |
|--------|---------|
| `$X_t \in \mathbb{R}^{N\times D}$` | State matrix of all *N* agents at time *t* |
| `$\lvert\Delta S_t\rvert = \lVert X_t - X_{t-1}\rVert_2$` | Mean L2-magnitude of state change (“information-flow speed”) |
| `$H(t)$` | Shannon entropy of the empirical state distribution |
| `$\rho(\cdot,\cdot)$` | Pearson correlation (or Spearman $\rho_s$) |

### Conjecture H1 – *Information-Emergence-of-Time (IETC)*  

> The global entropy of a closed information field never decreases.

\[
\frac{dH}{dt} \;\ge\; 0 \qquad(\forall\, t\ge0)
\]

### Conjecture H2 – *Multidimensional Information–Time Perception (MITPC)*  

> A larger instantaneous information-flow speed implies a larger entropy.

\[
\rho\!\bigl(\lvert\Delta S_t\rvert,\; H(t)\bigr) \;>\; 0
\]

*(If one only expects a monotone—rather than linear—relation,  
replace $\rho$ with Spearman’s $\rho_s$ or use mutual information.)*

---


### Why Test Them Empirically?  
*Neither* claim has rigorous proof; both are implicitly assumed in some  
complex-systems literature.  Before investing effort in theoretical  
proofs, we chose a comprehensive falsification campaign:

* **General-enough model** – agents on three canonical graph topologies,  
  with deterministic oscillation + Gaussian noise + attention-like  
  neighbour coupling.  
* **Wide parameter sweep** – nine σ–noise levels, two dimensions, two  
  epoch lengths, 10 random seeds each.  
* **Multiple entropy lenses** – Shannon, Rényi (α = 2), Tsallis  
  (q = 1.5).  
* **Linear & non-linear stats** – Pearson/Spearman for linear/monotone  
  trends and bin-based mutual information for arbitrary dependence.

> *Goal:* if the hypotheses hold, they should survive this barrage; if  
> they fail in most cases, we have strong empirical grounds to doubt  
> their universality.

### Experimental Storyboard (summary)  
1. **P₀ ∶ toy smoke test** – ensure code runs.  
2. **P₁ ∶ simple graph** – baseline check of H1/H2.  
3. **P₂ ∶ add topology & attention** – realism bump.  
4. **P₃ ∶ 6-config matrix** – identify “interesting pockets.”  
5. **P₄ ∶ alt-entropy + MI** – rule out non-linear loopholes.  
6. **P₅ ∶ 36-config exhaustive grid** – broad falsification.  
7. **P₆ ∶ very long runs (10k–50k steps)** – detect late-time drift.

Each phase refined code (memory safety, sampling stride, new metrics)  
while widening empirical coverage.

---

# Information-Field Time Experiments  
*Falsifying two intuitive hypotheses about “time” in information systems*  

---

## 1  Initial Hypotheses  

| ID | Statement | Short name |
|----|-----------|------------|
| **H1** | **Entropy-Arrow Conjecture** – When an information field evolves autonomously, its global entropy will *monotonically* increase, creating an intrinsic “arrow of time.” | `entropy_arrow` |
| **H2** | **Speed-of-Time Conjecture** – The larger the mean state-change magnitude \\(|\Delta S|\\) in the field, the faster the “perceived” flow of time; mathematically, entropy and \\(|\Delta S|\\) should be positively correlated. | `delta_entropy_corr` |

---

## 2  Experimental Road-map  

| Phase | Goal | Key changes |
|-------|------|-------------|
| **P₀** | Smoke-test tiny CA → confirm code path | baseline (↗) |
| **P₁** | Verify H1/H2 on *one* config | `experiment_extend_2.py`<br>• Shannon entropy & \\(|\Delta S|\\) |
| **P₂** | **Add graph topology + attention** | `experiment_extend_4.py`<br>• Watts–Strogatz, Barabási, Erdős–Rényi |
| **P₃** | Parameter sweep (σ, d, steps) | `run_matrix2.py` (6 configs × 5 seeds) |
| **P₄** | Extra entropy & mutual-information | `experiment_extend_6.py`<br>• Rényi (α=2), Tsallis (q=1.5)<br>• Mutual Information \(I(|\Delta S|;H)\) |
| **P₅** | Wide grid (3 graphs × 3 σ × 2 d × 2 steps) | `run_grid6.py` → 36 configs × 10 seeds |
| **P₆** | Long-range runs (10 k–50 k steps) & stride sampling | inside `experiment_extend_6.py` |

---

## 3  Key Optimisations  

* **Memory-safe entropy** – High-dim cut-off: if bins^d > 2 M, switch to “per-dimension entropy mean.”  
* **Stride sampling** – Record entropy/\\(|\Delta S|\\) every *k* steps to keep arrays small.  
* **Streamed batch** – `experiment_extend_3.py`: one subprocess per seed ⇒ RAM auto-released.  
* **Extra metrics** – `metrics_extras.py` supplies Rényi, Tsallis, Mutual-Info helpers.  

---

## 4  Result Presentation Assets  

| File | Description |
|------|-------------|
| **`pearson_ci.png`** | 36-bar chart of μ ± 95 % CI for Pearson r |
| `mi_heat_ws/ba/er.png` | 3 heat-maps (σ vs d) of mean mutual information |
| `summary_all.csv` | per-config seed-level stats |
| `aggregate_all.json` | μ & CI for each metric |
| `grid6_results.zip` | full raw output (uploaded) |

---

## 5  Summary of Findings  

### 5.1  Linear / Monotonic tests  
* `entropy_slope` 95 % CI across *all* 36 configs crosses 0 → **no monotone growth** ⇒ `entropy_arrow` **not supported**.  
* Pearson r: only **12 / 36 (33 %)** configs have CI not crossing 0, and these are confined to  
  *low-dim (d = 6) Watts–Strogatz with σ ≥ 0.3*.  
  In the remaining 67 % the sign is indeterminate → `delta_entropy_corr` **lacks universality**.

### 5.2  Non-linear checks  
* Mutual-information \(I(|\Delta S|;H)\) < 0.01 bit for every cell in the 3×3 heat-maps – strong evidence of *statistical independence*.

### 5.3  Robustness  
* Changing entropy measure (Shannon → Rényi / Tsallis) did **not** introduce an arrow.  
* Graph rewiring triggers periodic “entropy collapses,” but long-term mean unchanged.

---

## 6  Conclusions  

| Hypothesis | Verdict | Evidence |
|------------|---------|----------|
| **H1** Entropy-Arrow | **Falsified (within tested model family)** | 10 k–50 k step slopes ~0; no monotone rise under 99 % of configs |
| **H2** Speed-of-Time | **No universal support** | 67 % configs CI crosses 0; MI ≈ 0 |

> **Implication** – In stochastic information-field dynamics governed by local oscillation, Gaussian noise and graph-based attention, neither a built-in arrow of time nor a speed-of-time law emerges as a robust property.

---

## 7  Interesting Side-Notes  

* **Entropy collapses** coincide with graph rewiring events – akin to “punctuated equilibrium.”  
* Weak positive correlations only surface in *low-dim*, mid–high noise, small-world networks – hinting at geometric constraints rather than fundamental law.

---

## 8  Next-Step Ideas  

1. **Analytic proof** of an entropy upper-bound independent of \\(|\Delta S|\\).  
2. Test non-Gaussian noise (e.g., Lévy flight) or memory kernels.  
3. Introduce energy-like constraints to see if an arrow appears in dissipative systems.

---

### Reproduce Everything

```bash
# install deps
pip install -r requirements.txt   # numpy pandas scipy seaborn torch networkx sklearn

# 1. run grid (≈360 runs)
python -m src.run_grid6

# 2. analyse & create plots
python -m src.analyze_grid6

