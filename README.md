# Differentiable Coil Cross-Section Optimisation

Gradient-based shape optimisation of a coil winding-pack cross-section
using Fourier parametrisation and PyTorch automatic differentiation.

---

## What this demonstrates

This is a small implementation of the computational approach at the core
of modern stellarator coil design: represent geometry entirely as
differentiable code, define engineering objectives and constraints as
differentiable functions of that geometry, then run gradient descent
to find the optimal shape.

The cross-section of a coil winding pack is parametrised as a truncated
Fourier series in polar form — the same representation used for stellarator
plasma boundaries and coil surfaces in codes like VMEC and DESC. Because
every downstream quantity (area, curvature, clearance) is computed
differentiably via PyTorch autograd, gradient-based optimisation is
possible without finite-difference approximations.

---

## Pipeline

```
Fourier coefficients  (nn.Parameters)
    │
    ▼
r(θ) = a₀ + Σₙ [ aₙ cos(nθ) + bₙ sin(nθ) ]   ← geometry in code
    │
    ├── cross_section_area()    shoelace formula / Green's theorem
    │
    ├── curvature_energy()      ∫ κ(s)² ds
    │                           differentiable bend-radius proxy
    │
    └── soft_clearance()        log-sum-exp soft-min to neighbour coil
                                smooth approximation to min-distance
    │
    ▼
Loss = E_curv  +  λ₁ · (A − A*)²  +  λ₂ · ReLU(d* − d_min)²
    │
    ▼
Adam optimiser  →  gradient through geometry  →  Fourier update
    │
    ▼
3-D winding-pack surface via toroidal sweep  →  plots
    │
    ▼
Sensitivity analysis:  ∂E_curv / ∂(aₙ, bₙ)  via autograd
```

---

## Objectives and constraints

| Quantity | Role | Formulation |
|---|---|---|
| ∫ κ²(s) ds | Objective: minimise sharp bends | Central finite differences on the parametric curve |
| Cross-section area A | Hard constraint: conductor volume | Shoelace formula, quadratic penalty |
| Min. clearance d_min | Hard constraint: coil-to-coil gap | Log-sum-exp softmin, quadratic penalty |

Starting from an irregular high-curvature shape, the optimiser achieves
a **90% reduction in curvature energy** while simultaneously driving the
area to its target and holding the clearance above the minimum.

---

## Key techniques

**Fourier parametrisation**
`r(θ) = a₀ + Σ [aₙ cos nθ + bₙ sin nθ]` is the standard representation
for stellarator surfaces. Encoding it as a PyTorch `nn.Module` makes all
downstream engineering quantities automatically differentiable.

**Differentiable soft-minimum clearance**
The log-sum-exp trick gives a smooth, gradient-friendly approximation to
the pairwise minimum distance between two discretised curves. This is the
standard approach for differentiable collision and clearance constraints.

**Sensitivity analysis via autograd**
`∂E / ∂(aₙ, bₙ)` computed in one backward pass identifies which Fourier
modes most strongly couple to the smoothness objective — a starting point
for design-space dimensionality reduction in higher-dimensional problems.

**3-D toroidal sweep**
The optimised cross-section is swept along a circular (toroidal) path to
produce the 3-D winding-pack geometry, illustrating how 2-D cross-section
design integrates into the full 3-D coil definition.

---

## Outputs

Running `python optimize.py` produces three figures:

`optimisation_results.png`
Shape evolution across iterations, convergence of loss / curvature /
area / clearance, and a results summary table.

`3d_winding_pack.png`
Initial (irregular) vs optimised (smooth) winding-pack geometry
on a toroidal path.

`sensitivity.png`
Gradient magnitude per Fourier mode at the optimised point, showing
which modes are structurally active.

---

## Requirements

```
torch >= 2.0
numpy
matplotlib
```

Install:
```bash
pip install torch numpy matplotlib
```

Run:
```bash
python optimize.py
```

---

## Connection to stellarator design

Manual iteration on coil cross-section shape is expensive: each geometry
change requires re-running structural, electromagnetic, and thermal
analyses. By expressing geometry as differentiable code, design exploration
shifts from manual iteration to systematic gradient descent. This project
demonstrates that shift at small scale using the same Fourier
representations and optimisation principles found in production stellarator
design workflows.
