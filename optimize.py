"""
coil_shape_optimizer.py
=======================
Differentiable Fourier Cross-Section Shape Optimisation
for Stellarator Coil Winding Packs

Pipeline
--------
Fourier coefficients (nn.Parameters)
    → 2D closed cross-section  r(θ) = a₀ + Σ [aₙ cos nθ + bₙ sin nθ]
    → geometric quantities (all differentiable via PyTorch autograd):
        · cross_section_area()    — shoelace / Green's theorem
        · curvature_energy()      — ∫ κ²(s) ds  [bend-radius proxy]
        · soft_clearance()        — log-sum-exp soft-min to neighbour
    → Adam optimiser (gradient-based shape update)
    → 3D toroidal sweep → visualisation + sensitivity analysis

Why Fourier parametrisation?
    The same representation is standard for stellarator plasma boundaries
    and coil winding pack cross-sections (e.g. VMEC, DESC). Making it
    a differentiable nn.Module means any downstream engineering quantity
    — structural loads, clearances, manufacturing constraints — can be
    optimised with gradient descent instead of manual iteration.

Requirements: torch, numpy, matplotlib
"""

import copy
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")   # headless rendering
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401  (registers projection)

torch.manual_seed(42)
np.random.seed(42)


# ═══════════════════════════════════════════════════════════════
# 1.  PARAMETRIC GEOMETRY
# ═══════════════════════════════════════════════════════════════

class FourierCrossSection(nn.Module):
    """
    Closed 2-D curve in polar Fourier form:

        r(θ) = a₀  +  Σₙ [ aₙ cos(nθ) + bₙ sin(nθ) ]
        x(θ) = r(θ) cos θ,   y(θ) = r(θ) sin θ

    All operations are differentiable; gradients of area, curvature,
    or clearance propagate back to (a₀, aₙ, bₙ) without finite
    differences.
    """

    def __init__(self, n_modes: int = 8, r0: float = 0.8):
        super().__init__()
        self.n_modes = n_modes
        self.a0 = nn.Parameter(torch.tensor([r0]))
        self.an = nn.Parameter(torch.zeros(n_modes))
        self.bn = nn.Parameter(torch.zeros(n_modes))

    def forward(self, theta: torch.Tensor):
        r = self.a0.expand_as(theta).clone()
        for n in range(1, self.n_modes + 1):
            r = r + self.an[n - 1] * torch.cos(n * theta) \
                  + self.bn[n - 1] * torch.sin(n * theta)
        return r * torch.cos(theta), r * torch.sin(theta)

    def sample(self, n_pts: int = 300):
        """Sample n_pts equally-spaced points on [0, 2π)."""
        theta = torch.linspace(0, 2 * np.pi, n_pts + 1)[:-1]
        return self.forward(theta)


# ═══════════════════════════════════════════════════════════════
# 2.  DIFFERENTIABLE GEOMETRIC QUANTITIES
# ═══════════════════════════════════════════════════════════════

def cross_section_area(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Exact area via the shoelace formula (discrete Green's theorem).
    O(n), fully differentiable.
    """
    x1 = torch.roll(x, -1)
    y1 = torch.roll(y, -1)
    return 0.5 * torch.abs(torch.sum(x * y1 - x1 * y))


def curvature_energy(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Integrated squared curvature   E = ∫ κ(s)² ds

    Differentiable proxy for the minimum-bend-radius manufacturing
    constraint on HTS coil tape. Sharp bends → high E; smooth curves → low E.

    Discretisation: central finite differences on the parametric curve.
    """
    dx  = torch.roll(x, -1) - torch.roll(x,  1)   # x'  (central diff)
    dy  = torch.roll(y, -1) - torch.roll(y,  1)   # y'
    d2x = torch.roll(x, -1) - 2 * x + torch.roll(x, 1)  # x''
    d2y = torch.roll(y, -1) - 2 * y + torch.roll(y, 1)  # y''

    speed_sq = (dx ** 2 + dy ** 2).clamp(min=1e-12)
    kappa    = (dx * d2y - dy * d2x) / speed_sq.pow(1.5)
    ds       = speed_sq.sqrt()
    return torch.sum(kappa ** 2 * ds)


def soft_clearance(
    x1: torch.Tensor, y1: torch.Tensor,
    x2: torch.Tensor, y2: torch.Tensor,
    beta: float = 25.0,
) -> torch.Tensor:
    """
    Differentiable approximation of the minimum inter-curve distance.

    Uses the log-sum-exp softmin:
        soft_min(d) ≈ -log Σ exp(-β · d_ij) / β

    This is the standard trick for differentiable clearance/collision
    constraints in coil optimisation: smooth everywhere, exact in the
    limit β → ∞.
    """
    dx = x1[:, None] - x2[None, :]   # (n1, n2)
    dy = y1[:, None] - y2[None, :]
    d  = (dx ** 2 + dy ** 2).clamp(min=1e-12).sqrt()
    return -torch.logsumexp(-beta * d, dim=(0, 1)) / beta


# ═══════════════════════════════════════════════════════════════
# 3.  OPTIMISATION
# ═══════════════════════════════════════════════════════════════

def run_optimisation(
    n_modes: int         = 8,
    n_steps: int         = 500,
    lr: float            = 5e-3,
    target_area: float   = 1.8,     # conductor cross-section area [m²]
    min_gap: float       = 0.35,    # minimum coil-to-coil clearance [m]
    w_area: float        = 100.0,   # area constraint weight
    w_gap: float         = 250.0,   # clearance constraint weight
    n_pts: int           = 300,     # discretisation resolution
    snapshot_every: int  = 50,
):
    """
    Minimise ∫ κ² ds  subject to:
        |A − A*|  ≤ ε       (conductor volume / cross-section area)
        d_min ≥ d*          (coil-to-coil clearance)

    Both constraints are enforced as quadratic penalties; gradients
    are available everywhere in parameter space.

    Returns
    -------
    cs_init  : initial FourierCrossSection (deep-copied before optimisation)
    cs_opt   : optimised FourierCrossSection
    log      : dict of per-step scalars
    snaps    : list of (x, y, step) shape snapshots
    nb_pts   : (xnb, ynb) numpy arrays for the fixed neighbour coil
    """
    cs = FourierCrossSection(n_modes=n_modes, r0=0.8)
    # Initialise with an irregular, high-curvature shape so the optimiser
    # has meaningful curvature reduction to perform (circular start is
    # already near-optimal for ∫κ²ds).
    with torch.no_grad():
        cs.an.data = torch.tensor([0.05, -0.18, 0.22, -0.12, 0.09, -0.06, 0.14, -0.08])
        cs.bn.data = torch.tensor([0.12,  0.15, -0.10,  0.18, -0.07, 0.11, -0.05,  0.13])
    cs_init = copy.deepcopy(cs)

    # Fixed neighbouring coil: slight ellipse centred at (2.2, 0)
    cs_nb = FourierCrossSection(n_modes=2, r0=0.60)
    with torch.no_grad():
        cs_nb.an.data[0] = 0.12    # minor ellipticity
    for p in cs_nb.parameters():
        p.requires_grad_(False)

    with torch.no_grad():
        xnb, ynb = cs_nb.sample(n_pts)
        xnb = xnb + 2.2    # offset to (2.2, 0)

    optimiser = torch.optim.Adam(cs.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, n_steps)

    log   = {"loss": [], "curv": [], "area": [], "gap": []}
    snaps = []

    for step in range(n_steps):
        optimiser.zero_grad()

        x, y    = cs.sample(n_pts)
        A       = cross_section_area(x, y)
        E_curv  = curvature_energy(x, y)
        d_min   = soft_clearance(x, y, xnb, ynb)

        loss = (E_curv
                + w_area * (A - target_area) ** 2
                + w_gap  * torch.relu(min_gap - d_min) ** 2)

        loss.backward()
        optimiser.step()
        scheduler.step()

        log["loss"].append(loss.item())
        log["curv"].append(E_curv.item())
        log["area"].append(A.item())
        log["gap"].append(d_min.item())

        if step % snapshot_every == 0 or step == n_steps - 1:
            with torch.no_grad():
                xi, yi = cs.sample(n_pts)
            snaps.append((xi.numpy().copy(), yi.numpy().copy(), step))

    return cs_init, cs, log, snaps, (xnb.numpy(), ynb.numpy())


# ═══════════════════════════════════════════════════════════════
# 4.  3-D TOROIDAL SWEEP
# ═══════════════════════════════════════════════════════════════

def toroidal_sweep(
    cs: FourierCrossSection,
    R: float   = 5.0,    # torus major radius
    n_phi: int = 72,     # toroidal discretisation
    n_cs: int  = 200,    # cross-section discretisation
) -> np.ndarray:
    """
    Sweep a 2-D cross-section along a circular (toroidal) path.

    At each toroidal angle φ the local frame is:
        e_r = (cos φ, sin φ, 0)   [outward radial]
        e_z = (0, 0, 1)           [axial]

    giving the parametric surface:
        P(φ, θ) = ((R + x(θ)) cos φ,  (R + x(θ)) sin φ,  y(θ))

    Returns array of shape (n_phi, n_cs, 3).
    """
    phi_arr = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    with torch.no_grad():
        xc, yc = cs.sample(n_cs)
    xc, yc = xc.numpy(), yc.numpy()

    surf = np.empty((n_phi, n_cs, 3))
    for i, phi in enumerate(phi_arr):
        surf[i, :, 0] = (R + xc) * np.cos(phi)
        surf[i, :, 1] = (R + xc) * np.sin(phi)
        surf[i, :, 2] = yc
    return surf


# ═══════════════════════════════════════════════════════════════
# 5.  SENSITIVITY ANALYSIS
# ═══════════════════════════════════════════════════════════════

def compute_sensitivities(cs: FourierCrossSection, n_pts: int = 300):
    """
    Compute ∂E_curv / ∂aₙ  and  ∂E_curv / ∂bₙ  at the current parameters.

    Identifies which Fourier modes most strongly couple to the smoothness
    objective — useful for design-space reduction in high-dimensional
    parametric searches.
    """
    cs_tmp = copy.deepcopy(cs)
    x, y   = cs_tmp.sample(n_pts)
    E      = curvature_energy(x, y)
    E.backward()
    return (
        cs_tmp.an.grad.detach().numpy(),
        cs_tmp.bn.grad.detach().numpy(),
    )


# ═══════════════════════════════════════════════════════════════
# 6.  VISUALISATION
# ═══════════════════════════════════════════════════════════════

def _close(arr):
    """Append first element to close a curve for plotting."""
    return np.append(arr, arr[0])


def plot_optimisation_results(log, snaps, xnb, ynb, out="optimisation_results.png"):
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(
        "Differentiable Coil Cross-Section Optimisation  —  "
        "Fourier Parametrisation + PyTorch Autograd",
        fontsize=13, fontweight="bold",
    )

    # ── Shape evolution ──────────────────────────────────────
    ax = axes[0, 0]
    cmap_vals = plt.cm.plasma(np.linspace(0.1, 0.92, len(snaps)))
    for (xi, yi, step), c in zip(snaps, cmap_vals):
        lw  = 2.4 if step == snaps[-1][2] else 0.9
        lbl = f"step {step}" if step in (snaps[0][2], snaps[-1][2]) else None
        ax.plot(_close(xi), _close(yi), color=c, lw=lw, label=lbl)
    ax.plot(_close(xnb), _close(ynb), "k--", lw=1.5, label="neighbour (fixed)")
    ax.set_aspect("equal")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_title("Cross-section shape evolution", fontsize=10)
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.grid(alpha=0.25)

    # ── Total loss ───────────────────────────────────────────
    ax = axes[0, 1]
    ax.semilogy(log["loss"], "C0", lw=1.2)
    ax.set_title("Total loss (log scale)", fontsize=10)
    ax.set_xlabel("iteration"); ax.grid(alpha=0.25)

    # ── Curvature energy ─────────────────────────────────────
    ax = axes[0, 2]
    ax.semilogy(log["curv"], "C1", lw=1.2)
    ax.set_title("Curvature energy  ∫κ²ds", fontsize=10)
    ax.set_xlabel("iteration"); ax.grid(alpha=0.25)

    # ── Area constraint ──────────────────────────────────────
    ax = axes[1, 0]
    ax.plot(log["area"], "C2", lw=1.2)
    ax.axhline(1.8, ls="--", color="k", lw=1.0, label="target A* = 1.8")
    ax.set_title("Cross-section area", fontsize=10)
    ax.set_xlabel("iteration"); ax.legend(fontsize=8); ax.grid(alpha=0.25)

    # ── Clearance constraint ─────────────────────────────────
    ax = axes[1, 1]
    ax.plot(log["gap"], "C3", lw=1.2)
    ax.axhline(0.35, ls="--", color="k", lw=1.0, label="min gap d* = 0.35")
    ax.set_title("Soft-min clearance to neighbour", fontsize=10)
    ax.set_xlabel("iteration"); ax.legend(fontsize=8); ax.grid(alpha=0.25)

    # ── Annotate final values ────────────────────────────────
    ax = axes[1, 2]
    ax.axis("off")
    rows = [
        ("Quantity",            "Initial",     "Optimised"),
        ("Curvature energy",    f"{log['curv'][0]:.4f}",  f"{log['curv'][-1]:.4f}"),
        ("Area",                f"{log['area'][0]:.4f}",  f"{log['area'][-1]:.4f}"),
        ("Coil clearance",      f"{log['gap'][0]:.4f}",   f"{log['gap'][-1]:.4f}"),
    ]
    table = ax.table(
        cellText=rows[1:], colLabels=rows[0],
        cellLoc="center", loc="center",
        bbox=[0.0, 0.25, 1.0, 0.65],
    )
    table.auto_set_font_size(False); table.set_fontsize(10)
    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_facecolor("#2a2a2a"); cell.set_text_props(color="white")
    ax.set_title("Summary", fontsize=10)

    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def plot_3d_winding_pack(cs_init, cs_opt, out="3d_winding_pack.png"):
    fig = plt.figure(figsize=(14, 6))
    fig.suptitle(
        "3-D Toroidal Winding-Pack  —  Initial vs Optimised Cross-Section",
        fontsize=12, fontweight="bold",
    )

    for idx, (cs, title) in enumerate(
        [(cs_init, "Initial  (circular Fourier)"),
         (cs_opt,  "Optimised  (gradient-descent Fourier)")], 1
    ):
        ax  = fig.add_subplot(1, 2, idx, projection="3d")
        surf = toroidal_sweep(cs, R=5.0, n_phi=72, n_cs=200)

        # toroidal rings (cross-section at each φ)
        for i in range(0, surf.shape[0], 6):
            x = np.append(surf[i, :, 0], surf[i, 0, 0])
            y = np.append(surf[i, :, 1], surf[i, 0, 1])
            z = np.append(surf[i, :, 2], surf[i, 0, 2])
            ax.plot(x, y, z, "C0", lw=0.5, alpha=0.55)

        # longitudinal lines (along-torus direction)
        for j in range(0, surf.shape[1], 18):
            ax.plot(surf[:, j, 0], surf[:, j, 1], surf[:, j, 2],
                    "C1", lw=0.5, alpha=0.55)

        ax.set_title(title, fontsize=10)
        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
        ax.set_box_aspect([1, 1, 0.22])
        ax.tick_params(labelsize=7)

    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def plot_sensitivities(grad_an, grad_bn, n_modes, out="sensitivity.png"):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    modes = np.arange(1, n_modes + 1)

    axes[0].bar(modes, np.abs(grad_an), color="C0", edgecolor="k", lw=0.4)
    axes[0].set_title("|∂E_curv / ∂aₙ|  —  cosine mode sensitivities", fontsize=10)
    axes[0].set_xlabel("Fourier mode n"); axes[0].set_ylabel("|gradient|")
    axes[0].set_xticks(modes); axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(modes, np.abs(grad_bn), color="C2", edgecolor="k", lw=0.4)
    axes[1].set_title("|∂E_curv / ∂bₙ|  —  sine mode sensitivities", fontsize=10)
    axes[1].set_xlabel("Fourier mode n")
    axes[1].set_xticks(modes); axes[1].grid(axis="y", alpha=0.3)

    fig.suptitle(
        "Curvature-Energy Sensitivity to Fourier Modes  (via autograd)",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  Fourier Coil Cross-Section Optimiser")
    print("  Case study — Proxima Fusion, Computational Engineer")
    print("=" * 60)

    print("\n[1/4] Running optimisation (500 steps) ...")
    cs_init, cs_opt, log, snaps, (xnb, ynb) = run_optimisation(
        n_modes=8,
        n_steps=500,
        lr=5e-3,
        target_area=1.8,
        min_gap=0.35,
    )

    reduction = (log["curv"][0] - log["curv"][-1]) / log["curv"][0] * 100
    print(f"\n{'Metric':<30} {'Initial':>10}  {'Final':>10}")
    print("-" * 54)
    print(f"{'Curvature energy':.<30} {log['curv'][0]:>10.4f}  {log['curv'][-1]:>10.4f}  ({reduction:.1f}% reduction)")
    print(f"{'Area (target = 1.8)':.<30} {log['area'][0]:>10.4f}  {log['area'][-1]:>10.4f}")
    print(f"{'Clearance (min = 0.35)':.<30} {log['gap'][0]:>10.4f}  {log['gap'][-1]:>10.4f}")

    print("\n[2/4] Generating optimisation plot ...")
    plot_optimisation_results(log, snaps, xnb, ynb)

    print("\n[3/4] Generating 3-D winding-pack sweep ...")
    plot_3d_winding_pack(cs_init, cs_opt)

    print("\n[4/4] Running sensitivity analysis ...")
    grad_an, grad_bn = compute_sensitivities(cs_opt)
    plot_sensitivities(grad_an, grad_bn, n_modes=8)

    print("\nAll outputs written. Repository ready.")
