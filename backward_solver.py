"""2D backward (adjoint) solver for the viscous Cahn–Hilliard system.

This module expects a pre-computed forward solution `phi_hist` over a
two-dimensional grid and marches the adjoint variables `p`, `q`, and `r`
backward in time.  All spatial operators are assembled with SciPy sparse
matrices to handle larger grids efficiently.

The forward solver provides the material parameters `c1`, `c2`, `tau`, and
`gamma`, as well as the 2‑D Neumann Laplacian builder
`laplacian_matrix_neumann`.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sps
from scipy.sparse.linalg import spsolve
from typing import Optional, Tuple

from Forward_solver import (
    laplacian_matrix_neumann,
    c1,
    c2,
    tau,
    gamma,
)


def fpp_log(phi: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Second derivative of the logarithmic potential with safety clipping."""

    ph = np.clip(phi, -1 + eps, 1 - eps)
    return 2.0 * c1 / (1.0 - ph**2) - 2.0 * c2


def run_backward(
    phi_hist: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    t_hist: np.ndarray,
    b1: float,
    b2: float,
    phi_Q: Optional[np.ndarray] = None,
    phi_T_target: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve the adjoint system backward in time on a 2‑D grid.

    Parameters
    ----------
    phi_hist:
        Array of shape ``(M+1, Nx+1, Ny+1)`` containing the forward solution.
    x, y:
        1‑D spatial grids of length ``Nx+1`` and ``Ny+1`` respectively.
    t_hist:
        Monotonically increasing time levels of length ``M+1``.
    b1, b2:
        Objective functional weights.
    phi_Q, phi_T_target:
        Optional target fields with the same shapes as ``phi_hist`` and the
        final ``phi`` slice respectively.
    """

    M_plus1, Nx1, Ny1 = phi_hist.shape
    Nx = Nx1 - 1
    Ny = Ny1 - 1
    hx = x[1] - x[0]
    hy = y[1] - y[0]
    Nloc = (Nx + 1) * (Ny + 1)

    # Flatten spatial fields for linear algebra operations
    phi_hist_f = phi_hist.reshape(M_plus1, Nloc)

    if phi_Q is None:
        phi_Q_f = np.zeros_like(phi_hist_f)
    else:
        phi_Q_f = phi_Q.reshape(M_plus1, Nloc)

    if phi_T_target is None:
        phi_T_f = np.zeros(Nloc)
    else:
        phi_T_f = phi_T_target.reshape(Nloc)

    # --- Grid operators ---
    L = laplacian_matrix_neumann(Nx, Ny, hx, hy)  # sparse CSR
    L2 = L @ L
    I = sps.eye(Nloc, format="csr")

    # --- Allocate adjoint arrays (flattened) ---
    p = np.zeros((M_plus1, Nloc))
    q = np.zeros_like(p)
    r = np.zeros_like(p)

    # --- Terminal conditions at t = T ---
    rhs_T = b2 * (phi_hist_f[-1] - phi_T_f)
    p[-1] = spsolve(I - tau * L, rhs_T)
    q[-1] = -L @ p[-1]
    r[-1] = np.zeros(Nloc)

    # --- Adjoint operators ---
    def A_adjoint(phi_n: np.ndarray, dt_n: float) -> sps.csr_matrix:
        fpp_vals = fpp_log(phi_n)
        return I - tau * L + 0.5 * dt_n * L2 - 0.5 * dt_n * (sps.diags(fpp_vals) @ L)

    def B_adjoint(phi_np1: np.ndarray, dt_n: float) -> sps.csr_matrix:
        fpp_vals = fpp_log(phi_np1)
        return I - tau * L - 0.5 * dt_n * L2 + 0.5 * dt_n * (sps.diags(fpp_vals) @ L)

    # --- Backward time march ---
    for n in range(M_plus1 - 2, -1, -1):
        dt_n = t_hist[n + 1] - t_hist[n]
        if dt_n <= 0:
            continue

        src = 0.5 * dt_n * b1 * (
            (phi_hist_f[n] - phi_Q_f[n]) + (phi_hist_f[n + 1] - phi_Q_f[n + 1])
        )
        rhs = B_adjoint(phi_hist_f[n + 1], dt_n) @ p[n + 1] + src

        try:
            p[n] = spsolve(A_adjoint(phi_hist_f[n], dt_n), rhs)
        except Exception:
            p[n] = spsolve(A_adjoint(phi_hist_f[n], dt_n) + 1e-10 * I, rhs)

        q[n] = -L @ p[n]

        gamma_factor_back = (gamma - 0.5 * dt_n) / (gamma + 0.5 * dt_n)
        gamma_factor_source = (0.5 * dt_n) / (gamma + 0.5 * dt_n)
        r[n] = gamma_factor_back * r[n + 1] + gamma_factor_source * (q[n] + q[n + 1])

    # Reshape adjoints back to (M+1, Nx+1, Ny+1)
    shape3d = (M_plus1, Nx + 1, Ny + 1)
    return p.reshape(shape3d), q.reshape(shape3d), r.reshape(shape3d)

