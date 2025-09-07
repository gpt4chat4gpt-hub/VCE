

# In backward_solver_fixed.py
import numpy as np
from typing import Tuple, Optional
from Forward_solver import (
    laplacian_matrix_neumann,
    c1, c2, tau, gamma,
)

def fpp_log(phi: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Second derivative of the logarithmic potential with safety clipping"""
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
    """
    Solve the adjoint system backward in time using a pre-computed forward solution
    on a 2-D grid.
    """
    M_plus1, Nx_plus1, Ny_plus1 = phi_hist.shape
    n_nodes = Nx_plus1 * Ny_plus1

    if phi_Q is None:
        phi_Q = np.zeros_like(phi_hist)
    if phi_T_target is None:
        phi_T_target = np.zeros((Nx_plus1, Ny_plus1))

    hx = x[1] - x[0]
    hy = y[1] - y[0]
    dts = np.diff(t_hist)
    L = laplacian_matrix_neumann(Nx_plus1 - 1, Ny_plus1 - 1, hx, hy)
    L2 = L @ L
    I = np.eye(n_nodes)

    p = np.zeros((M_plus1, n_nodes))
    q = np.zeros((M_plus1, n_nodes))
    r = np.zeros((M_plus1, n_nodes))
    
    # --- Terminal Conditions at t=T ---
    rhs_T = b2 * (phi_hist[-1].reshape(-1) - phi_T_target.reshape(-1))
    p[-1] = np.linalg.solve(I - tau * L, rhs_T)
    q[-1] = -L @ p[-1]
    r[-1] = np.zeros(n_nodes)
    
    # --- Adjoint Operators ---
    def A_adjoint(phi_n, dt_n):
        fpp_diag = np.diag(fpp_log(phi_n))
        return I - tau * L + 0.5 * dt_n * L2 - 0.5 * dt_n * (fpp_diag @ L)

    def B_adjoint(phi_np1, dt_n):
        fpp_diag = np.diag(fpp_log(phi_np1))
        return I - tau * L - 0.5 * dt_n * L2 + 0.5 * dt_n * (fpp_diag @ L)

    # --- Backward Time March ---
    for n in range(M_plus1 - 2, -1, -1):
        dt_n = t_hist[n + 1] - t_hist[n]
        if dt_n <= 0:
            continue

        phi_n = phi_hist[n].reshape(-1)
        phi_np1 = phi_hist[n + 1].reshape(-1)
        src = 0.5 * dt_n * b1 * (
            (phi_n - phi_Q[n].reshape(-1)) + (phi_np1 - phi_Q[n + 1].reshape(-1))
        )
        rhs = B_adjoint(phi_np1, dt_n) @ p[n + 1] + src

        try:
            p[n] = np.linalg.solve(A_adjoint(phi_n, dt_n), rhs)
        except np.linalg.LinAlgError:
            p[n] = np.linalg.solve(A_adjoint(phi_n, dt_n) + 1e-10 * I, rhs)

        q[n] = -L @ p[n]

        gamma_factor_back = (gamma - 0.5 * dt_n) / (gamma + 0.5 * dt_n)
        gamma_factor_source = (dt_n * 0.5) / (gamma + 0.5 * dt_n)
        r[n] = gamma_factor_back * r[n + 1] + gamma_factor_source * (q[n] + q[n + 1])

    return (
        p.reshape(M_plus1, Nx_plus1, Ny_plus1),
        q.reshape(M_plus1, Nx_plus1, Ny_plus1),
        r.reshape(M_plus1, Nx_plus1, Ny_plus1),
    )