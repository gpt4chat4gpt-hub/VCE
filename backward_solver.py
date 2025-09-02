import numpy as np
from typing import Tuple, Optional
from Forward_solver import (
    laplacian_matrix_neumann,
    c1, c2, tau, gamma,
)


def laplacian_matrix_neumann_2d(nx: int, ny: int, hx: float, hy: float) -> np.ndarray:
    """Build 2D Neumann Laplacian using Kronecker products."""
    Lx = laplacian_matrix_neumann(nx, hx)
    Ly = laplacian_matrix_neumann(ny, hy)
    Ix = np.eye(nx + 1)
    Iy = np.eye(ny + 1)
    return np.kron(Iy, Lx) + np.kron(Ly, Ix)


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
    """Solve the adjoint system backward in time for a 2D domain."""

    M_plus1, Ny_plus1, Nx_plus1 = phi_hist.shape

    if phi_Q is None:
        phi_Q = np.zeros_like(phi_hist)
    if phi_T_target is None:
        phi_T_target = np.zeros((Ny_plus1, Nx_plus1))

    hx = x[1] - x[0]
    hy = y[1] - y[0]
    L = laplacian_matrix_neumann_2d(Nx_plus1 - 1, Ny_plus1 - 1, hx, hy)
    L2 = L @ L
    I = np.eye(Nx_plus1 * Ny_plus1)

    p = np.zeros_like(phi_hist)
    q = np.zeros_like(phi_hist)
    r = np.zeros_like(phi_hist)

    rhs_T = b2 * (phi_hist[-1] - phi_T_target)
    rhs_T_flat = rhs_T.reshape(-1)
    p_T_flat = np.linalg.solve(I - tau * L, rhs_T_flat)
    p[-1] = p_T_flat.reshape(Ny_plus1, Nx_plus1)
    q[-1] = (-L @ p_T_flat).reshape(Ny_plus1, Nx_plus1)
    r[-1] = np.zeros((Ny_plus1, Nx_plus1))

    def A_adjoint(phi_n_flat: np.ndarray, dt_n: float) -> np.ndarray:
        fpp_diag = np.diag(fpp_log(phi_n_flat))
        return I - tau * L + 0.5 * dt_n * L2 - 0.5 * dt_n * (fpp_diag @ L)

    def B_adjoint(phi_np1_flat: np.ndarray, dt_n: float) -> np.ndarray:
        fpp_diag = np.diag(fpp_log(phi_np1_flat))
        return I - tau * L - 0.5 * dt_n * L2 + 0.5 * dt_n * (fpp_diag @ L)

    for n in range(M_plus1 - 2, -1, -1):
        dt_n = t_hist[n + 1] - t_hist[n]
        if dt_n <= 0:
            continue

        src = 0.5 * dt_n * b1 * (
            (phi_hist[n] - phi_Q[n]) + (phi_hist[n + 1] - phi_Q[n + 1])
        )
        src_flat = src.reshape(-1)
        phi_np1_flat = phi_hist[n + 1].reshape(-1)
        rhs = B_adjoint(phi_np1_flat, dt_n) @ p[n + 1].reshape(-1) + src_flat

        try:
            p_n_flat = np.linalg.solve(A_adjoint(phi_hist[n].reshape(-1), dt_n), rhs)
        except np.linalg.LinAlgError:
            p_n_flat = np.linalg.solve(
                A_adjoint(phi_hist[n].reshape(-1), dt_n) + 1e-10 * I, rhs
            )

        p[n] = p_n_flat.reshape(Ny_plus1, Nx_plus1)
        q[n] = (-L @ p_n_flat).reshape(Ny_plus1, Nx_plus1)

        gamma_factor_back = (gamma - 0.5 * dt_n) / (gamma + 0.5 * dt_n)
        gamma_factor_source = (dt_n * 0.5) / (gamma + 0.5 * dt_n)
        r[n] = gamma_factor_back * r[n + 1] + gamma_factor_source * (
            q[n] + q[n + 1]
        )

    return p, q, r
