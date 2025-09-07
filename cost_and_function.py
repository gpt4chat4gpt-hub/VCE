import numpy as np


def calculate_cost(
    phi_hist: np.ndarray,
    u: np.ndarray,
    phi_Q_target: np.ndarray,
    phi_T_target: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    t_hist: np.ndarray,
    b1: float,
    b2: float,
    b3: float,
    kappa: float,
) -> float:
    """Calculate the discrete cost functional for a 2-D spatial domain.

    Parameters
    ----------
    phi_hist : np.ndarray
        State history from the forward solver with shape ``(M+1, Nx+1, Ny+1)``.
    u : np.ndarray
        Control history with the same shape as ``phi_hist``.
    phi_Q_target : np.ndarray
        Space-time target with the same shape as ``phi_hist``.
    phi_T_target : np.ndarray
        Terminal target with shape ``(Nx+1, Ny+1)``.
    x, y : np.ndarray
        Spatial grids in the ``x`` and ``y`` directions.
    t_hist : np.ndarray
        Time grid of length ``M+1``.
    b1, b2, b3, kappa : float
        Weights in the cost functional.
    """
    # Term 1: Tracking cost (b1)
    error_sq = (phi_hist - phi_Q_target) ** 2
    integral_in_space_b1 = np.trapz(
        np.trapz(error_sq, x=y, axis=2), x=x, axis=1
    )
    cost1 = (b1 / 2.0) * np.trapz(integral_in_space_b1, x=t_hist)

    # Term 2: Terminal cost (b2)
    final_error_sq = (phi_hist[-1] - phi_T_target) ** 2
    cost2 = (b2 / 2.0) * np.trapz(
        np.trapz(final_error_sq, x=y, axis=1), x=x, axis=0
    )

    # Term 3: Control energy (b3)
    u_sq = u ** 2
    integral_in_space_b3 = np.trapz(
        np.trapz(u_sq, x=y, axis=2), x=x, axis=1
    )
    cost3 = (b3 / 2.0) * np.trapz(integral_in_space_b3, x=t_hist)

    # Term 4: Sparsity cost (kappa)
    u_abs = np.abs(u)
    integral_in_space_kappa = np.trapz(
        np.trapz(u_abs, x=y, axis=2), x=x, axis=1
    )
    cost4 = kappa * np.trapz(integral_in_space_kappa, x=t_hist)

    total_cost = cost1 + cost2 + cost3 + cost4

    print(f"  Tracking Cost (J1): {cost1:.6g}")
    print(f"  Terminal Cost (J2): {cost2:.6g}")
    print(f"  Control Energy (J3): {cost3:.6g}")
    print(f"  Sparsity Cost (J4): {cost4:.6g}")
    print(f"-----------------------------")
    print(f"  Total Cost: {total_cost:.6g}")

    return total_cost


def calculate_gradient(r: np.ndarray, u: np.ndarray, b3: float) -> np.ndarray:
    """Gradient of the smooth part of the cost functional.

    Parameters
    ----------
    r : np.ndarray
        Adjoint state ``r`` with the same shape as the control.
    u : np.ndarray
        Current control array.
    b3 : float
        Weight of the quadratic control term.
    """
    return r + b3 * u


def perform_gradient_step(
    u_current: np.ndarray, grad_smooth: np.ndarray, alpha: float
) -> np.ndarray:
    """Perform one gradient-descent step on the smooth cost part."""
    return u_current - alpha * grad_smooth
