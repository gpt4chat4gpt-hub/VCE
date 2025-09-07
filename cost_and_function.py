import numpy as np
from scipy import sparse


def _trapz_weights(coords: np.ndarray) -> np.ndarray:
    """Return trapezoidal integration weights for a uniform grid."""
    h = coords[1] - coords[0]
    w = np.ones_like(coords)
    w[0] = w[-1] = 0.5
    return w * h


def _integrate_space(arr2d: np.ndarray, wx_sp: sparse.csr_matrix, wy_sp: sparse.csr_matrix) -> float:
    """Integrate a 2D array using sparse weight matrices."""
    arr_sp = sparse.csr_matrix(arr2d)
    return float(wx_sp.dot(arr_sp).dot(wy_sp)[0, 0])


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
    """Calculate the value of the discrete cost functional for a 2D domain."""

    wx = _trapz_weights(x)
    wy = _trapz_weights(y)
    wx_sp = sparse.csr_matrix(wx.reshape(1, -1))
    wy_sp = sparse.csr_matrix(wy.reshape(-1, 1))

    # Term 1: Tracking Cost (b1 term)
    error_sq = (phi_hist - phi_Q_target) ** 2
    integral_in_space_b1 = np.array(
        [_integrate_space(error_sq[n], wx_sp, wy_sp) for n in range(phi_hist.shape[0])]
    )
    cost1 = (b1 / 2.0) * np.trapezoid(integral_in_space_b1, x=t_hist)

    # Term 2: Terminal Cost (b2 term)
    final_error_sq = (phi_hist[-1] - phi_T_target) ** 2
    cost2 = (b2 / 2.0) * _integrate_space(final_error_sq, wx_sp, wy_sp)

    # Term 3: Control Energy Cost (b3 term)
    u_sq = u ** 2
    integral_in_space_b3 = np.array(
        [_integrate_space(u_sq[n], wx_sp, wy_sp) for n in range(u.shape[0])]
    )
    cost3 = (b3 / 2.0) * np.trapezoid(integral_in_space_b3, x=t_hist)

    # Term 4: Sparsity Cost (kappa term)
    u_abs = np.abs(u)
    integral_in_space_kappa = np.array(
        [_integrate_space(u_abs[n], wx_sp, wy_sp) for n in range(u.shape[0])]
    )
    cost4 = kappa * np.trapezoid(integral_in_space_kappa, x=t_hist)

    total_cost = cost1 + cost2 + cost3 + cost4

    print(f"  Tracking Cost (J1): {cost1:.6g}")
    print(f"  Terminal Cost (J2): {cost2:.6g}")
    print(f"  Control Energy (J3): {cost3:.6g}")
    print(f"  Sparsity Cost (J4): {cost4:.6g}")
    print(f"-----------------------------")
    print(f"  Total Cost: {total_cost:.6g}")

    return total_cost


def calculate_gradient(r: np.ndarray, u: np.ndarray, b3: float) -> np.ndarray:
    """Calculate the gradient of the smooth part of the cost functional."""
    shape = r.shape
    r_flat = sparse.csr_matrix(r.reshape(shape[0], -1))
    u_flat = sparse.csr_matrix(u.reshape(shape[0], -1))
    grad_flat = r_flat + b3 * u_flat
    return np.asarray(grad_flat).reshape(shape)


def perform_gradient_step(
    u_current: np.ndarray, grad_smooth: np.ndarray, alpha: float
) -> np.ndarray:
    """Perform one gradient descent step on the smooth part of the cost functional."""
    shape = u_current.shape
    u_flat = sparse.csr_matrix(u_current.reshape(shape[0], -1))
    grad_flat = sparse.csr_matrix(grad_smooth.reshape(shape[0], -1))
    u_temp_flat = u_flat - alpha * grad_flat
    return np.asarray(u_temp_flat).reshape(shape)


if __name__ == "__main__":
    print("--- Testing 2D cost and gradient functions ---")
    M, Nx, Ny = 5, 8, 6
    t_hist = np.linspace(0.0, 1.0, M)
    x = np.linspace(0.0, 1.0, Nx)
    y = np.linspace(0.0, 1.0, Ny)

    phi_hist = np.random.rand(M, Nx, Ny)
    u = np.random.rand(M, Nx, Ny)
    phi_Q_target = np.zeros_like(phi_hist)
    phi_T_target = np.zeros((Nx, Ny))

    b1, b2, b3, kappa = 0.01, 0.05, 0.1, 1e-6

    total_cost = calculate_cost(
        phi_hist, u, phi_Q_target, phi_T_target, x, y, t_hist, b1, b2, b3, kappa
    )
    print(f"Example total cost: {total_cost:.6f}")

    r_test = np.random.rand(M, Nx, Ny)
    grad = calculate_gradient(r_test, u, b3)
    u_updated = perform_gradient_step(u, grad, 0.1)
    print("Gradient and update steps computed successfully.")

