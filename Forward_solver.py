import numpy as np
import matplotlib.pyplot as plt

# === PARAMETERS ===
# Grid resolution in each spatial direction
Nx = 64                  # Number of intervals along x  (=> Nx+1 nodes)
Ny = 64                  # Number of intervals along y  (=> Ny+1 nodes)

# Domain lengths (square domain by default)
Lx = 1.0
Ly = 1.0

# Spatial step sizes (assume uniform grid)
hx = Lx / Nx
hy = Ly / Ny

dt_initial = 1e-4        # Initial time step size
T = 1.0                  # Total simulation time
tau = 0.01               # Relaxation parameter for phi-equation keep it .01
gamma = 10.0             # Relaxation parameter for w-equation
c1 = 0.5                 # Flory–Huggins convex coefficient
c2 = 1.0                 # Concave (quadratic) coefficient  as 1
kappa = 0.03**2          # gradient-energy coefficient ~ (interface thickness)^2

# Derived parameters (optional)
theta   = 2 * c1
theta_c = 2 * c2

# Safeguards / tolerances
delta_sep = 1e-2         # keep |phi| <= 1 - delta_sep
dt = dt_initial
DEBUG = True
COMPUTE_ENERGY = False  # Only compute energy when needed

# === CORE FUNCTIONS ===
def instability_report(c1, c2, kappa, tau, Lx, Nmodes=12):
    import numpy as np
    a = 2*(c1 - c2)  # curvature at φ≈0 for your f'
    ks = np.pi * np.arange(1, Nmodes+1) / Lx
    q  = ks**2
    lam = (-kappa*q**2 - a*q) / (1 + tau*q)   # growth rates λ(k)
    print(f"a={a:.3g},  max λ={lam.max():.3g} at mode n={lam.argmax()+1},  unstable modes={(lam>0).sum()}")
    return lam

def regularized_log(phi, eps=None):
    
    if eps is None:
        eps = max(1e-8, 0.5*delta_sep)
    phi_s = np.clip(phi, -1+eps, 1-eps)
    return np.log((1 + phi_s)/(1 - phi_s))

def laplacian_matrix_neumann_1d(N, h):
    """1-D Laplacian with Neumann boundary conditions."""
    a = 1.0 / (h*h)
    L = np.zeros((N+1, N+1))
    diag_indices = np.arange(1, N)
    L[diag_indices, diag_indices-1] = a
    L[diag_indices, diag_indices] = -2*a
    L[diag_indices, diag_indices+1] = a
    L[0, 0], L[0, 1]   = -2*a,  2*a
    L[N, N-1], L[N, N] =  2*a, -2*a
    return L

def laplacian_matrix_neumann(Nx, Ny, hx, hy):
    """2-D Laplacian with Neumann boundary conditions using Kronecker products."""
    Lx_mat = laplacian_matrix_neumann_1d(Nx, hx)
    Ly_mat = laplacian_matrix_neumann_1d(Ny, hy)
    Ix = np.eye(Nx + 1)
    Iy = np.eye(Ny + 1)
    return np.kron(Iy, Lx_mat) + np.kron(Ly_mat, Ix)

def apply_laplacian(L, v):
   
    return L @ v

def initialize_mu(phi, w, c1, c2, L):
  
    lap = apply_laplacian(L, phi)
    f_prime = c1 * regularized_log(phi) - 2.0*c2*phi
    return -kappa * lap + f_prime - w

def solve_w(w_old, dt, gamma, u_n, u_np1):
 
    gamma_dt = gamma/dt
    return ((gamma_dt - 0.5)*w_old + 0.5*(u_np1 + u_n)) / (gamma_dt + 0.5)

def solve_mu_residual(phi_new, phi_old, mu_new, mu_old, dt, L):

    lap_new = apply_laplacian(L, mu_new)
    lap_old = apply_laplacian(L, mu_old)
    return (phi_new - phi_old) / dt - 0.5*(lap_new + lap_old)

def solve_phi_residual(phi_new, phi_old, mu_new, mu_old, w_new, w_old, dt, tau, c1, c2, L):
 
    lap_new = apply_laplacian(L, phi_new)
    lap_old = apply_laplacian(L, phi_old)
    
    f_cvx = c1 * regularized_log(phi_new)   # implicit convex
    f_ccv = -2.0 * c2 * phi_old            # explicit concave
    mu_avg = 0.5*(mu_new + mu_old)
    w_avg  = 0.5*(w_new  + w_old)

    return (tau*(phi_new - phi_old)/dt) - 0.5 * kappa * (lap_new + lap_old) + (f_cvx + f_ccv) - mu_avg - w_avg

def assemble_jacobian(phi_new, dt, tau, c1, L):
    """Assemble the 2x2 block Jacobian for the Newton step."""
    n = phi_new.size
    size = 2 * n
    J = np.zeros((size, size))
    t = tau / dt
    s = 1.0 / dt

    # K_phi_phi
    Kpp = -0.5 * kappa * L.copy()
    phi_sq = phi_new**2
    Diag = 2.0 * c1 / (1.0 - phi_sq)  # safe since line search enforces |phi|<1
    np.fill_diagonal(Kpp, np.diag(Kpp) + t + Diag)

    # K_phi_mu, K_mu_phi, K_mu_mu
    I = np.eye(n)
    Kpm = -0.5 * I
    Kmp = s * I
    Kmm = -0.5 * L

    # Pack using slicing
    J[:n, :n] = Kpp
    J[:n, n:] = Kpm
    J[n:, :n] = Kmp
    J[n:, n:] = Kmm
    return J

def newton_raphson(phi_old, mu_old, w_old, w_new, dt, tau, c1, c2, delta_sep, L):
 
    phi_new = phi_old.copy()
    mu_new = mu_old.copy()
    tol = 1e-6
    max_iter = 50
    n = phi_old.size

    for k in range(max_iter):
        # Residuals (compute once per iteration)
        res_phi = solve_phi_residual(phi_new, phi_old, mu_new, mu_old, w_new, w_old, dt, tau, c1, c2, L)
        res_mu  = solve_mu_residual(phi_new, phi_old, mu_new, mu_old, dt, L)
        R = np.concatenate([res_phi, res_mu])
        norm_R = np.linalg.norm(R)

        if DEBUG and k % 10 == 0:
            mass_defect = np.sum(res_mu)
            if not np.isfinite(mass_defect):
                raise RuntimeError("Non-finite mass_defect; check φ bounds/log regularization.")
            if abs(mass_defect) > 1e-12:
                print(f"[warn] mass defect = {mass_defect:.3e}")

        if norm_R < tol:
            return phi_new, mu_new  # converged

        # Analytic Jacobian
        J = assemble_jacobian(phi_new, dt, tau, c1, L)

        # Solve for Newton step with regularization if needed
        try:
            delta = np.linalg.solve(J, -R)
        except np.linalg.LinAlgError:
            delta = np.linalg.solve(J + 1e-10*np.eye(J.shape[0]), -R)

        dphi = delta[:n]
        dmu = delta[n:]

        # Step ceiling to keep φ in (-1+delta_sep, 1-delta_sep)
        # Vectorized computation
        with np.errstate(divide='ignore', invalid='ignore'):
            pos_mask = dphi > 0
            neg_mask = dphi < 0
            
            if np.any(pos_mask):
                alpha_pos = np.min((1 - delta_sep - phi_new[pos_mask]) / dphi[pos_mask])
            else:
                alpha_pos = np.inf
                
            if np.any(neg_mask):
                alpha_neg = np.min((-1 + delta_sep - phi_new[neg_mask]) / dphi[neg_mask])
            else:
                alpha_neg = np.inf
                
            alpha_max = min(alpha_pos, alpha_neg)
            
        if not np.isfinite(alpha_max) or alpha_max <= 0:
            alpha_max = 1.0
        alpha = min(1.0, 0.9 * alpha_max)

        # Armijo backtracking
        eta = 1e-3
        for _ in range(12):
            phi_t = phi_new + alpha * dphi
            mu_t  = mu_new  + alpha * dmu
            if np.all(np.abs(phi_t) < 1 - delta_sep):
                Rphi_t = solve_phi_residual(phi_t, phi_old, mu_t, mu_old, w_new, w_old, dt, tau, c1, c2, L)
                Rmu_t  = solve_mu_residual(phi_t, phi_old, mu_t, mu_old, dt, L)
                Rt = np.concatenate([Rphi_t, Rmu_t])
                if np.linalg.norm(Rt) <= (1 - eta * alpha) * norm_R:
                    phi_new, mu_new = phi_t, mu_t
                    break
            alpha *= 0.5
        else:
            # line search failed: return last iterate (partially improved state)
            return phi_new, mu_new

    # Not converged within max_iter: return last iterate
    return phi_new, mu_new

def free_energy(phi, kappa, c1, c2, hx, hy, w=None, eps=None):
    """Compute a simple discrete free energy on a 2-D grid."""
    if eps is None:
        eps = 1e-8

    # Gradient part using forward differences
    dphix = np.diff(phi, axis=0)
    dphiy = np.diff(phi, axis=1)
    E_grad = (kappa/2.0) * (
        np.sum(dphix**2) / hx + np.sum(dphiy**2) / hy
    ) * hx * hy

    # Bulk part
    phi_s = np.clip(phi, -1 + eps, 1 - eps)
    psi = c1*((1+phi_s)*np.log(1+phi_s) + (1-phi_s)*np.log(1-phi_s)) - c2*(phi_s**2)
    E_bulk = hx * hy * np.sum(psi)

    E = E_grad + E_bulk

    if w is not None:
        E -= hx * hy * np.sum(w * phi)
    return E

def init_phi_random(Nx, Ny, delta_sep, amp=0.01, seed=42, enforce_zero_mean=True):
    rng = np.random.default_rng(seed)
    phi0 = amp * rng.standard_normal((Nx + 1, Ny + 1))
    if enforce_zero_mean:
        phi0 -= np.mean(phi0)
    phi0 = np.clip(phi0, -1 + delta_sep, 1 - delta_sep)
    return phi0


def source_u(t, x, y):
    return np.zeros((Nx + 1, Ny + 1))

def run_main_simulation(store_history=False, control_input=None, verbose=True):
    x = np.linspace(0, Lx, Nx + 1)
    y = np.linspace(0, Ly, Ny + 1)

    # initial phi inside (-1,1)
    phi = init_phi_random(Nx, Ny, delta_sep, amp=0.01, seed=42, enforce_zero_mean=True).reshape(-1)

    w = np.zeros_like(phi)

    # precompute Neumann Laplacian ONCE
    Lmat = laplacian_matrix_neumann(Nx, Ny, hx, hy)

    mu = initialize_mu(phi, w, c1, c2, Lmat)  # Use precomputed L

    current_time = 0.0
    step = 0

    # Pre-allocate history arrays if requested
    if store_history:
        M_steps = int(np.ceil(T / dt))
        phi_hist = np.empty((M_steps + 1, Nx + 1, Ny + 1), dtype=np.float64)
        t_hist = np.empty(M_steps + 1, dtype=np.float64)
        phi_hist[0] = phi.reshape(Nx + 1, Ny + 1)
        t_hist[0] = current_time
        n_store = 0

    zero_source = np.zeros_like(phi)
    time_tol = 1e-10

    while current_time < T - time_tol:
        dt_step = min(dt, T - current_time)
        if current_time + dt_step > T:
            dt_step = T - current_time

        if control_input is not None:
            if step < control_input.shape[0] - 1:
                u_n = control_input[step].reshape(-1)
                u_np1 = control_input[step + 1].reshape(-1)
            else:
                u_n = control_input[step].reshape(-1)
                u_np1 = control_input[step].reshape(-1)
        else:
            u_n = zero_source
            u_np1 = zero_source
        w_new = solve_w(w, dt_step, gamma, u_n, u_np1)

        if COMPUTE_ENERGY and (step % 100 == 0):
            E_prev = free_energy(phi.reshape(Nx + 1, Ny + 1), kappa, c1, c2, hx, hy, w=None)

        phi_new, mu_new = newton_raphson(phi, mu, w, w_new, dt_step, tau, c1, c2, delta_sep, Lmat)

        if COMPUTE_ENERGY and (step % 100 == 0):
            E_now = free_energy(phi_new.reshape(Nx + 1, Ny + 1), kappa, c1, c2, hx, hy, w=None)
            print(f"ΔE = {E_now - E_prev:.3e}")

        phi = np.clip(phi_new, -1 + delta_sep, 1 - delta_sep)
        mu = mu_new
        w = w_new

        current_time += dt_step
        step += 1

        if store_history:
            n_store += 1
            if n_store < phi_hist.shape[0]:
                phi_hist[n_store] = phi.reshape(Nx + 1, Ny + 1)
                t_hist[n_store] = min(current_time, T)
            else:
                phi_hist = np.vstack([phi_hist, phi.reshape(1, Nx + 1, Ny + 1)])
                t_hist = np.append(t_hist, min(current_time, T))
                n_store += 1

        if verbose and (step % 100 == 0 or current_time >= T):
            print(f"Step {step:5d} | t={current_time:.4e} | ||phi||_inf={np.max(np.abs(phi)):.5f}")

        if DEBUG and step % 500 == 0:
            mass = np.sum(phi)
            if not np.isfinite(mass):
                print("[warn] non-finite mass at step", step)

    if verbose:
        print("Simulation complete.")

    if not store_history:
        plt.figure(figsize=(6, 5))
        plt.imshow(phi.reshape(Nx + 1, Ny + 1), origin='lower', extent=(0, Lx, 0, Ly))
        plt.title(f"Final state at t={T}")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.colorbar(label="φ(x,y,T)")
        plt.show()

    if store_history:
        return phi_hist[:n_store + 1].copy(), x, y, t_hist[:n_store + 1].copy()
    else:
        return None

if __name__ == '__main__':
    # When executed directly, run the simulation without storing the history
    run_main_simulation(store_history=False)
