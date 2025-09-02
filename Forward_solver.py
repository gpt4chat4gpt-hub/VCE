import numpy as np
import matplotlib.pyplot as plt

# === PARAMETERS ===
N = 64                 # Number of intervals (=> N+1 nodes)
Lx = 1.0                 # Domain length
h  = Lx / N              # Spatial step size (N intervals)
dt_initial = 1e-4      # Initial time step size
T = 1.0                  # Total simulation time
tau = 0.01            # Relaxation parameter for phi-equation keep it .01
gamma = 10.0             # Relaxation parameter for w-equation
c1 = 0.5                 # Flory–Huggins convex coefficient
c2 = 1.0            # Concave (quadratic) coefficient  as 1 
kappa = 0.03**2  # gradient-energy coefficient ~ (interface thickness)^2

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

def laplacian_matrix_neumann(N, h):
   
    a = 1.0 / (h*h)
    L = np.zeros((N+1, N+1))
    # Build using vectorization
    diag_indices = np.arange(1, N)
    L[diag_indices, diag_indices-1] = a
    L[diag_indices, diag_indices] = -2*a
    L[diag_indices, diag_indices+1] = a
    # boundaries: (Lv)_0 = 2/h^2 (v_1 - v_0), (Lv)_N = 2/h^2 (v_{N-1} - v_N)
    L[0, 0], L[0, 1]   = -2*a,  2*a
    L[N, N-1], L[N, N] =  2*a, -2*a
    return L

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
  
    Nloc = len(phi_new) - 1
    size = 2*(Nloc+1)
    J = np.zeros((size, size))
    t = tau/dt
    s = 1.0/dt

    # K_phi_phi
    Kpp = -0.5 * kappa * L.copy()
    # Vectorized diagonal computation
    phi_sq = phi_new**2
    Diag = 2.0*c1 / (1.0 - phi_sq)      # safe since line search enforces |phi|<1
    np.fill_diagonal(Kpp, np.diag(Kpp) + t + Diag)

    # K_phi_mu, K_mu_phi, K_mu_mu
    I = np.eye(Nloc+1)
    Kpm = -0.5 * I
    Kmp =  s   * I
    Kmm = -0.5 * L

    # pack using slicing
    J[:Nloc+1, :Nloc+1]       = Kpp
    J[:Nloc+1, Nloc+1:]       = Kpm
    J[Nloc+1:, :Nloc+1]       = Kmp
    J[Nloc+1:, Nloc+1:]       = Kmm
    return J

def newton_raphson(phi_old, mu_old, w_old, w_new, dt, tau, c1, c2, delta_sep, L):
 
    phi_new = phi_old.copy()
    mu_new  = mu_old.copy()
    tol = 1e-6
    max_iter = 50
    Nloc = len(phi_old) - 1

    # trapezoidal weights for conservation check (w^T L = 0)
    wts = np.ones(Nloc + 1)
    wts[0]  = 0.5
    wts[-1] = 0.5
    wts_h = h * wts  # Precompute weighted factor

    for k in range(max_iter):
        # Residuals (compute once per iteration)
        res_phi = solve_phi_residual(phi_new, phi_old, mu_new, mu_old, w_new, w_old, dt, tau, c1, c2, L)
        res_mu  = solve_mu_residual(phi_new, phi_old, mu_new, mu_old, dt, L)
        R = np.concatenate([res_phi, res_mu])
        norm_R = np.linalg.norm(R)

        if DEBUG and k % 10 == 0:  # Reduce frequency of debug checks
            # weighted mass balance (discrete integral with trapz weights)
            mass_defect = np.dot(wts_h, res_mu)
            if not np.isfinite(mass_defect):
              raise RuntimeError("Non-finite mass_defect; check φ bounds/log regularization.")
            if abs(mass_defect) > 1e-12:
                print(f"[warn] weighted mass defect = {mass_defect:.3e}")

        if norm_R < tol:
            return phi_new, mu_new  # converged

        # Analytic Jacobian
        J = assemble_jacobian(phi_new, dt, tau, c1, L)

        # Solve for Newton step with regularization if needed
        try:
            delta = np.linalg.solve(J, -R)
        except np.linalg.LinAlgError:
            delta = np.linalg.solve(J + 1e-10*np.eye(J.shape[0]), -R)

        dphi = delta[:Nloc+1]
        dmu  = delta[Nloc+1:]

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

def trapz_weights(n_nodes: int) -> np.ndarray:
    w = np.ones(n_nodes)
    w[0] = 0.5
    w[-1] = 0.5
    return w

def free_energy(phi, kappa, c1, c2, h, w=None, eps=None):
    
    wts = trapz_weights(len(phi))
    # gradient part: ∫ (kappa/2) |phi_x|^2 dx  ≈  (kappa/(2h)) Σ (Δphi)^2
    dphi = np.diff(phi)  # More efficient than phi[1:] - phi[:-1]
    E_grad = (kappa/(2.0*h)) * np.sum(dphi**2)

    # bulk part: ψ(φ) with safe logs
    if eps is None: 
        eps = 1e-8
    phi_s = np.clip(phi, -1+eps, 1-eps)
    psi = c1*((1+phi_s)*np.log(1+phi_s) + (1-phi_s)*np.log(1-phi_s)) - c2*(phi_s**2)
    E_bulk = h * np.dot(wts, psi)

    E = E_grad + E_bulk

    # optional external coupling: -∫ w φ dx
    if w is not None:
        E -= h * np.dot(wts, w*phi)
    return E

def init_phi_random(N, delta_sep, amp=0.01, seed=42, enforce_zero_mean=True):
   
    rng = np.random.default_rng(seed)
    phi0 = amp * rng.standard_normal(N + 1)

    # optional: enforce trapezoidal zero mean (mass-neutral start)
    if enforce_zero_mean:
        wts = trapz_weights(N + 1)
        m = np.dot(wts, phi0) / wts.sum()
        phi0 -= m

    # safety: stay within the log domain
    phi0 = np.clip(phi0, -1 + delta_sep, 1 - delta_sep)
    return phi0


def source_u(t, x):
   
    return np.zeros_like(x)

def run_main_simulation(store_history = False, control_input=None, verbose=True):
  
    
    x = np.linspace(0, Lx, N + 1)
    # initial phi inside (-1,1)
    phi = init_phi_random(N, delta_sep, amp=0.01, seed=42, enforce_zero_mean=True)

    w   = np.zeros(N + 1)
    
    # precompute Neumann Laplacian ONCE
    Lmat = laplacian_matrix_neumann(N, h)
    
    mu  = initialize_mu(phi, w, c1, c2, Lmat)     # Use precomputed L

    current_time = 0.0
    step = 0

    # Pre-allocate history arrays if requested
    if store_history:
        # compute the expected number of time levels. We take ceiling to be safe
        M_steps = int(np.ceil(T / dt))
        phi_hist = np.empty((M_steps + 1, N + 1), dtype=np.float64)
        t_hist = np.empty(M_steps + 1, dtype=np.float64)
        # record initial state
        phi_hist[0] = phi.copy()
        t_hist[0] = current_time
        # index to store next state
        n_store = 0
    
    # Precompute source function if it's constant
    zero_source = np.zeros(N + 1)
    
    # Use tolerance to avoid floating-point issues at final time
    time_tol = 1e-10
    
    while current_time < T - time_tol:
        dt_step = min(dt, T - current_time)
        
        # Ensure we don't overshoot due to floating-point arithmetic
        if current_time + dt_step > T:
            dt_step = T - current_time

    
        # w^{n+1}
        if control_input is not None:
            # Use the provided control history at the current and next time step
            # control_input shape is (M+1, N+1), extract spatial values at time step
            if step < control_input.shape[0] - 1:
                u_n = control_input[step, :]
                u_np1 = control_input[step + 1, :]
            else:
                # At the last step, use the last control value for both
                u_n = control_input[step, :]
                u_np1 = control_input[step, :]
        else:
            # If no control is provided, fall back to zero
            u_n = zero_source
            u_np1 = zero_source
        w_new = solve_w(w, dt_step, gamma, u_n, u_np1)
                
        # Only compute energy when needed
        if COMPUTE_ENERGY and (step % 100 == 0):
            E_prev = free_energy(phi, kappa, c1, c2, h, w=None)
        
        # Newton solve for (phi^{n+1}, mu^{n+1})
        phi_new, mu_new = newton_raphson(phi, mu, w, w_new, dt_step, tau, c1, c2, delta_sep, Lmat)
        
        if COMPUTE_ENERGY and (step % 100 == 0):
            E_now  = free_energy(phi_new, kappa, c1, c2, h, w=None)
            print(f"ΔE = {E_now - E_prev:.3e}")   # should be <= 0 (up to tiny roundoff)
        
        # accept & clamp very lightly for safety
        phi = np.clip(phi_new, -1+delta_sep, 1-delta_sep)
        mu  = mu_new
        w   = w_new

        # update bookkeeping
        current_time += dt_step
        step += 1

        # store the updated state if requested
        if store_history:
            n_store += 1
            # ensure we don't exceed preallocated array
            if n_store < phi_hist.shape[0]:
                phi_hist[n_store] = phi
                # Ensure final time is exactly T, not T + small epsilon
                t_hist[n_store] = min(current_time, T)
            else:
               
                phi_hist = np.vstack([phi_hist, phi[None, :]])
                t_hist = np.append(t_hist, min(current_time, T))
                n_store += 1

        if verbose and (step % 100 == 0 or current_time >= T):
            print(f"Step {step:5d} | t={current_time:.4e} | ||phi||_inf={np.max(np.abs(phi)):.5f}")

        # optional per-step checks (only when debugging)
        if DEBUG and step % 500 == 0:  # Reduce frequency
            mass = np.sum(phi)  # Simple mass check
            if not np.isfinite(mass):
                print("[warn] non-finite mass at step", step)

    if verbose:
        print("Simulation complete.")

    # plot final state (only when not storing history)
    if not store_history:
        plt.figure(figsize=(10, 6))
        plt.plot(x, phi, label=f'Final state at t={T}')
        plt.title("Final Profile of φ")
        plt.xlabel("x")
        plt.ylabel("φ(x, T)")
        plt.grid(True)
        plt.legend()
        plt.show()

    if store_history:
        return phi_hist[:n_store+1].copy(), x, t_hist[:n_store+1].copy()
    else:
        return None

if __name__ == '__main__':
    # When executed directly, run the simulation without storing the history
    run_main_simulation(store_history=False)
