"""
backward_solver_fixed.py — CORRECTED adjoint for viscous Cahn–Hilliard (1D, Neumann)

Fixed issues:
1. Proper backward time integration
2. Corrected signs in adjoint operators
3. Fixed terminal condition for r
4. Added stability checks
"""
import numpy as np
from typing import Tuple, Optional
import matplotlib.pyplot as plt
from working_model_optimized import (
    run_main_simulation,
    laplacian_matrix_neumann,
    c1, c2, tau, gamma,
)

def fpp_log(phi: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Second derivative of the logarithmic potential with safety clipping"""
    ph = np.clip(phi, -1 + eps, 1 - eps)
    return 2.0 * c1 / (1.0 - ph**2) - 2.0 * c2

def run_backward(
    b1: float = 0.01,
    b2: float = 0.05,
    phi_Q: Optional[np.ndarray] = None,
    phi_T_target: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Solve the adjoint system backward in time.
    
    The adjoint system evolves from t=T to t=0, solving for (p,q,r).
    """
    # Forward pass → φ history
    print("Running forward simulation...")
    phi_hist, x, t_hist = run_main_simulation(store_history=True)
    M_plus1, N_plus1 = phi_hist.shape
    print(f"Forward simulation complete: {M_plus1} time steps, {N_plus1} spatial points")
    
    # Targets
    if phi_Q is None:
        phi_Q = np.zeros_like(phi_hist)
    elif phi_Q.shape != phi_hist.shape:
        raise ValueError("phi_Q must have shape (M+1, N+1)")
    if phi_T_target is None:
        phi_T_target = np.zeros(N_plus1)
    elif phi_T_target.shape[0] != N_plus1:
        raise ValueError("phi_T_target must have length N+1")
    
    # Grid parameters
    h  = float(x[1] - x[0]) if N_plus1 > 1 else 1.0
    
    # Check for uniform time stepping
    dts = np.diff(t_hist)
    if not np.all(dts > 0):
        # Drop any duplicate/non-monotone times (e.g. your last two T’s are equal)
        keep = np.r_[True, dts > 0]
        t_hist = t_hist[keep]
        phi_hist = phi_hist[keep]
        dts = np.diff(t_hist)
    M_plus1 = phi_hist.shape[0]
    print(f"After cleanup: {M_plus1} time steps")
    print(f"Grid spacing h={h:.6f}, dt_n in [{dts.min():.6e}, {dts.max():.6e}]")
        
    # Operators
    L  = laplacian_matrix_neumann(N_plus1 - 1, h)
    L2 = L @ L
    I  = np.eye(N_plus1)
    
    # Allocate adjoints
    p = np.zeros_like(phi_hist)
    q = np.zeros_like(phi_hist)
    r = np.zeros_like(phi_hist)
    
    # FIXED: Terminal conditions at t=T (index -1)
    # (I - τ L) p^M = b2 (φ^M - φ_Ω)
    rhs_T = b2 * (phi_hist[-1] - phi_T_target)
    p[-1] = np.linalg.solve(I - tau*L, rhs_T)
    q[-1] = -L @ p[-1]
    r[-1] = np.zeros(N_plus1)  # FIXED: Should be array, not scalar
    
    print(f"Terminal conditions set: ||p(T)||={np.linalg.norm(p[-1]):.6f}")
    
    # CORRECTED: Adjoint operators with proper signs for backward evolution
    # These operators are the adjoints (transposes) of the forward operators
    def A_adjoint(phi_n: np.ndarray) -> np.ndarray:
        """Adjoint of forward operator at time n"""
        fpp_diag = np.diag(fpp_log(phi_n))
        # Adjoint involves transpose, but for symmetric operators L, L^T = L
        return I - tau*L + 0.5*dt_n*L2 - 0.5*dt_n*(fpp_diag @ L)
    
    def B_adjoint(phi_np1: np.ndarray) -> np.ndarray:
        """Adjoint of forward operator at time n+1"""
        fpp_diag = np.diag(fpp_log(phi_np1))
        # Note the sign change compared to forward operator
        return I - tau*L - 0.5*dt_n*L2 + 0.5*dt_n*(fpp_diag @ L)
    
    # Backward time march: from n=M-1 down to n=0
    print("Starting backward time integration...")
    for n in range(M_plus1 - 2, -1, -1):
        dt_n = t_hist[n+1] - t_hist[n]
        if dt_n <= 0:
            continue  
        # Source term from objective functional
        src = 0.5*dt_n*b1 * ((phi_hist[n] - phi_Q[n]) + (phi_hist[n+1] - phi_Q[n+1]))
        
        # Adjoint equation: A*_n p^n = B*_{n+1} p^{n+1} + source
        rhs = B_adjoint(phi_hist[n+1]) @ p[n+1] + src
        
        # Check for numerical issues
        if np.any(~np.isfinite(rhs)):
            print(f"Warning: Non-finite values in RHS at step n={n}")
            
        try:
            p[n] = np.linalg.solve(A_adjoint(phi_hist[n]), rhs)
        except np.linalg.LinAlgError:
            print(f"Warning: Singular matrix at step n={n}, using regularization")
            p[n] = np.linalg.solve(A_adjoint(phi_hist[n]) + 1e-10*I, rhs)
        
        # Update q and r
        q[n] = -L @ p[n]
        
        # FIXED: r equation for backward evolution
        # The sign and coefficient need to match the adjoint of the forward r-equation
        gamma_factor_back = (gamma - 0.5*dt_n)/(gamma + 0.5*dt_n)
        gamma_factor_source = (dt_n*0.5)/(gamma + 0.5*dt_n)
        r[n] = gamma_factor_back * r[n+1] + gamma_factor_source * (q[n] + q[n+1])
        
        # Progress indicator
        if n % 100 == 0:
            print(f"  Backward step n={n}: ||p||={np.linalg.norm(p[n]):.6f}, "
                  f"||q||={np.linalg.norm(q[n]):.6f}, ||r||={np.linalg.norm(r[n]):.6f}")
    
    print("Backward integration complete.")
    
    # Verify initial conditions (at t=0)
    print(f"Initial (t=0) norms: ||p(0)||={np.linalg.norm(p[0]):.6f}, "
          f"||q(0)||={np.linalg.norm(q[0]):.6f}, ||r(0)||={np.linalg.norm(r[0]):.6f}")
    
    return p, q, r, x, t_hist

def verify_adjoint_consistency(phi_hist, p, q, r, L, t_hist, b1, b2, phi_Q, phi_T_target):
    """
    Verify that the adjoint solution satisfies the adjoint equations.
    """
    M = phi_hist.shape[0] - 1
    N_plus1 = phi_hist.shape[1]
    I = np.eye(N_plus1)
    L2 = L @ L
    
    errors = {
        'terminal_p': 0,
        'terminal_q': 0, 
        'terminal_r': 0,
        'adjoint_eq_max': 0,
        'adjoint_eq_mean': 0,
        'constraint_q': 0,
        'r_ode': 0
    }
    
    # Check terminal conditions
    rhs_T = b2 * (phi_hist[-1] - phi_T_target)
    errors['terminal_p'] = np.linalg.norm((I - tau*L) @ p[-1] - rhs_T)
    errors['terminal_q'] = np.linalg.norm(q[-1] + L @ p[-1])
    errors['terminal_r'] = np.linalg.norm(r[-1])
    
    # Check adjoint equation at each time step
    adjoint_residuals = []
    for n in range(M):
        dt_n_actual = t_hist[n+1] - t_hist[n]
        if dt_n_actual <= 0:
            continue
        # Build operators
        fpp_n = np.diag(fpp_log(phi_hist[n]))
        fpp_np1 = np.diag(fpp_log(phi_hist[n+1]))
        
        A_adj = I - tau*L + 0.5*dt_n_actual*L2 - 0.5*dt_n_actual*(fpp_n @ L)
        B_adj = I - tau*L - 0.5*dt_n_actual*L2 + 0.5*dt_n_actual*(fpp_np1 @ L)
        
        src = 0.5*dt_n_actual*b1 * ((phi_hist[n] - phi_Q[n]) + (phi_hist[n+1] - phi_Q[n+1]))
        res = A_adj @ p[n] - B_adj @ p[n+1] - src
        adjoint_residuals.append(np.linalg.norm(res))
    
    errors['adjoint_eq_max'] = np.max(adjoint_residuals)
    errors['adjoint_eq_mean'] = np.mean(adjoint_residuals)
    
    # Check constraint q = -L p
    constraint_error = 0
    for n in range(M+1):
        constraint_error = max(constraint_error, np.linalg.norm(q[n] + L @ p[n]))
    errors['constraint_q'] = constraint_error
    
    # Check r ODE
    r_residuals = []
    for n in range(M):
        dt_n_actual = t_hist[n+1] - t_hist[n]
        if dt_n_actual <= 0:
            continue
        gamma_factor = (gamma + 0.5*dt_n_actual)
        res_r = - gamma * (r[n+1] - r[n])/dt_n_actual + 0.5*(r[n+1] + r[n]) - 0.5*(q[n+1] + q[n])
        r_residuals.append(np.linalg.norm(res_r))
    errors['r_ode'] = np.max(r_residuals) if r_residuals else 0
    
    return errors

if __name__ == '__main__':
    # Run adjoint with specified weights
    b1_used = 0.01
    b2_used = 0
    
    print("="*60)
    print("Running backward (adjoint) solver")
    print(f"Weights: b1={b1_used}, b2={b2_used}")
    print("="*60)
    
    p, q, r, x, t = run_backward(b1=b1_used, b2=b2_used)
    
    # Print summary statistics
    print("\n" + "="*60)
    print("ADJOINT SOLUTION SUMMARY")
    print("="*60)
    print(f"Adjoint shapes  p, q, r: {p.shape}, {q.shape}, {r.shape}")
    print(f"Grid/time sizes x, t   : {x.size}, {t.size}")
    print(f"Time range: [{t[0]:.4f}, {t[-1]:.4f}]")
    print(f"Spatial domain: [{x[0]:.4f}, {x[-1]:.4f}]")
    
    # Norms at initial and final times
    print("\nNorms at t=0 (initial for adjoint):")
    print(f"  ||p(0)||={np.linalg.norm(p[0]):.6e}, ||q(0)||={np.linalg.norm(q[0]):.6e}, ||r(0)||={np.linalg.norm(r[0]):.6e}")
    print("\nNorms at t=T (terminal for adjoint):")
    print(f"  ||p(T)||={np.linalg.norm(p[-1]):.6e}, ||q(T)||={np.linalg.norm(q[-1]):.6e}, ||r(T)||={np.linalg.norm(r[-1]):.6e}")
    
    # Save results
    np.savez("adjoint_fields_fixed.npz", p=p, q=q, r=r, x=x, t=t)
    print("\nSaved adjoint_fields_fixed.npz")
    
    # Verification
    print("\n" + "="*60)
    print("VERIFICATION OF ADJOINT EQUATIONS")
    print("="*60)
    
    # Re-run forward to get phi_hist for verification
    phi_hist, _, _ = run_main_simulation(store_history=True)
    h = float(x[1] - x[0])
    L = laplacian_matrix_neumann(x.size - 1, h)
    
    phi_Q = np.zeros_like(phi_hist)
    phi_T_target = np.zeros_like(x)
    
    errors = verify_adjoint_consistency(
        phi_hist, p, q, r, L, t, b1_used, b2_used, phi_Q, phi_T_target
    )
    
    print("Residual errors:")
    print(f"  Terminal condition (p): {errors['terminal_p']:.2e}")
    print(f"  Terminal condition (q): {errors['terminal_q']:.2e}")
    print(f"  Terminal condition (r): {errors['terminal_r']:.2e}")
    print(f"  Adjoint equation (max/mean): {errors['adjoint_eq_max']:.2e} / {errors['adjoint_eq_mean']:.2e}")
    print(f"  Constraint q = -Lp: {errors['constraint_q']:.2e}")
    print(f"  r-ODE residual: {errors['r_ode']:.2e}")
    
    # Check if errors are acceptably small
    tol = 1e-6
    all_good = all(v < tol for v in errors.values())
    if all_good:
        print("\n✓ All verification checks PASSED (errors < 1e-6)")
    else:
        print("\n⚠ Some verification checks have larger errors")
        
    print("="*60)
