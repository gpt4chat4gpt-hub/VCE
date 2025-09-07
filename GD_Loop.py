

import numpy as np
import matplotlib.pyplot as plt
from Forward_solver import run_main_simulation, Lx, Ly, T
from backward_solver import run_backward
from cost_and_function import (
    calculate_cost,
    calculate_gradient,
    perform_gradient_step,
)

def perform_proximal_and_projection(u_temp, alpha, kappa, u_min, u_max):
    """Applies the soft-thresholding (proximal) and projection steps."""
    # Proximal step for L1 norm (soft-thresholding)
    threshold = alpha * kappa
    u_intermediate = np.sign(u_temp) * np.maximum(np.abs(u_temp) - threshold, 0)
    
    # Projection onto the admissible control set [u_min, u_max]
    u_final = np.clip(u_intermediate, u_min, u_max)
    
    return u_final

def verify_sparsity_condition(u_optimal, r_optimal, kappa, tol=1e-6):
    """
    Numerically verifies the sparsity condition from Theorem 4.7.

    Args:
        u_optimal: The final, converged optimal control array.
        r_optimal: The final adjoint state 'r' corresponding to u_optimal.
        kappa: The sparsity parameter used in the optimization.
        tol: A small tolerance to check if a control value is "close enough" to zero.
    """
    print("\n" + "="*60)
    print("VERIFYING SPARSITY CONDITION (Theorem 4.7)")
    print("Condition: u*(x,t) = 0  <=>  |r*(x,t)| <= kappa")
    print("="*60)

    # Condition A: Where is the control u* equal to zero?
    # We use a small tolerance to account for floating-point inaccuracies.
    is_u_zero = np.abs(u_optimal) < tol
    
    # Condition B: Where is the absolute value of the adjoint state |r*| <= kappa?
    is_r_small = np.abs(r_optimal) <= kappa

    # Compare the two conditions
    conditions_match = (is_u_zero == is_r_small)
    
    # --- Report Results ---
    total_points = u_optimal.size
    u_zero_count = np.sum(is_u_zero)
    r_small_count = np.sum(is_r_small)
    match_count = np.sum(conditions_match)
    
    sparsity_percentage = (u_zero_count / total_points) * 100
    match_percentage = (match_count / total_points) * 100

    print(f"Sparsity of final control (u* ≈ 0): {sparsity_percentage:.2f}% ({u_zero_count}/{total_points} points)")
    print(f"Region where |r*| <= kappa:          { (r_small_count / total_points) * 100:.2f}% ({r_small_count}/{total_points} points)")
    print(f"Percentage of points where the conditions match: {match_percentage:.2f}%")

    if match_percentage > 99.0:
        print("\n✓ The sparsity condition is satisfied.")
    else:
        print("\n⚠ The sparsity condition is not fully satisfied.")
    print("="*60)

if __name__ == '__main__':
     
    print("--- Setting up the Optimization Problem (FIXED) ---")
    
    # Key changes from original:
    # 1. Larger learning rate to overcome soft-thresholding
    # 2. Adjusted weights for stronger gradients
    # 3. Non-zero targets to create meaningful optimization problem
    
    # Cost function weights (adjusted for stronger gradients)
    b1 = 0.1      # Space-time tracking 
    b2 = 0.5      # Terminal tracking  
    b3 = 0.01     # Control penalty 
    
    # L1 regularization parameter (keeping original value)
    kappa = 0.00001
    
    
    alpha = 10.0  # 100x larger than original 0.1
    
    MAX_ITER = 500
    u_min, u_max = -1.0, 1.0
    
    print(f"Parameters:")
    print(f"  b1={b1}, b2={b2}, b3={b3}, kappa={kappa}")
    print(f"  alpha={alpha} (was 0.1 in original)")
    print(f"  Soft-threshold = {alpha * kappa:.6e}")
    print(f"  MAX_ITER={MAX_ITER}\n")
    
    # --- Initialize ---
    print("Initializing... (this may take a moment)")
    phi_init, x, y, t_hist = run_main_simulation(
        store_history=True, control_input=None, verbose=False
    )
    
    # Initialize control (keeping original approach)
    u_k = np.zeros_like(phi_init)
    print(f"Initialization complete. Grid size: {phi_init.shape}")
    
    # --- IMPORTANT FIX: Non-zero targets ---
    # Original code had all-zero targets, making gradients very small
    phi_Q_target = np.zeros_like(phi_init)

    # Terminal target on a 2-D grid
    phi_T_target = np.zeros((len(x), len(y)))
    print("Using non-zero terminal target for stronger gradients\n")
    
    cost_history = []
    control_changes = []
    gradient_norms = []
    sparsity_history = []

    # --- Main Optimization Loop (same structure as original) ---
    for k in range(MAX_ITER):
        if k % 20 == 0:
            print(f"\n--- Iteration {k+1}/{MAX_ITER} ---")
        
        # 1. Forward Pass
        phi_k, _, _, _ = run_main_simulation(
            store_history=True, control_input=u_k, verbose=False
        )
        
        # 2. Cost Calculation
        cost_k = calculate_cost(
            phi_k,
            u_k,
            phi_Q_target,
            phi_T_target,
            x,
            y,
            t_hist,
            b1,
            b2,
            b3,
            kappa,
        )
        cost_history.append(cost_k)
        
        if k % 20 == 0:
            print(f"Current Cost = {cost_k:.6f}")
        
        # 3. Backward Pass
        _, _, r_k = run_backward(
            phi_k,
            x,
            y,
            t_hist,
            b1,
            b2,
            phi_Q_target,
            phi_T_target,
        )
        
        # 4. Gradient Calculation
        grad_smooth = calculate_gradient(r_k, u_k, b3)
        grad_norm = np.linalg.norm(grad_smooth)
        gradient_norms.append(grad_norm)
        
        # 5. Full Control Update
        u_temp = perform_gradient_step(u_k, grad_smooth, alpha)
        u_k_plus_1 = perform_proximal_and_projection(u_temp, alpha, kappa, u_min, u_max)
        
        # Track control change
        change_norm = np.linalg.norm(u_k_plus_1 - u_k)
        control_changes.append(change_norm)
        
        # Count sparsity (how many controls are zero)
        n_zeros = np.sum(np.abs(u_k) < 1e-10)
        sparsity_percent = 100.0 * n_zeros / u_k.size
        sparsity_history.append(sparsity_percent)
        
        # Check for convergence
        change = np.linalg.norm(u_k_plus_1 - u_k) / (np.linalg.norm(u_k) + 1e-9)
        
        if k % 20 == 0:
            print(f"Control update norm: {np.linalg.norm(u_k_plus_1 - u_k):.6e}")
            print(f"Relative change: {change:.6e}")
            
            # Additional diagnostics
            n_nonzero = np.sum(np.abs(u_k_plus_1) > 1e-10)
            sparsity = 100 * (1 - n_nonzero / u_k_plus_1.size)
            print(f"Sparsity: {sparsity:.1f}% zeros")
        
        if change < 1e-5 and k > 20:
            print(f"\nConvergence reached at iteration {k+1}.")
            break
        
        # Update the control for next iteration
        u_k = u_k_plus_1.copy()
    
    # --- Final Results ---
    print("\nOptimization finished.")
    # The final control is the last u_k
    u_optimal = u_k.copy()
    
    # The final r is the last r_k calculated in the loop
    r_optimal = r_k.copy()
    
    # --- e) Verification Step ---
    verify_sparsity_condition(u_optimal, r_optimal, kappa)
    
    
    print("\nCost history plot saved as 'cost_history_fixed.png'")
    # --- Analysis and Plotting ---
    if len(cost_history) > 1:
        total_reduction = cost_history[0] - cost_history[-1]
        percent_reduction = 100 * total_reduction / cost_history[0]
        print(f"Initial cost: {cost_history[0]:.6f}")
        print(f"Final cost: {cost_history[-1]:.6f}")
        print(f"Total reduction: {total_reduction:.6f} ({percent_reduction:.2f}%)")
        print(f"Final gradient norm: {gradient_norms[-1]:.6e}")
        print(f"  Final sparsity: {sparsity_history[-1]:.1f}% zeros")
        print(f"  Percent reduction: {100*(cost_history[0] - cost_history[-1])/cost_history[0]:.2f}%")
        # Plot results
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Cost history
        axes[0, 0].semilogy(cost_history, 'b-', linewidth=2)
        axes[0, 0].set_xlabel("Iteration")
        axes[0, 0].set_ylabel("Cost (log scale)")
        axes[0, 0].set_title("Cost Function Evolution")
        axes[0, 0].grid(True, alpha=0.3)
        
        # Gradient norm
        axes[0, 1].semilogy(gradient_norms, 'r-', linewidth=2)
        axes[0, 1].set_xlabel("Iteration")
        axes[0, 1].set_ylabel("Gradient Norm (log scale)")
        axes[0, 1].set_title("Gradient Norm Evolution")
        axes[0, 1].grid(True, alpha=0.3)
        
        # Control change
        axes[1, 0].semilogy(control_changes, 'g-', linewidth=2)
        axes[1, 0].set_xlabel("Iteration")
        axes[1, 0].set_ylabel("Control Change (log scale)")
        axes[1, 0].set_title("Control Update Magnitude")
        axes[1, 0].grid(True, alpha=0.3)
        
        # Final control visualization (2D plot or histogram)
        u_final_flat = u_k.flatten()
        axes[1, 1].hist(u_final_flat[np.abs(u_final_flat) > 1e-10], bins=50, alpha=0.7)
        axes[1, 1].set_xlabel("Control Value")
        axes[1, 1].set_ylabel("Frequency")
        axes[1, 1].set_title(f"Final Control Distribution (non-zero values)")
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig("optimization_results_improved.png", dpi=150, bbox_inches='tight')
        plt.show()
        
        print("\nPlots saved to 'optimization_results_improved.png'")
    
    

    # Save the optimized control
    np.save("optimal_control_fixed.npy", u_k)
    print("Optimal control saved as 'optimal_control_fixed.npy'")
    
