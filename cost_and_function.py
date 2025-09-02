


# --- 1. Import your functions from their respective files ---
from Forward_solver import run_main_simulation

import numpy as np

def calculate_cost(
    phi_hist: np.ndarray,
    u: np.ndarray,
    phi_Q_target: np.ndarray,
    phi_T_target: np.ndarray,
    x: np.ndarray,
    t_hist: np.ndarray,
    b1: float,
    b2: float,
    b3: float,
    kappa: float
) -> float:
    """
    Calculates the value of the discrete cost functional.

    Args:
        phi_hist: State history from the forward solver, shape (M+1, N+1).
        u: Control function, same shape as phi_hist.
        phi_Q_target: Space-time target, same shape as phi_hist.
        phi_T_target: Final time target, shape (N+1,).
        x: Spatial grid.
        t_hist: Time grid.
        b1, b2, b3, kappa: Cost functional weights.

    Returns:
        The total cost (a single scalar value).
    """
    # Term 1: Tracking Cost (b1 term)
    error_sq = (phi_hist - phi_Q_target)**2
    # First, integrate over space for each time step
    integral_in_space_b1 = np.trapezoid(error_sq, x=x, axis=1)
    # Then, integrate the result over time
    cost1 = (b1 / 2.0) * np.trapezoid(integral_in_space_b1, x=t_hist)

    # Term 2: Terminal Cost (b2 term)
    final_error_sq = (phi_hist[-1] - phi_T_target)**2
    cost2 = (b2 / 2.0) * np.trapezoid(final_error_sq, x=x)

    # Term 3: Control Energy Cost (b3 term)
    u_sq = u**2
    integral_in_space_b3 = np.trapezoid(u_sq, x=x, axis=1)
    cost3 = (b3 / 2.0) * np.trapezoid(integral_in_space_b3, x=t_hist)

    # Term 4: Sparsity Cost (kappa term)
    u_abs = np.abs(u)
    integral_in_space_kappa = np.trapezoid(u_abs, x=x, axis=1)
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
    """
    Calculates the gradient of the smooth part of the cost functional.

    Args:
        r: The adjoint state variable 'r' from the backward solver.
        u: The current control function.
        b3: The weight of the control energy term in the cost functional.

    Returns:
        The gradient of the smooth part of the cost, with the same shape as r and u.
    """
    # The formula is derived from the first-order necessary optimality conditions
    grad_smooth = r + b3 * u
    return grad_smooth

# From update_step.py
def perform_gradient_step(
    u_current: np.ndarray, 
    grad_smooth: np.ndarray, 
    alpha: float
) -> np.ndarray:
    """
    Performs one gradient descent step on the smooth part of the cost functional.
    """
    u_temp = u_current - alpha * grad_smooth
    return u_temp

if __name__ == '__main__':
    print("--- Running Forward Solver and Cost Calculation In-Memory ---")

    # a) Run the forward solver to get the results directly into variables
    print("Step 1: Running the forward simulation...")
    phi_hist_from_ram, x_from_ram, t_hist_from_ram = run_main_simulation(store_history=True)
    print("Forward simulation complete. Data is now in memory.")

    # b) Define parameters and targets for the cost function
    print("Step 2: Setting up cost function parameters...")
    b1, b2, b3, kappa = 0.01, 0.05, 0.1, 0.000001
    
    # Define the control you want to score (e.g., u=0 for the baseline)
    u_to_score = np.zeros_like(phi_hist_from_ram)
    
    # Define your targets (shapes must match the simulation output)
    phi_Q_target = np.zeros_like(phi_hist_from_ram)
    phi_T_target = np.zeros_like(x_from_ram)

    # c) Directly pass the variables to the cost function
    print("Step 3: Calculating the score using in-memory data...")
    total_score = calculate_cost(
        phi_hist_from_ram, 
        u_to_score, 
        phi_Q_target, 
        phi_T_target,
        x_from_ram, 
        t_hist_from_ram, 
        b1, b2, b3, kappa
    )

    print("\n-------------------------------------------")
    print(f"The final calculated cost is: {total_score:.6f}")
    print("-------------------------------------------")
    print("Testing the gradient calculator function...")
    
    # Create some dummy data for testing
    shape = (101, 65)
    r_test = np.random.rand(*shape)
    u_test = 0 # Fixed: renamed from 'u' to 'u_test'
    b3_test = 0.1
    
    # Call the function
    gradient = calculate_gradient(r_test, u_test, b3_test)
    
    print(f"Input r shape: {r_test.shape}")
    print(f"Input u shape: {u_test.shape}")
    print(f"Output gradient shape: {gradient.shape}")
    print(f"A sample value from r: {r_test[50, 30]:.4f}")
    print(f"A sample value from u: {u_test[50, 30]:.4f}")
    print(f"The corresponding gradient value should be r + b3*u = {r_test[50, 30] + b3_test * u_test[50, 30]:.4f}")
    print(f"Actual calculated gradient value: {gradient[50, 30]:.4f}")
    
    # Check if the calculation is correct
    assert np.allclose(gradient, r_test + b3_test * u_test), "Test failed!"
    print("\nTest passed successfully!")
