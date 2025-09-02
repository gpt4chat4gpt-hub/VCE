from mpi4py import MPI
import importlib

# 2D_Forward_solver has a filename beginning with a digit, so load it dynamically
solver = importlib.import_module("2D_Forward_solver")

def main():
    """Run multiple independent phase-separation simulations across MPI ranks.

    Each rank uses a unique random seed and writes its own GIF to avoid file
    conflicts. Results are named ``phase_separation_rankX.gif`` where ``X`` is
    the MPI rank.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        print(f"Launching {size} simulations across MPI ranks")

    filename = f"phase_separation_rank{rank}.gif"
    seed = 42 + rank
    solver.save_phase_separation_animation(filename=filename, seed=seed)

    if rank == 0:
        print("All simulations completed.")

if __name__ == "__main__":
    main()
