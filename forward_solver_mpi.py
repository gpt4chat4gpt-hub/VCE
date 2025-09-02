from mpi4py import MPI
import importlib.util
from pathlib import Path

# Load 2D_Forward_solver dynamically since the file name starts with a digit
spec = importlib.util.spec_from_file_location("two_d_forward_solver", Path(__file__).with_name("2D_Forward_solver.py"))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
run_forward_solver_mpi = module.run_forward_solver_mpi


def main():
    local_phi = run_forward_solver_mpi()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    if rank == 0:
        print("Distributed solver completed. Local result shape:", local_phi.shape)


if __name__ == "__main__":
    main()
