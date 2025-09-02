from mpi4py import MPI
import numpy as np

try:
    from petsc4py import PETSc
except Exception:  # pragma: no cover - petsc4py may be unavailable during lint/tests
    PETSc = None


def partition_grid(Nx: int, Ny: int, comm: MPI.Comm):
    """Return start and end indices in the y-direction for the local rank."""
    size = comm.Get_size()
    rank = comm.Get_rank()

    counts = [(Ny + 1) // size] * size
    for i in range((Ny + 1) % size):
        counts[i] += 1
    starts = [sum(counts[:i]) for i in range(size)]
    start = starts[rank]
    end = start + counts[rank]
    return start, end


def exchange_halos(arr: np.ndarray, comm: MPI.Comm, rank: int, size: int):
    """Exchange halo rows of *arr* with vertical neighbours.

    The array *arr* is assumed to have shape ``(local_ny+2, Nx+1)`` with a
    single ghost row at the top and bottom. Data in the first and last interior
    rows are sent to neighbours and the received data populate the ghost rows.
    """
    up = rank - 1
    down = rank + 1

    if up >= 0:
        comm.Sendrecv(sendbuf=arr[1, :], dest=up, sendtag=11,
                      recvbuf=arr[0, :], source=up, recvtag=22)
    if down < size:
        comm.Sendrecv(sendbuf=arr[-2, :], dest=down, sendtag=22,
                      recvbuf=arr[-1, :], source=down, recvtag=11)


def _create_laplacian_matrix(Nx: int, Ny: int, comm: MPI.Comm):
    if PETSc is None:
        raise RuntimeError("petsc4py is required for distributed solve")

    rank = comm.Get_rank()
    start, end = partition_grid(Nx, Ny, comm)
    n_global = (Nx + 1) * (Ny + 1)

    A = PETSc.Mat().createAIJ([n_global, n_global], nnz=5, comm=comm)
    A.setFromOptions()
    A.setUp()

    for j in range(start, end):
        for i in range(Nx + 1):
            row = j * (Nx + 1) + i
            cols = [row]
            vals = [-4.0]
            if i > 0:
                cols.append(row - 1)
                vals.append(1.0)
            if i < Nx:
                cols.append(row + 1)
                vals.append(1.0)
            if j > 0:
                cols.append(row - (Nx + 1))
                vals.append(1.0)
            if j < Ny:
                cols.append(row + (Nx + 1))
                vals.append(1.0)
            A.setValues(row, cols, vals)

    A.assemblyBegin(); A.assemblyEnd()
    return A


def distributed_laplacian_solve(rhs_local: np.ndarray, Nx: int, Ny: int, comm: MPI.Comm) -> np.ndarray:
    """Solve a 2-D Poisson problem with zero Neumann boundaries.

    Parameters
    ----------
    rhs_local : np.ndarray
        Flattened local right-hand side for interior rows (without halo).
    Nx, Ny : int
        Number of intervals in x and y direction.
    comm : MPI.Comm
        Active communicator.
    """
    if PETSc is None:
        raise RuntimeError("petsc4py is required for distributed solve")

    size = comm.Get_size()
    rank = comm.Get_rank()
    start, end = partition_grid(Nx, Ny, comm)
    local_rows = (end - start) * (Nx + 1)
    n_global = (Ny + 1) * (Nx + 1)

    A = _create_laplacian_matrix(Nx, Ny, comm)

    # Build RHS vector
    b = PETSc.Vec().createMPI(n_global, comm=comm)
    for j in range(end - start):
        gstart = (start + j) * (Nx + 1)
        b.setValues(range(gstart, gstart + Nx + 1), rhs_local[j*(Nx+1):(j+1)*(Nx+1)])
    b.assemblyBegin(); b.assemblyEnd()

    x = PETSc.Vec().createMPI(n_global, comm=comm)
    ksp = PETSc.KSP().create(comm=comm)
    ksp.setOperators(A)
    ksp.setFromOptions()
    ksp.setType('cg')
    ksp.getPC().setType('jacobi')
    ksp.solve(b, x)

    # Extract local solution
    out = np.zeros((end - start, Nx + 1))
    for j in range(end - start):
        gstart = (start + j) * (Nx + 1)
        out[j, :] = x.getValues(range(gstart, gstart + Nx + 1))
    return out


def run_forward_solver_mpi(Nx: int = 32, Ny: int = 32, steps: int = 1) -> np.ndarray:
    """Simple driver performing ``steps`` implicit solves of a diffusion equation.

    The function demonstrates domain partitioning, halo exchange and a
    distributed PETSc solve. It returns the final local slice of ``phi``.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    start, end = partition_grid(Nx, Ny, comm)
    local_ny = end - start

    # local array with halo rows
    phi = np.zeros((local_ny + 2, Nx + 1))

    for _ in range(steps):
        exchange_halos(phi, comm, rank, size)
        rhs = phi[1:-1, :].reshape(-1)
        phi[1:-1, :] = distributed_laplacian_solve(rhs, Nx, Ny, comm).reshape(local_ny, Nx + 1)

    return phi[1:-1, :]
