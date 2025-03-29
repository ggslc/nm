"""A set of linear solvers for the linear system Ax = b."""

#3rd party
import jax.numpy as jnp
from petsc4py import PETSc

#native
import sys

#local apps
sys.path.insert(1, './')
from sparsity_utils import dodgy_coo_to_csr


def sparse_linear_solve(values, coordinates, jac_shape, b, x0, mode="jax-native"):

    match mode:
        case "jax_bicgstab":
            #https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_array.html
            A = BCOO((values, coordinates), shape=jac_shape)

            #If you don't include this preconditioner then things really go to shit
            diag_indices = jnp.where(coordinates[:, 0] == coordinates[:, 1])[0]
            jacobi_values = values[diag_indices]
            jacobi_indices = coordinates[diag_indices, :]
            M = BCOO((1.0 / jacobi_values, jacobi_indices), shape=jac_shape)
            preconditioner = lambda x: M @ x

            x, info = jax.scipy.sparse.linalg.bicgstab(A, b, x0=x0, M=preconditioner,
                                                      tol=1e-10, atol=1e-10,
                                                      maxiter=10000)

            # print(x)

            # Verify convergence
            residual = np.linalg.norm(b - A @ x)

        case "jax-native":
            iptr, j, values = dodgy_coo_to_csr(values, coordinates, jac_shape, return_decomposition=True)
            x = jax.experimental.sparse.linalg.spsolve(values, j, iptr, b)
            residual = None

        case "scipy-umfpack":
            csr_array = dodgy_coo_to_csr(values, coordinates, jac_shape)
            x = scipy.sparse.linalg.spsolve(csr_array, np.array(b))

            residual = np.linalg.norm(b - csr_array @ x)

    return x, residual


