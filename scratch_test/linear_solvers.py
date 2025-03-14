"""A set of linear solvers for the linear system Ax = b."""

#3rd party
import jax.numpy as jnp
from petsc4py import PETSc

#native

#local apps




def direct_jax_solve(A: jnp.ndarray,b: jnp.ndarray) -> jnp.ndarray:
    """Solve the linear system Ax = b using a direct solver."""
    return jnp.linalg.solve(A, b)


def scipy_umfpack(A, b):
    """Solve the linear system Ax = b using the scipy scipy.sparse.linalg.spsolve with umfpack."""
    return jnp.linalg.solve(A, b)


def scipy_sparse_qr(A, b):
    """Solve the linear system Ax = b using the jax.experimental.scipy.sparse.linalg.spsolve."""
    return jnp.linalg.solve(A, b)

