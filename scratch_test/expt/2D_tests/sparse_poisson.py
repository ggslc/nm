#1st party
from pathlib import Path
import sys


#local apps
sys.path.insert(1, "../../../utils/")
from sparsity_utils import scipy_coo_to_csr,\
                           basis_vectors_and_coords_2d_square_stencil,\
                           make_sparse_jacrev_fct_new



#3rd party
from petsc4py import PETSc
#from mpi4py import MPI

import jax
import jax.numpy as jnp
from jax import jacfwd, jacrev
import matplotlib.pyplot as plt
import jax.scipy.linalg as lalg
from jax.scipy.optimize import minimize
from jax import custom_vjp
from jax.experimental.sparse import BCOO

import numpy as np
import scipy

from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm

np.set_printoptions(precision=1, suppress=False, linewidth=np.inf, threshold=np.inf)




def make_residual_function(C, f):
    ny, nx = C.shape

    def residual_function(u_1d):

        u_2d = u_1d.reshape((ny, nx))

        #dudx = jnp.zeros((ny, nx+1))
        #dudy = jnp.zeros((ny+1,nx))

        #dudx = dudx.at[:, 1:-1].set((u[:,1:]-u[:,:-1])/dx)
        #dudx = dudx.at[:, 0].set((2*u[:,0])/dx)
        #dudx = dudx.at[:, -1].set((2*u[:,-1])/dx)

        #dudy = dudy.at[:, 1:-1].set((u[:,1:]-u[:,:-1])/dy)
        #dudy = dudy.at[:, 0].set((2*u[:,0])/dy)
        #dudy = dudy.at[:, -1].set((2*u[:,-1])/dy)


        x_flux_diffs = jnp.zeros((ny, nx))
        y_flux_diffs = jnp.zeros((ny, nx))

        x_flux_diffs = x_flux_diffs.at[:, 1:-1].set((dy/dx) * (u_2d[:,2:] + u_2d[:,:-2] - 2*u_2d[:,1:-1]))
        x_flux_diffs = x_flux_diffs.at[:, 0].set((dy/dx) * (-2*u_2d[:, 0]))
        x_flux_diffs = x_flux_diffs.at[:,-1].set((dy/dx) * (-2*u_2d[:,-1]))

        y_flux_diffs = y_flux_diffs.at[1:-1, :].set((dx/dy) * (u_2d[2:,:] + u_2d[:-2,:] - 2*u_2d[1:-1,:]))
        y_flux_diffs = y_flux_diffs.at[0, :].set((dx/dy) * (-2*u_2d[0, :]))
        y_flux_diffs = y_flux_diffs.at[-1,:].set((dx/dy) * (-2*u_2d[-1,:]))


        volume_term = (C*u_2d + f)*dx*dy

        return (x_flux_diffs + y_flux_diffs - volume_term).reshape(-1)

    return residual_function


def make_newton_solver(C, f, n_iterations):

    residual_func = make_residual_function(C, f)

    basis_vectors, i_coordinate_sets\
            = basis_vectors_and_coords_2d_square_stencil(nr, nc, 1)

    i_coordinate_sets = jnp.concatenate(i_coordinate_sets)
    j_coordinate_sets = jnp.tile(jnp.arange(nr*nc), len(basis_vectors))
    mask = ~jnp.isnan(i_coordinate_sets)

    sparse_jacrev_func, densify_func = make_sparse_jacrev_fct_new(basis_vectors,\
                                             i_coordinate_sets,\
                                             j_coordinate_sets,\
                                             mask)


    def solver(u_trial):
        
        u = u_trial.copy()

        #interesting to see that it doesn't go that well if I do only one iteration
        #even though the problem is linear... wtf.
        for i in range(n_iterations):
            print(jnp.max(jnp.abs(residual_func(u))))

            residual_jac_sparse = sparse_jacrev_func(residual_func, (u,))

            residual_jac_dense = densify_func(residual_jac_sparse, nr*nc)

            rhs = -residual_func(u)

            #print(residual_jac_dense)
            #print(rhs)
            #raise

            u += lalg.solve(residual_jac_dense, rhs)

        print(jnp.max(jnp.abs(residual_func(u))))

        return u

    return solver


def solve_petsc_sparse(values, coordinates, jac_shape, b, ksp_type='gmres', preconditioner='hypre', precondition_only=False):
    comm = PETSc.COMM_WORLD
    size = comm.Get_size()

    iptr, j, values = scipy_coo_to_csr(values, coordinates, jac_shape, return_decomposition=True)

    

    #rows_local = int(jac_shape[0] / size)

    #A = PETSc.Mat().createAIJ(size=jac_shape, csr=(iptr, j, values), bsize=[rows_local, jac_shape], comm=comm)
    A = PETSc.Mat().createAIJ(size=jac_shape, csr=(iptr.astype(np.int32), j.astype(np.int32), values), comm=comm)
    
    b = PETSc.Vec().createWithArray(b, comm=comm)
    
    x = b.duplicate()
    
    
    # Create a linear solver
    ksp = PETSc.KSP().create()
    ksp.setType(ksp_type)

    ksp.setOperators(A)
    #ksp.setFromOptions()
    
    
    if preconditioner == 'hypre':
        pc = ksp.getPC()
        pc.setType('hypre')
        pc.setHYPREType('boomeramg')
    else:
        pc = ksp.getPC()
        pc.setType(preconditioner)

    if precondition_only:
        pc.apply(b, x)
    else:
        ksp.solve(b, x)
    
    # Print the solution
    #x.view()

    x_jnp = jnp.array(x.getArray())

    return x_jnp


def sparse_linear_solve(values, coordinates, jac_shape, b, x0=None, mode="scipy-umfpack"):
    print("solver-mode: {}".format(mode))

    match mode:
        case "jax-scipy-bicgstab":
            coordinates = coordinates.T

            if x0 is None:
                x0 = jnp.zeros_like(b)

            #https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_array.html
            A = BCOO((values, coordinates.astype(jnp.int32)), shape=jac_shape)

            #If you don't include this preconditioner then things really go to shit
            diag_indices = jnp.where(coordinates[:, 0] == coordinates[:, 1])[0]
            jacobi_values = values[diag_indices]
            jacobi_indices = coordinates[diag_indices, :]
            M = BCOO((1.0 / jacobi_values, jacobi_indices.astype(jnp.int32)), shape=jac_shape)
            preconditioner = lambda x: M @ x

            x, info = jax.scipy.sparse.linalg.bicgstab(A, b, x0=x0, M=preconditioner,
                                                      tol=1e-10, atol=1e-10,
                                                      maxiter=10)

            # print(x)

            # Verify convergence
            residual = np.linalg.norm(b - A @ x)

        case "jax-native":
            iptr, j, values = scipy_coo_to_csr(values, coordinates, jac_shape, return_decomposition=True)
            x = jax.experimental.sparse.linalg.spsolve(values, j, iptr, b)
            residual = None

        case "scipy-umfpack":
            csr_array = scipy_coo_to_csr(values, coordinates, jac_shape)
            x = scipy.sparse.linalg.spsolve(csr_array, np.array(b), use_umfpack=True)

            residual = np.linalg.norm(b - csr_array @ x)

    return x, residual


def make_newton_solver_sparse_jac(C, f, n_iterations):

    residual_func = make_residual_function(C, f)

    basis_vectors, i_coordinate_sets\
            = basis_vectors_and_coords_2d_square_stencil(nr, nc, 1)

    i_coordinate_sets = jnp.concatenate(i_coordinate_sets)
    j_coordinate_sets = jnp.tile(jnp.arange(nr*nc), len(basis_vectors))
    mask = ~jnp.isnan(i_coordinate_sets)


    sparse_jacrev_func, _ = make_sparse_jacrev_fct_new(basis_vectors,\
                                             i_coordinate_sets,\
                                             j_coordinate_sets,\
                                             mask)


    i_coordinate_sets = i_coordinate_sets[mask]
    j_coordinate_sets = j_coordinate_sets[mask]
    coords = jnp.stack([i_coordinate_sets, j_coordinate_sets])
    def solver(u_trial):
        
        u = u_trial.copy()

        #interesting to see that it doesn't go that well if I do only one iteration
        #even though the problem is linear... wtf.
        for i in range(n_iterations):
            print(jnp.max(jnp.abs(residual_func(u))))

            residual_jac_sparse = sparse_jacrev_func(residual_func, (u,))

            rhs = -residual_func(u)

            #print(residual_jac_dense)
            #print(rhs)
            #raise
            
            print(residual_jac_sparse[mask].shape)
            print(coords.shape)
            
            #du, res = sparse_linear_solve(residual_jac_sparse[mask], coords, (nr*nc, nr*nc), rhs, mode="scipy-umfpack")
            #du, res = sparse_linear_solve(residual_jac_sparse[mask], coords, (nr*nc, nr*nc), rhs, mode="jax-native")
            #du, res = sparse_linear_solve(residual_jac_sparse[mask], coords, (nr*nc, nr*nc), rhs, mode="jax-scipy-bicgstab")
            
            #NOTE: Weird issue when we get to nr, nc = 5000
            #print(coords.max())
            #raise

            du = solve_petsc_sparse(residual_jac_sparse[mask], coords, (nr*nc, nr*nc), rhs, precondition_only=False)
            
            u += du

        print(jnp.max(jnp.abs(residual_func(u))))

        return u

    return solver



def spherical_wave(nr, nc, amplitude=1, frequency=10, wavelength=0.1):
    y = jnp.linspace(0, 1, nr)
    x = jnp.linspace(0, 1, nc)
    yy, xx = jnp.meshgrid(y, x, indexing='ij')

    cy, cx = (0.5, 0.5)
    r = jnp.sqrt((yy - cy)**2 + (xx - cx)**2)

    wave = amplitude * jnp.sin(2 * jnp.pi * frequency * r)

    return wave

nr = 4000
nc = 4000
dy = 1/nr
dx = 1/nc

C = spherical_wave(nr, nc, amplitude=1000)


#plt.imshow(C, vmin=0, vmax=1000)
#plt.show()
#raise

f = 1

u_init = jnp.ones_like(C).reshape(-1)

#1 should be enough as the problem is linear but seemingly benefits from another
n_iterations = 2
#solver = make_newton_solver(C, f, n_iterations)
solver = make_newton_solver_sparse_jac(C, f, n_iterations)

u_final = solver(u_init)


#plt.plot(u_final.reshape((nr,nc))[50,:])
#plt.show()
#raise

plt.imshow(u_final.reshape((nr,nc)), cmap="RdBu_r", vmin=-0.03, vmax=0.03)
plt.show()















