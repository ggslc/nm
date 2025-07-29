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
from mpi4py import MPI

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
        x_flux_diffs = x_flux_diffs.at[:, 0].set((dy/dx) * (-2*u_2d[:,0]))
        x_flux_diffs = x_flux_diffs.at[:, -1].set((dy/dx) * (-2*u_2d[:,-1]))

        y_flux_diffs = y_flux_diffs.at[1:-1, :].set((dx/dy) * (u_2d[2:,:] + u_2d[:-2,:] - 2*u_2d[1:-1,:]))
        y_flux_diffs = y_flux_diffs.at[0, :].set((dx/dy) * (-2*u_2d[0,:]))
        y_flux_diffs = y_flux_diffs.at[-1, :]  .set((dx/dy) * (-2*u_2d[-1,:]))


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

        for i in range(n_iterations):

            residual_jac_sparse = sparse_jacrev_func(residual_func, (u,))

            residual_jac_dense = densify_func(residual_jac_sparse, nr*nc)

            rhs = -residual_func(u)

            print(residual_jac_dense)
            print(rhs)
            raise

            u += lalg.solve(residual_jac_dense, rhs)

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

nr = 5
nc = 5
dy = 1/nr
dx = 1/nc

C = spherical_wave(nr, nc)


#plt.imshow(C, vmin=0, vmax=1)
#plt.show()
#raise

f = 1

u_init = jnp.ones_like(C).reshape(-1)

n_iterations = 10
solver = make_newton_solver(C, f, n_iterations)

u_final = solver(u_init)

plt.imshow(u_final.reshape((nr,nc)))
plt.show()















