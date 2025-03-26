

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

import scipy

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm




def make_vto(s):

    def vto(u, mu):
        mu_longer = jnp.zeros((n+1,))
        mu_longer = mu_longer.at[:n].set(mu)

        dudx = jnp.zeros((n+1,))
        dudx = dudx.at[1:n].set((u[1:n] - u[:n-1])/dx)
        dudx = dudx.at[0].set(2*u[0]/dx)

        mu_nl = mu_longer * (jnp.abs(dudx)+epsilon)**(-2/3)
        # mu_nl = mu_longer.copy()

        flux = mu_nl * dudx

        sgrad = jnp.zeros((n,))
        sgrad = sgrad.at[-1].set(-s)

        return flux[1:n+1] - flux[:n] - sgrad

    return vto



def solve_petsc_dense(A, b, ksp_type='gmres', preconditioner=None, precondition_only=False):
    #ksp_type could be 'cg', 'bicg', 'bcgs', ...

    comm = MPI.COMM_WORLD
    size = comm.Get_size()

    print(f"Number of processes available: {size}")

    raise

    # Create PETSc matrix and vector from numpy arrays
    A = PETSc.Mat().createDense(comm=comm, size=A.shape, array=A)
    b = PETSc.Vec().createWithArray(b, comm=comm)

    # Create a solution vector
    x = b.duplicate()

    # Create a linear solver
    ksp = PETSc.KSP().create()
    ksp.setType(ksp_type)

    ksp.setOperators(A)
    ksp.setFromOptions()
    
    
    if preconditioner == 'hypre':
        pc = ksp.getPC()
        pc.setType('hypre')
        pc.setHYPREType('boomeramg')

    if precondition_only:
        pc.apply(b, x)
    else:
        ksp.solve(b, x)
    
    # Print the solution
    x.view()

    x_jnp = jnp.array(x.getArray())

    return x_jnp



#def solve_petsc_sparse():



def make_solver(u_trial, intermediates=False):

    def newton_solve(mu):
        vto = make_vto(s)

        vto_jac = jacfwd(vto)

        # vto_jac = jacrev(vto)

        u = u_trial.copy()

        if intermediates:
            us = [u]

        for i in range(10):
            jac = vto_jac(u, mu)
            rhs = -vto(u, mu)
            
            #du = lalg.solve(jac, rhs)
            du = solve_petsc_dense(jac, rhs, preconditioner='hypre')
            # du = linear_solve(jac, rhs) #making this change alone makes no difference to the hessian computation

            u = u.at[:].set(u+du)
            if intermediates:
              us.append(u)

        if intermediates:
          return u, us
        else:
          return u

    return newton_solve






lx = 1
n = 100
dx = lx/n
x = jnp.linspace(0,lx,n)
s = 1

epsilon = 1e-10

mu = jnp.zeros((n,)) + 1
mu = mu.at[:].set(1-(x/2))
mu = mu.at[int(2*n/3)].set(0.25)



u_trial = jnp.exp(x)-1



newton_solve_with_intermediates = make_solver(u_trial, intermediates=True)

u_end, us = newton_solve_with_intermediates(mu)




plt.figure(figsize=(10,5))
plt.plot(x, mu, c='k', linewidth=0.5, marker='x')
plt.gca().set_ylim(0, 1.2)
for axis in ['bottom','left']:
    plt.gca().spines[axis].set_linewidth(2.5)
for axis in ['top','right']:
    plt.gca().spines[axis].set_linewidth(0)
plt.tick_params(axis='both', which='major', labelsize=12)
plt.savefig("../misc/input_to_solve_native.png", dpi=100)


fig, ax = plt.subplots(figsize=(10,5))
colors = cm.coolwarm(jnp.linspace(0, 1, len(us)))
for i, u in enumerate(np.array(us) / 3.5):
    plt.plot(x, u, color=colors[i], label=str(i))
# change all spines
for axis in ['bottom','left']:
    plt.gca().spines[axis].set_linewidth(2.5)
for axis in ['top','right']:
    plt.gca().spines[axis].set_linewidth(0)
plt.tick_params(axis='both', which='major', labelsize=12)
plt.savefig("../misc/solve_native.png", dpi=100)








