""" 1D Ice shelf tests """

#external
import jax
import jax.numpy as jnp
from jax import jacfwd, jacrev
import matplotlib.pyplot as plt
import jax.scipy.linalg as lalg
from jax.scipy.optimize import minimize
from jax import custom_vjp

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

#internal
import os

#local apps
# - 





@custom_vjp
@jax.jit
def linear_solve(A, b):
    # the reason for writing it out like this, despite the fact that lalg.solve
    #is already differentiable and probably has a primitive is that we want to
    #be able to easily replace the solver.
    solution = lalg.solve(A, b)
    return solution

def linear_solve_fwd(A, b):
    solution = linear_solve(A, b)
    return solution, (A, b, solution)

def linear_solve_bwd(res, c_bar):
    A, b, x = res

    lambda_ = linear_solve(jnp.transpose(A), c_bar)
    # calling this here means that higher-order derivatives can be calculated without further definition.

    b_bar = lambda_.copy()
    A_bar = - jnp.outer(lambda_, x)

    return A_bar, b_bar

linear_solve.defvjp(linear_solve_fwd, linear_solve_bwd)





def make_solver_custom_vjp(u_trial, intermediates=False):

    @custom_vjp
    @jax.jit
    def newton_solve(mu):
        vto = make_vto(s)
        vto_jac = jacfwd(vto, argnums=0)
        # vto_jac = jacrev(vto)

        u = u_trial.copy()

        if intermediates:
            us = [u]

        for i in range(10):
            jac = vto_jac(u, mu)
            rhs = -vto(u, mu)
            # du = lalg.solve(jac, rhs)
            du = linear_solve(jac, rhs)
            u = u.at[:].set(u+du)
            if intermediates:
              us.append(u)

        if intermediates:
          return u, us
        else:
          return u


    def newton_solve_fwd(mu):
      #what residuals do we need eh? (i.e. gubbins we need to keep track of for the backward pass)
      vto = make_vto(s)

      u = newton_solve(mu) #REALLY NEED TO THINK ABOUT WHAT TO DO IF INTERMEDIATES!!!


      #use AD enjine to calculate the jacobians (so we're not going completely oldschool)
      # vto_jac_u_func = jacfwd(vto, argnums=0)
      # vto_jac_mu_func = jacfwd(vto, argnums=1)
      # vto_jac_u = vto_jac_u_func(u, mu)
      # vto_jac_mu = vto_jac_mu_func(u, mu)
      # AKA (just keeping stuff around until everyone is very happy with JAX):
      vto_jac_u = jacfwd(vto, argnums=0)(u, mu) #can retrieve this from fwd computation if do properly I think.
      vto_jac_mu = jacfwd(vto, argnums=1)(u, mu)

      #What's kind of nuts is that using u_trial, instead of the current u makes this waayyyyy faster for the
      #inverse problem! THIS DESERVES SOME SERIOUS THOUGHT!!!!!!!!

      #this is a little weird... It seems like we ought to be able to take some
      #of the stuff from the /actual/ forward run, rather than running some of
      #it twice...

      #Returns primal output and residuals to be used in backward pass by f_bwd.
      return u, (vto_jac_u, vto_jac_mu)


    def newton_solve_bwd(res, u_bar):
      vto_jac_u, vto_jac_mu = res # Gets residuals computed in f_fwd

      vto_jac_u_transpose = jnp.transpose(vto_jac_u)
      vto_jac_mu_transpose = jnp.transpose(vto_jac_mu)

      #call to solve adjoint problem:
      # lambda_ = lalg.solve(vto_jac_u_transpose, -u_bar)
      lambda_ = linear_solve(vto_jac_u_transpose, -u_bar)

      mu_bar = jnp.matmul(vto_jac_mu_transpose, lambda_)

      return (mu_bar,)

    newton_solve.defvjp(newton_solve_fwd, newton_solve_bwd)

    return newton_solve


