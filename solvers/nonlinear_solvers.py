#1st Party
import sys

#3rd Party
import jax
import jax.numpy as jnp
from jax import jacfwd, jacrev
import jax.scipy.linalg as lalg
from jax.scipy.optimize import minimize

#local apps
sys.path.insert(1, "./")
import linear_solvers as ls


def quasi_newton_coupled_comp(C, B_int, iterations, dt, bmr, acc=None):
    
    if acc is None:
        acc=accumulation.copy()

    mom_res = make_linear_momentum_residual_osd_at_gl()
    adv_res = make_adv_residual(dt, acc)

    jac_mom_res_fn = jacfwd(mom_res, argnums=(0,1))
    jac_adv_res_fn = jacfwd(adv_res, argnums=(0,1))
    
    def new_mu(u):
    
        dudx = jnp.zeros((n+1,))
        dudx = dudx.at[1:-1].set((u[1:] - u[:-1])/dx)
        dudx = dudx.at[-1].set(dudx[-2])
        #set reflection boundary condition
        dudx = dudx.at[0].set(2*u[0]/dx)
    
        mu_nl = B_int * (jnp.abs(dudx)+epsilon_visc)**(-2/3)

        #mu_nl = B * (epsilon_visc**(-2/3))

        return mu_nl

    def new_beta(u, h):

        grounded_mask = jnp.where((b+h)<(h*(1-rho/rho_w)), 0, 1)

        #beta = C * ((jnp.abs(u))**(-2/3)) * grounded_mask
        beta = C * (1/(jnp.abs(u)**(2/3) + (1e-8)**(2/3))) * grounded_mask
        #beta = C * (1e-6)**(-2/3) * grounded_mask

        return beta

    def continue_condition(state):
        _,_,_, i,res,resrat = state
        return i<iterations


    def step(state):
        u, h, h_init, i, prev_res, prev_resrat = state

        beta = new_beta(u, h)
        mu = new_mu(u)

        jac_mom_res = jac_mom_res_fn(u, h, mu, beta)
        jac_adv_res = jac_adv_res_fn(u, h, h_init, bmr)

        full_jacobian = jnp.block(
                                  [ [jac_mom_res[0], jac_mom_res[1]],
                                    [jac_adv_res[0], jac_adv_res[1]] ]
                                  )
    
        rhs = jnp.concatenate((-mom_res(u, h, mu, beta), -adv_res(u, h, h_init, bmr)))
    
        dvar = lalg.solve(full_jacobian, rhs)
    
        u = u.at[:].set(u+dvar[:n])
        h = h.at[:].set(h+dvar[n:])

        #TODO: add one for the adv residual too...
        res = jnp.max(jnp.abs(mom_res(u, h, mu, beta))) 

        return u, h, h_init, i+1, res, prev_res/res


    def iterator(u_init, h_init):    

        resrat = np.inf
        res = np.inf

        initial_state = u_init, h_init, h_init, 0, res, resrat

        u, h, h_init, itn, res, resrat = jax.lax.while_loop(continue_condition, step, initial_state)

        return u, h, res
       
    return iterator




