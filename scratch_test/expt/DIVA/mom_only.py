from pathlib import Path

from PIL import Image

import jax
import jax.numpy as jnp
from jax import jacfwd, jacrev
import jax.scipy.linalg as lalg
from jax.scipy.optimize import minimize

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


np.set_printoptions(precision=10, suppress=False, linewidth=np.inf, threshold=np.inf)

def interp_mu_onto_faces(mu_centre):
    mu_face = jnp.zeros((n+1,))
    mu_face = mu_face.at[1:-1].set(0.5 * (mu_centre[:-1] + mu_centre[1:]))
    mu_face = mu_face.at[-1].set(mu_centre[-1]) #doesn't actually do anything to mb as
                                                #dudx defined to be zero there, but
                                                #useful for translating back!
    mu_face = mu_face.at[0].set(mu_centre[1])
    return mu_face

def interp_mu_onto_centres(mu_face):
    mu_centre = jnp.zeros((n,))
    mu_centre = mu_centre.at[1:-1].set(0.5 * (mu_face[1:-2] + mu_face[2:-1]))
    mu_centre = mu_centre.at[0].set(mu_face[0])
    mu_centre = mu_centre.at[-1].set(mu_face[-1])
    return mu_centre


def make_linear_momentum_residual():
    #one-sided differences at the grounding line!

    def mom_res(u, h, mu_face, beta, b):

        s_gnd = h + b
        s_flt = h*(1-rho/rho_w)
        s = jnp.maximum(s_gnd, s_flt)
        
        is_floating = s_flt > s_gnd
        ffci = jnp.where(jnp.any(is_floating), jnp.argmax(is_floating), n)


        dudx = jnp.zeros((n+1,))
        dudx = dudx.at[1:n].set((u[1:n] - u[:n-1])/dx)
        #dudx = dudx.at[-2].set(dudx[-3])
        dudx = dudx.at[-1].set(0)
        ##set (or 'use' I guess) reflection boundary condition
        dudx = dudx.at[0].set(2*u[0]/dx)


        sliding = beta * u * dx
        #making sure the Jacobian is full rank!
        sliding = sliding.at[:].set(jnp.where(h>0, jnp.where(s_gnd>s_flt, sliding, 0), u * dx))

        
        h_face = jnp.zeros((n+1,))
        h_face = h_face.at[1:n].set(0.5 * (h[1:n] + h[:n-1]))
        h_face = h_face.at[-1].set(0)
        h_face = h_face.at[0].set(h[0])


        flux = h_face * mu_face * dudx


        h_grad_s = jnp.zeros((n,))
        h_grad_s = h_grad_s.at[1:n-1].set(h[1:n-1] * 0.5 * (s[2:n] - s[:n-2]))
        h_grad_s = h_grad_s.at[-1].set(-h[-1] * 0.5 * s[-2])
        #h_grad_s = h_grad_s.at[-2].set(-0.1)
        h_grad_s = h_grad_s.at[0].set(h[0] * 0.5 * (s[1] - s[0]))
      
        #one-sided differences at gl
        h_grad_s = h_grad_s.at[ffci].set(h[ffci] * (s[ffci+1] - s[ffci]))
        h_grad_s = h_grad_s.at[ffci-1].set(h[ffci-1] * (s[ffci-1] - s[ffci-2]))
     
        #scale
        h_grad_s = rho * g * h_grad_s

        #print(flux)
        #print(sliding)
        #print(h_grad_s)

        #plt.plot(-h_grad_s)
        #plt.plot(sliding)
        #plt.plot(flux)
        #plt.show()
        
        return flux[1:] - flux[:-1] - h_grad_s - sliding
        #return - h_grad_s - sliding
        #return flux[1:] - flux[:-1] - sliding

    return mom_res


def make_adv_residual(dt, accumulation):
    
    def adv_res(u, h, h_old, basal_melt_rate):
        s_gnd = h + b
        s_flt = h*(1-rho/rho_w)

        acc = jnp.where(s_gnd<s_flt, accumulation-basal_melt_rate, accumulation)
        acc = acc.at[:].set(jnp.where(h>0, acc, 0))

        h_face = jnp.zeros((n+1,))
        h_face = h_face.at[1:n+1].set(h[:n]) #upwind values
        h_face = h_face.at[0].set(h[0].copy())

        u_face = jnp.zeros((n+1,))
        u_face = u_face.at[1:n].set(0.5*(u[1:n]+u[:n-1]))
        u_face = u_face.at[-1].set(2*u[-1] - u[-2]) #extrapolating u (linear)

        h_flux = h_face * u_face
        #the two lines below were supposed to stop everything piling up in the last
        #cell but they had the opposite effect for some reason...
        #h_flux = h_flux.at[-2].set(h_flux[-3]) #stop everythin piling up at the end.
        #NOTE: This changes things a lot:
        #h_flux = h_flux.at[-1].set(h_flux[-2].copy())
        h_flux = h_flux.at[0].set(0)


        #return  (h - h_old)/dt - ( h_flux[1:(n+1)] - h_flux[:n] )/dx - acc
        #I think the above is a sign error in the flux!
        return  (h - h_old)*dx + dt*( h_flux[1:(n+1)] - h_flux[:n] ) - dt*dx*acc

    return adv_res








def make_mom_solver(C, B, thickness, iterations, compile_=False):

    mom_res = make_linear_momentum_residual()

    jac_mom_res_fn = jacfwd(mom_res, argnums=0)


    def new_viscosity(u_va, u_vv, dudz, b, s):
        pass
        #return mu_vv, mu_va


    def arthern_function(mu_vv, b, s, m=1):
        #s can be higher dimensional than b if we want to calc value up to multiple layers
        #be careful to make the integral consistent (b to s2 should be b to s1 plus s1 to s2).
        #Maybe a better way to do this is with something like "layer indices" and just pick
        #out layers defined by mu_vv...
        pass
        #return fm


    #f2 = arthern_function(mu_vv, b, s, 2)
    def new_beta_eff(beta, f2):

        beta_eff = beta / (1 + beta*f2)

        return beta_eff


    def new_dudz(mu_vv, u_va, beta_eff, s, z):
        
        dudz = (1/mu_vv) * beta_eff * (s-z) * (1/h)

        return dudz

 
    def new_u_vv(dudz, u_va, beta, beta_eff, h, s, b, z, mu_vv, f2):

        f1_vv = arthern_function(mu_vv, b, z, m=1)

        u_vv = (u_va / (1 + beta*f2)) + (beta_eff * u_va / h) * f1_vv


    #NEED TO SORT OUT CELL CENTRES VS CELL FACES
    def setup_and_solve_linear_prob(u_va, mu_va, beta_eff, h, b):

        jac_mom_res = jac_mom_res_fn(u_va, h, mu_va, beta_eff, b)

        delta_u = lalg.solve(jac_mom_res, -mom_res(u_va, h, mu_va, beta_eff, b))

        u_va = u_va + delta_u

        residual = jnp.max(jnp.abs(mom_res(u_va, h, mu_va, beta_eff, b)))

        return u_va, residual


    def step(state):
        pass


    def continue_condition(state):
        _,_,_,_,_,_,_,_,_, i = state
        return i<iterations


    def iterator(u_va_init, dudz_init, h, s, zs):
        
        residual_ratio = jnp.inf
        residual = 1

        initial_state = u_va_init, dudz_init#, etc 

        jax.lax.while_loop(continue_condition, step, initial_state)

        return #stuff





def make_joint_qn_solver(C, B, iterations, dt, acc=0, compile_=False):
    
    mom_res = make_linear_momentum_residual()
    adv_res = make_adv_residual(dt, acc)

    jac_mom_res_fn = jacfwd(mom_res, argnums=(0,1))
    jac_adv_res_fn = jacfwd(adv_res, argnums=(0,1))
    
    def new_mu(u):
    
        dudx = jnp.zeros((n+1,))
        dudx = dudx.at[1:-1].set((u[1:] - u[:-1])/dx)
        dudx = dudx.at[-1].set(dudx[-2])
        #set reflection boundary condition
        dudx = dudx.at[0].set(2*u[0]/dx)
    
        mu_nl = B * (jnp.abs(dudx)+epsilon_visc)**(-2/3)

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


    if compile_:
        return jax.jit(iterator)
    else:
        return iterator























