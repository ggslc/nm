
import jax
import jax.numpy as jnp
from jax import jacfwd, jacrev
import jax.scipy.linalg as lalg
from jax.scipy.optimize import minimize

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


np.set_printoptions(precision=1, suppress=False, linewidth=np.inf)

#mu face-centred
#d face-centred

def make_vto_nl(h, beta, mu_face):

    s_gnd = h + b
    s_flt = h*(1-0.917/1.027)
    s = jnp.maximum(s_gnd, s_flt)
    s = s.at[-1].set(0)


    def vto(u, d):

        dudx = jnp.zeros((n+1,))
        dudx = dudx.at[1:n].set((u[1:n] - u[:n-1])/dx)
        #set reflection boundary condition
        dudx = dudx.at[0].set(2*u[0]/dx)

        
        sliding = beta * u * dx
        sliding = sliding.at[:].set(jnp.where(h>0, jnp.where(s_gnd>s_flt, sliding, 0), u * dx))

        
        h_face = jnp.zeros((n+1,))
        h_face = h_face.at[1:n-1].set(0.5 * (h[1:n-1] + h[:n-2]))
        h_face = h_face.at[0].set(h[0])


        flux = h_face * (1-d) * mu_face * dudx


        h_grad_s = jnp.zeros((n,))
        h_grad_s = h_grad_s.at[1:n-1].set(0.917 * h[1:n-1] * 0.5 * (s[2:] - s[:n-2]))
        h_grad_s = h_grad_s.at[0].set(0.917 * h[0] * 0.5 * (s[1] - s[0]))
        


        return flux[1:(n+1)] - flux[:n] - h_grad_s - sliding

    return vto



def make_adv_expt(dt, gamma=0.01, A=1):

    def adv(u, d):

        #NOTE: maybe best to do this with finite differences?


        #NOTE: Ignore below. I think we can do with hd i.e. l rather than d.
        #then convert l to d if needed elsewhere.



        dudx = jnp.zeros((n,))
        dudx = dudx.at[1:n-1].set((u[2:n] - u[:n-2])/dx)
        dudx = dudx.at[0].set((u[1] - u[0])/dx)
        #the last one doesn't matter I don't think
        
        #TODO: I don't think it makes sense to have anything face-centred to
        #be honest. Let's just colocate everything!

        mu_nl = mu * (jnp.abs(dudx)+epsilon)**(-2/3)

        #tau_xx = 0.5 * (1-d) * mu_nl * dudx

        source = gamma * A * dx * ((0.5 * mu_nl * dudx - rho * g * h * d)**4)


        hd_face = jnp.zeros((n+1,))
        hd_face = hd_face.at[1:n].set(h[:n-1] * d[:n-1]) #upwind values

        u_face = jnp.zeros((n+1,))
        u_face = u_face.at[1:n-1].set(0.5*(u[1:n-1]+u[:n-2]))
        u_face = u_face.at[-2].set(u[-2])

        thk_flux = h_face * u_face
        thk_flux = thk_flux.at[-1].set(thk_flux[-2])    

        dhdt = (h-h_old) * dx / dt

        return thk_flux[1:(n+1)] - thk_flux[:n] + dhdt
        
    return adv







