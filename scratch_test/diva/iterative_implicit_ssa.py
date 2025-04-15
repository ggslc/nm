
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

def make_vto(mu_face):

    def vto(u, h):

        C = C.at[-1].set(1)

        s_gnd = h + b
        s_flt = h*(1-0.917/1.027)
        s = jnp.maximum(s_gnd, s_flt)
        s = s.at[-1].set(0)


        p_W = 1.027 * jnp.maximum(0, h-s)
        p_I = 0.917 * h
        phi = 1 - (p_W / p_I)
        C = 300 * phi
        C = jnp.where(s_gnd>s_flt, C, 0)



        dudx = jnp.zeros((n+1,))
        dudx = dudx.at[1:n].set((u[1:n] - u[:n-1])/dx)
        #set reflection boundary condition
        dudx = dudx.at[0].set(2*u[0]/dx)

        # mu_nl = mu_longer * (jnp.abs(dudx)+epsilon)**(-2/3)
        # mu_nl = mu_longer.copy()


        sliding = C * u * dx

        
        h_face = jnp.zeros((n+1,))
        h_face = h_face.at[1:n-1].set(0.5 * (h[1:n-1] + h[:n-2]))
        h_face = h_face.at[0].set(h[0])


        flux = h_face * mu_face * dudx


        h_grad_s = jnp.zeros((n,))
        h_grad_s = h_grad_s.at[1:n-1].set(0.917 * h[1:n-1] * 0.5 * (s[2:] - s[:n-2]))
        h_grad_s = h_grad_s.at[0].set(0.917 * h[0] * 0.5 * (s[1] - s[0]))
        


        return flux[1:(n+1)] - flux[:n] - h_grad_s - sliding

    return vto



def make_adv(dt):

    def adv(u, h, h_old):

        u_face = jnp.zeros((n+1,))
        h_face = jnp.zeros((n+1,))

        u_face = u_face.at[1:n].set(0.5*(u[1:n]+u[:n-1]))
        h_face = h_face.at[1:n-1].set(h[:n-2]) #upwind values
        h_face = h_face.at[0].set(h[0]) #don't mind as u[0]=0

        thk_flux = h_face * u_face
    

        dhdt = (h-h_old) * dx / dt

        return thk_flux[1:(n+1)] - thk_flux[:n] - dhdt
        
    return adv




def do_iterations(h_init, u_init, dt, mu_face_va):

    vto = make_vto(mu_face_va)
    adv = make_adv(dt)
   
    jac_vto_fn = jacfwd(vto, argnums=0)
    jac_adv_fn = jacfwd(adv, argnums=1)

    u = u_init.copy()
    h = h_init.copy()

    h_old = h_init.copy()


    us = []
    hs = []

    for i in range(10):
        jac_vto = jac_vto_fn(u, h)

        u = u + lalg.solve(jac_vto, -vto(u, h))

        jac_adv = jac_adv_fn(u, h, h)

        h = h + lalg.solve(jac_adv, -adv(u, h, h_old))

        us.append(u)
        hs.append(h)


    return us, hs














