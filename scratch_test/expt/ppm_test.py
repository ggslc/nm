
import jax
import jax.numpy as jnp
from jax import jacfwd, jacrev
import jax.scipy.linalg as lalg
from jax.scipy.optimize import minimize

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


np.set_printoptions(precision=1, suppress=False, linewidth=np.inf)


def make_ppm_step(u, dt, h_in, dx):
    #assume u and h are colocated
    n = u.shape[0]

    def ppm_step(h):
        #follow steps laid out in:
        #Application of the Piecewise Parabolic Method (PPM) to Meteorological Modeling
        #by carpenter et al., 1989
        #and stuff learned from Colella and Woodward 1984.

        #Boundary condition is that h is prescribed at the lhs of the domain.


        #STEP 1: 
        #Approximate piecewise linear profile.
        #central difference, left-sided diff, right-sided diff
        #located on cell centres
        h_grad_cd = jnp.zeros((n,))
        h_grad_lsd = jnp.zeros((n,))
        h_grad_rsd = jnp.zeros((n,))

        h_grad_cd  = h_grad_cd.at[1:n-1].set((h[2:] - h[:n-1])/dx)
        h_grad_cd  = h_grad_cd.at[-1].set(0)
        h_grad_cd  = h_grad_cd.at[0].set(0)
        h_grad_lsd = h_grad_lsd.at[1:].set((h[1:] - h[:n-1])/dx)
        h_grad_lsd = h_grad_lsd.at[0].set(0)
        h_grad_rsd = h_grad_rsd.at[:n-1].set((h[1:] - h[:n-1])/dx)
        h_grad_rsd = h_grad_rsd.at[-1].set(0)

        h_grad_cd_mag = jnp.abs(h_grad_cd)
        h_grad_lsd_mag = jnp.abs(h_grad_lsd)
        h_grad_rsd_mag = jnp.abs(h_grad_rsd)

        min_mag_grads_cd_lsd = jnp.where(h_grad_cd_mag < h_grad_lsd_mag, h_grad_cd, h_grad_lsd)
        min_mag_grads_cd_rsd = jnp.where(h_grad_cd_mag < h_grad_rsd_mag, h_grad_cd, h_grad_rsd)
        min_mag_grads = jnp.where(min_mag_grads_cd_lsd < min_mag_grads_cd_rsd,\
                                   min_mag_grads_cd_lsd, min_mag_grads_cd_rsd)

        h_r = jnp.zeros_like((n-1,))
        h_l = jnp.zeros_like((n-1,))
        h_r = h_r.at[:].set(h[:n-1]+min_mag_grads[:n-1]*0.5*dx)
        h_l = h_l.at[:].set(h[1:]-min_mag_grads[1:]*0.5*dx)



        #STEP 2:
        #First guess at face-centred values by fitting cubic to grads and values
        h_ls = jnp.zeros((n-1,))
        h_ls = h_ls.at[:].set( (1/2)*(h[:n-1]+h[1:]) - (1/6)*(min_mag_grads[1:]-min_mag_grads[:n-1]) )

        h_rs = h_ls.copy()



        #STEP 3:
        #STEEPENING! Let's ignore this for now :)


        #STEP 4:
        h_six = jnp.zeros((n,))
        h_six = h_six.at[0].set(6*h[0])
        h_six = h_six.at[-1].set(6*h[-1])
        h_six = h_six.at[1:n-1].set(6*(h[1:n-1] - 0.5*(h_ls[:n-2] + h_rs[1:n-1])))
        
        delta_h = jnp.zeros((n,))
        delta_h = delta_h.at[1:n-1].set((h_rs[:n-2] - h_ls[1:n-1])/dx)
        delta_h = delta_h.at[0].set(0)
        delta_h = delta_h.at[-1].set(0)


        condition_1 = jnp.where(( delta_h**2 < (delta_h*h_six)), 1, 0)
        condition_2 = jnp.where((-delta_h**2 < (delta_h*h_six)), 1, 0)
        #TODO: LEFT THINGS HERE FOR THE WEEKEND, WORK OUT WTF IS GOING ON
        condition_3 = jnp.where(((h_rs[])*() < 0), 1, 0)
        #TODO: I've realised this is a bad way of doing things. Each h point
        #should have a h_ls and h_rs even if the [0] for l and [-1] for r
        #are just placeholders. It just makes things easier. Change in the code!!!





       return h_new

    return ppm_step


