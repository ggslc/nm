
import jax
import jax.numpy as jnp
from jax import jacfwd, jacrev
import jax.scipy.linalg as lalg
from jax.scipy.optimize import minimize

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


np.set_printoptions(precision=1, suppress=False, linewidth=np.inf)


def make_vto_nl(h, beta, mu_cc):

    mu_face = jnp.zeros((n+1,))
    mu_face = mu_face.at[1:n-1].set(0.5 * (mu_cc[1:n-1] + mu_cc[:n-2]))
    mu_face = mu_face.at[0].set(mu_cc[0])

    s_gnd = h + b
    s_flt = h*(1-0.917/1.027)
    s = jnp.maximum(s_gnd, s_flt)
    s = s.at[-1].set(0)


    def vto(u, d):

        d_face = jnp.zeros((n+1,))
        d_face = d_face.at[1:n-1].set(0.5 * (d[1:n-1] + d[:n-2]))
        d_face = d_face.at[0].set(d[0])


        dudx = jnp.zeros((n+1,))
        dudx = dudx.at[1:n].set((u[1:n] - u[:n-1])/dx)
        #set reflection boundary condition
        dudx = dudx.at[0].set(2*u[0]/dx)

        
        sliding = beta * u * dx
        sliding = sliding.at[:].set(jnp.where(h>0, jnp.where(s_gnd>s_flt, sliding, 0), u * dx))

        
        h_face = jnp.zeros((n+1,))
        h_face = h_face.at[1:n-1].set(0.5 * (h[1:n-1] + h[:n-2]))
        h_face = h_face.at[0].set(h[0])


        mu_face_nl = mu_face * (jnp.abs(dudx)+epsilon)**(-2/3)


        flux = h_face * (1-d_face) * mu_face * dudx


        h_grad_s = jnp.zeros((n,))
        h_grad_s = h_grad_s.at[1:n-1].set(0.917 * h[1:n-1] * 0.5 * (s[2:] - s[:n-2]))
        h_grad_s = h_grad_s.at[0].set(0.917 * h[0] * 0.5 * (s[1] - s[0]))
        


        return flux[1:(n+1)] - flux[:n] - h_grad_s - sliding

    return vto



def make_adv(h, dt, gamma=0.001, A=1):

    def adv(u, d, d_old):

        hd = h*d
        hd_old = h*d_old
        
        dudx = jnp.zeros((n,))
        dudx = dudx.at[1:n-1].set((u[2:n] - u[:n-2])/dx)
        dudx = dudx.at[0].set((u[1] - u[0])/dx)
        #the last one doesn't matter I don't think
        

        mu_nl = mu * (jnp.abs(dudx)+epsilon)**(-2/3)

        #tau_xx = 0.5 * (1-d) * mu_nl * dudx

        source = gamma * A * dx * ((0.5 * mu_nl * dudx - rho * g * hd)**4)


        hd_face = jnp.zeros((n+1,))
        hd_face = hd_face.at[1:n].set(hd[:n-1]) #upwind values

        u_face = jnp.zeros((n+1,))
        u_face = u_face.at[1:n-1].set(0.5*(u[1:n-1]+u[:n-2]))
        u_face = u_face.at[-2].set(u[-2])

        hd_flux = hd_face * u_face
        hd_flux = hd_flux.at[-1].set(hd_flux[-2])

        dhd_dt = (hd - hd_old) * dx / dt

        return hd_flux[1:(n+1)] - hd_flux[:n] + dhd_dt - source
        
    return adv


def solver(u_trial, d_trial, dt, num_iterations, num_timesteps):

    def newton_solve():

        vto = make_vto_nl(h, beta, mu)
        adv = make_adv(h, dt)

        vto_jac_fn = jacfwd(vto, argnums=(0,1))
        adv_jac_fn = jacfwd(adv, argnums=(0,1))

        u = u_trial.copy()
        d = d_trial.copy()
        d_old = d_trial.copy()


        for j in range(num_timesteps):
            print(j)
            for i in range(num_iterations):
                vto_jac = vto_jac_fn(u, d)
                adv_jac = adv_jac_fn(u, d, d_old)


                full_jacobian = jnp.block(
                                          [[vto_jac[0], vto_jac[1]],
                                          [adv_jac[0], adv_jac[1]]]
                                )
                print(full_jacobian)
                raise

                #print(np.array(vto_jac[0]))
                #print("-------------------")
                #print("-------------------")
                #print("-------------------")
                #print(np.array(advo_jac[0]))
                #print("-------------------")
                #print("-------------------")
                #print("-------------------")
                #print(np.array(full_jacobian))
                #raise


                rhs = jnp.concatenate((-vto(u, d), -advo(u, d, d_old)))

                dvar = lalg.solve(full_jacobian, rhs)

                u = u.at[:].set(u+dvar[:n])
                d = d.at[:].set(d+dvar[n:])




            dus.append([d, u])

#            plotboth(h, u, title="Timestep {}, iteration {}".format(j+1, i),\
#                    savepath="../misc/full_implicit_tests/{}_{}.png".format(j+1,i),\
#                    axis_limits = [[-15, 30],[0, 150]], show_plots=False)

            d_old = d.copy()



        return u, d, dus

    return newton_solve



def plotgeom(thk):

    s_gnd = b + thk
    s_flt = thk*(1-0.917/1.027)
    s = jnp.maximum(s_gnd, s_flt)

    base = s-thk

    #plot b, s and base on lhs y axis, and beta on rhs y axis
    fig, ax1 = plt.subplots(figsize=(10,5))

    ax1.plot(s, label="surface")
    # ax1.plot(base, label="base")
    ax1.plot(base, label="base")
    ax1.plot(b, label="bed")

    #legend
    ax1.legend(loc='upper right')

    #axis labels
    ax1.set_xlabel("x")
    ax1.set_ylabel("elevation")

    plt.show()



rho = 1
g = 1



lx = 1
n = 10
dx = lx/n
x = jnp.linspace(0,lx,n)


mu = jnp.zeros((n,)) + 1


#OVERDEEPENED BED


h = 5*jnp.exp(-2*x*x)
h = h.at[-1].set(0)


b_intermediate = jnp.zeros((n,))-2

s_gnd = b_intermediate + h
s_flt = h*(1-0.917/1.027)
s = jnp.maximum(s_gnd, s_flt)
s = s.at[-1].set(0)


b = jnp.zeros((n,))-2

h = jnp.minimum(s-b, s/(1-0.917/1.027))
h = h.at[-1].set(0)


# #linear sliding, constant beta:
# beta = jnp.where(s_gnd>s_flt, 1, 0)

phi = 1
beta = 300 * phi
beta = jnp.where(s_gnd>s_flt, beta, 0)


base = s - h


epsilon = 1e-10




#plotgeom(h)
#raise


dt = 0.0001
num_iterations = 10
num_timesteps = 10

u_trial = jnp.exp(x)-1
d_trial = jnp.zeros((n,))

ns = solver(u_trial, d_trial, dt, num_iterations, num_timesteps)

u_end, d_end, dus = ns()

raise




h_init = h.copy()
u_init = jnp.zeros((n,)) + 1
dt = 5e-2

rough_cn = 0.7 * dt/dx
print("cn roughly ", rough_cn)

mu_face_va = jnp.zeros((n+1,))+1

timesteps = 100
iterations = 20


h_t = h_init.copy()
u_t = u_init.copy()


hs = []
us = []


#us, hs, mus = do_iterations_nl(h_t, u_t, dt, mu_face_va, beta, iterations)


for i in range(timesteps):
    print(i)
    u_ts, h_ts, mu_ts = do_iterations_nl(h_t, u_t, dt, mu_face_va, beta, iterations)
    
    u_t = u_ts[-1]
    h_t = h_ts[-1]
    mu_t = mu_ts[-1]

    hs.append(h_t.copy())
    us.append(u_t.copy())


#print(us)
#print(hs)
#plotboths(hs[::5], us[::5], iterations)
plotboths(hs, us, timesteps)

#plotboth(hs[-1], us[-1])
  


