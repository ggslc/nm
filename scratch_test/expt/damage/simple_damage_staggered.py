

import jax
import jax.numpy as jnp
from jax import jacfwd, jacrev
import jax.scipy.linalg as lalg
from jax.scipy.optimize import minimize

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


np.set_printoptions(precision=1, suppress=False, linewidth=np.inf)

#u, beta cell centres
#h, mu, d, etc cell faces

def make_vto_nl(h, beta, mu):

    s_gnd = h + b
    s_flt = h*(1-0.917/1.027)
    s = jnp.maximum(s_gnd, s_flt)
    s = s.at[-1].set(0)


    def vto(u, d):


        dudx = jnp.zeros((n+1,))
        dudx = dudx.at[1:n-1].set((u[1:n-1] - u[:n-2])/dx)
        #set reflection boundary condition
        dudx = dudx.at[0].set(2*u[0]/dx)
        
        
        sliding = beta * u * dx

        sliding = sliding.at[:].set(jnp.where(h[1:]>0, jnp.where(s_gnd[1:]>s_flt[1:], sliding, 0), u * dx))

        
        mu_nl = mu * (jnp.abs(dudx)+epsilon)**(-2/3)


        d = d.at[-10].set(0.9)
        
        flux = h * (1-d) * mu_nl * dudx


        h_grad_s = jnp.zeros((n,))
        #h_grad_s = h_grad_s.at[:n-2].set(0.5*(h[:n-2]+h[1:n-1]) * (s[1:n-1] - s[:n-2]))
        #h_grad_s = h_grad_s.at[-2].set(-0.5*(h[-3]+h[-2]) * s[-3])
        h_grad_s = h_grad_s.at[:n-1].set(0.5*(h[:n-1]+h[1:n]) * (s[1:n] - s[:n-1]))
        h_grad_s = h_grad_s.at[-1].set(0)
       
        
        print("==================")
        print(flux)
        print(h_grad_s)
        print(sliding)
        print("==================")


        return flux[1:] - flux[:n] - 100*h_grad_s - 1000*sliding

    return vto



def make_adv(h, dt, gamma=1e4, A=1):

    def adv(u, d, d_old):

        hd = h*d
        hd_old = h*d_old
        
        dudx = jnp.zeros((n,))
        dudx = dudx.at[1:n-2].set((u[2:n-1] - u[:n-3])/dx)
        #leaving the last two to be zero
        dudx = dudx.at[0].set((u[1] - u[0])/dx)
        #the last one doesn't matter I don't think
        

        mu_nl = mu * (jnp.abs(dudx)+epsilon)**(-2/3)

        #tau_xx = 0.5 * (1-d) * mu_nl * dudx

        source = gamma * A * dx * ((0.5 * mu_nl * dudx - rho * g * hd)**4)
        source = source.at[-1].set(0)
        #print(source)

        hd_face = jnp.zeros((n+1,))
        hd_face = hd_face.at[1:n].set(hd[:n-1]) #upwind values

        u_face = jnp.zeros((n+1,))
        u_face = u_face.at[1:n-1].set(0.5*(u[1:n-1]+u[:n-2]))
        u_face = u_face.at[-2].set(u[-2])

        hd_flux = hd_face * u_face
        hd_flux = hd_flux.at[-1].set(hd_flux[-2])
        hd_flux = hd_flux.at[0].set(0)

        dhd_dt = (hd - hd_old) * dx / dt

        return dhd_dt - source

        #return hd_flux[1:(n+1)] - hd_flux[:n] + dhd_dt - source
        
    return adv



def make_u_solver(u_trial, num_iterations, d):

    def u_solve():

        vto = make_vto_nl(h, beta, mu)

        #vto(u_trial, d)
        #raise

        vto_jac_fn = jacfwd(vto, argnums=0)

        u = u_trial.copy()

        for i in range(num_iterations):
            vto_jac = vto_jac_fn(u, d)
            print(vto_jac)
            du = lalg.solve(vto_jac, -vto(u, d))
            u = u.at[:].set(u + du)

        return u

    return u_solve


def solver(u_trial, d_trial, dt, num_iterations, num_timesteps):

    def newton_solve():

        vto = make_vto_nl(h, beta, mu)
        adv = make_adv(h, dt)

        vto_jac_fn = jacfwd(vto, argnums=(0,1))
        adv_jac_fn = jacfwd(adv, argnums=(0,1))

        u = u_trial.copy()
        d = d_trial.copy()
        d_old = d_trial.copy()

        ds = []
        us = []

        for j in range(num_timesteps):
            print(j)
            for i in range(num_iterations):
                vto_jac = vto_jac_fn(u, d)
                adv_jac = adv_jac_fn(u, d, d_old)


                full_jacobian = jnp.block(
                                          [[vto_jac[0], vto_jac[1]],
                                          [adv_jac[0], adv_jac[1]]]
                                          )[:2*n-1, :2*n-1]

                
                #print(full_jacobian)
                #raise

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


                rhs = jnp.concatenate((-vto(u, d), -adv(u, d, d_old)[:n-1]))

                dvar = lalg.solve(full_jacobian, rhs)

       #         print(dvar)

                u = u.at[:].set(u+dvar[:n])
                d = d.at[:n-1].set(d[:n-1]+dvar[n:2*n-1])

                print(jnp.max(jnp.abs(vto(u, d))), jnp.max(jnp.abs(adv(u, d, d_old))))



            ds.append(d)
            us.append(u)

#            plotboth(h, u, title="Timestep {}, iteration {}".format(j+1, i),\
#                    savepath="../misc/full_implicit_tests/{}_{}.png".format(j+1,i),\
#                    axis_limits = [[-15, 30],[0, 150]], show_plots=False)

            d_old = d.copy()


            #plt.plot(d)
            #plt.ylim(-0.2, 1.2)
            #plt.show()
            #
            #plt.plot(u)
            #plt.show()

        return u, d, ds, us

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


def plotboths(ds, speeds, upper_lim, title=None, savepath=None, axis_limits=None, show_plots=True):

    fig, ax1 = plt.subplots(figsize=(10,6))
    ax2 = ax1.twinx()

    n = len(ds)

    cmap = cm.rainbow
    cs = cmap(jnp.linspace(0, 1, n))

    for d, speed, c1 in list(zip(ds, us, cs)):
        ax1.plot(d, c=c1)
        ax2.plot(speed, color=c1, marker=".", linewidth=0)

    #add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=upper_lim))
    sm._A = []
    cbar = fig.colorbar(sm, ax=ax1, orientation='horizontal', pad=0.15)
    cbar.set_label('timestep')

    #axis labels
    ax1.set_xlabel("x")
    ax1.set_ylabel("damage")
    ax2.set_ylabel("speed")

    if axis_limits is not None:
        ax1.set_ylim(axis_limits[0])
        ax2.set_ylim(axis_limits[1])

    if title is not None:
        plt.title(title)
    if savepath is not None:
        plt.savefig(savepath)

    if show_plots:
        plt.show()




rho = 1
g = 1



lx = 1
n = 100
dx = lx/n
x = jnp.linspace(0,lx,n)


mu = jnp.zeros((n+1,)) + 1


#OVERDEEPENED BED


h = jnp.zeros((n+1,))
h = h.at[:n].set(5*jnp.exp(-2*x*x*x*x))


b_intermediate = jnp.zeros((n+1,))-2

s_gnd = b_intermediate + h
s_flt = h*(1-0.917/1.027)
s = jnp.maximum(s_gnd, s_flt)
s = s.at[-1].set(0)


b = jnp.zeros((n+1,))-2

h = jnp.minimum(s-b, s/(1-0.917/1.027))
h = h.at[-1].set(0)


# #linear sliding, constant beta:
# beta = jnp.where(s_gnd>s_flt, 1, 0)

p_W = 1.027 * jnp.maximum(0, h[1:]-s[1:])
p_I = 0.917 * h[1:]
phi = 1 - (p_W / p_I)
#phi = 1
beta = 1 * phi
beta = jnp.where(s_gnd[1:]>s_flt[1:], beta, 0)


base = s - h


epsilon = 1e-10




#plotgeom(h)
#raise



##It's hard to tell exactly, but I think this is working as expected...
#u_solve = make_u_solver(jnp.exp(x)-1, 10)
#
#u_end = u_solve()
#
#plt.plot(u_end)
#plt.show()
#
#raise


d_test = jnp.zeros((n+1,))
#d_test = d_test.at[80:81].set(0.75)

u_solve = make_u_solver(jnp.exp(x)-1, 10, d_test)

u_end = u_solve()

print(u_end)

plt.plot(u_end)
plt.show()


raise

##Coupled, time-dependent problem:

dt = 0.01
num_iterations = 30
num_timesteps = 5

u_trial = jnp.exp(x)-1
d_trial = jnp.zeros((n,))

ns = solver(u_trial, d_trial, dt, num_iterations, num_timesteps)

u_end, d_end, ds, us = ns()

plotboths(ds, us, num_timesteps, axis_limits=[[-0.2, 1.2], [0,1.5]])

raise



