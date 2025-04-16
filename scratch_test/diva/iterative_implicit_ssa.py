
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

def make_vto(mu_face, beta):
    
    def vto(u, h):

        s_gnd = h + b
        s_flt = h*(1-0.917/1.027)
        s = jnp.maximum(s_gnd, s_flt)
        s = s.at[-1].set(0)


        #p_W = 1.027 * jnp.maximum(0, h-s)
        #p_I = 0.917 * h
        ##phi = 1 - (p_W / p_I)
        #phi = 1
        #beta = beta.at[:].set(beta * phi)
        
        #beta = beta.at[:].set(jnp.where(s_gnd>s_flt, beta, 0))

        #beta = beta[:].set(jnp.where(h > 0, beta, 1))


        dudx = jnp.zeros((n+1,))
        dudx = dudx.at[1:n].set((u[1:n] - u[:n-1])/dx)
        #set reflection boundary condition
        dudx = dudx.at[0].set(2*u[0]/dx)

        # mu_nl = mu_longer * (jnp.abs(dudx)+epsilon)**(-2/3)
        # mu_nl = mu_longer.copy()


        sliding = beta * u * dx
        sliding = sliding.at[:].set(jnp.where(h>0, jnp.where(s_gnd>s_flt, sliding, 0), u * dx))

        
        h_face = jnp.zeros((n+1,))
        h_face = h_face.at[1:n-1].set(0.5 * (h[1:n-1] + h[:n-2]))
        h_face = h_face.at[0].set(h[0])


        flux = h_face * mu_face * dudx


        h_grad_s = jnp.zeros((n,))
        h_grad_s = h_grad_s.at[1:n-1].set(0.917 * h[1:n-1] * 0.5 * (s[2:] - s[:n-2]))
        h_grad_s = h_grad_s.at[0].set(0.917 * h[0] * 0.5 * (s[1] - s[0]))
        


        return flux[1:(n+1)] - flux[:n] - h_grad_s - sliding

    return vto


def make_vto_nl(beta):

    def vto(u, h, mu_face):

        s_gnd = h + b
        s_flt = h*(1-0.917/1.027)
        s = jnp.maximum(s_gnd, s_flt)
        s = s.at[-1].set(0)


        dudx = jnp.zeros((n+1,))
        dudx = dudx.at[1:n].set((u[1:n] - u[:n-1])/dx)
        #set reflection boundary condition
        dudx = dudx.at[0].set(2*u[0]/dx)

        
        sliding = beta * u * dx
        sliding = sliding.at[:].set(jnp.where(h>0, jnp.where(s_gnd>s_flt, sliding, 0), u * dx))

        
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

        u_face = u_face.at[1:n-1].set(0.5*(u[1:n-1]+u[:n-2]))
        u_face = u_face.at[-2].set(u[-2])

        h_face = h_face.at[1:n].set(h[:n-1]) #upwind values
        h_face = h_face.at[0].set(h[0]) #don't mind as u[0]=0

        thk_flux = h_face * u_face
        thk_flux = thk_flux.at[-1].set(thk_flux[-2])    

        dhdt = (h-h_old) * dx / dt

        return thk_flux[1:(n+1)] - thk_flux[:n] + dhdt
        
    return adv





def do_iterations(h_init, u_init, dt, mu_face_va, beta, iterations):

    vto = make_vto(mu_face_va, beta)
    adv = make_adv(dt)
   
    #vto(u_init, h_init)
    #raise

    jac_vto_fn = jacfwd(vto, argnums=0)
    jac_adv_fn = jacfwd(adv, argnums=1)

    u = u_init.copy()
    h = h_init.copy()

    h_old = h_init.copy()


    us = []
    hs = []

    for i in range(iterations):

        jac_vto = jac_vto_fn(u, h)

        u = u + lalg.solve(jac_vto, -vto(u, h))
        
        
        jac_adv = jac_adv_fn(u, h, h)
        
        #print(jac_adv)
        #raise

        h = h + lalg.solve(jac_adv, -adv(u, h, h_old))

        
        residual = jnp.max(jnp.abs(vto(u, h)))
        print(residual)

        if residual < 1e-3:
            return us[-1], hs[-1]
        
        us.append(u)
        hs.append(h)


    return us[-1], hs[-1]


def new_mu(mu_faces_zero, u):

    dudx = jnp.zeros((n+1,))
    dudx = dudx.at[1:n].set((u[1:n] - u[:n-1])/dx)
    #set reflection boundary condition
    dudx = dudx.at[0].set(2*u[0]/dx)

    mu_nl = mu_faces_zero * (jnp.abs(dudx)+epsilon+1e-6)**(-2/3)

    return mu_nl


def do_iterations_nl(h_init, u_init, dt, mu_face_va, beta, iterations):

    vto = make_vto_nl(beta)
    adv = make_adv(dt)
   
    #vto(u_init, h_init)
    #raise

    jac_vto_fn = jacfwd(vto, argnums=0)
    jac_adv_fn = jacfwd(adv, argnums=1)

    u = u_init.copy()
    h = h_init.copy()

    h_old = h_init.copy()


    us = [u_init]
    hs = [h_init]
    mus = [mu_face_va]

    alpha = 1

    for i in range(iterations):
        mu_face = new_mu(mu_face_va, u)


        jac_vto = jac_vto_fn(u, h, mu_face)

        u = u + alpha*lalg.solve(jac_vto, -vto(u, h, mu_face))
        
        
        jac_adv = jac_adv_fn(u, h, h)
        
        #print(jac_adv)
        #raise

        h = h.copy()

        h = h + lalg.solve(jac_adv, -adv(u, h, h_old))

        #print(u)
        #print(h)
        #print(mu_face)
        #
        residual = jnp.max(jnp.abs(vto(u, h, mu_face)))
        print(residual)

        if residual < 1e-3:
            return us, hs, mus
        
        us.append(u)
        hs.append(h)
        mus.append(mu)


    return us, hs, mus


def plotu(u):

    fig, ax = plt.subplots(figsize=(10,5))

    # colors = cm.coolwarm(jnp.linspace(0, 1, len(us)))

    # for i, u in enumerate(np.array(us) / 3.5):
    #     plt.plot(x, u, color=colors[i], label=str(i))
    # # change all spines
    # for axis in ['bottom','left']:
    #     plt.gca().spines[axis].set_linewidth(2.5)
    # for axis in ['top','right']:
    #     plt.gca().spines[axis].set_linewidth(0)

    plt.plot(x, u, color='k')

    #increase size of x and y tick labels
    plt.tick_params(axis='both', which='major', labelsize=12)

    #axis label
    plt.gca().set_ylabel("speed")

    plt.show()

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




def plotboths(thks, speeds, upper_lim, title=None, savepath=None, axis_limits=None, show_plots=True):

    fig, ax1 = plt.subplots(figsize=(10,6))
    ax2 = ax1.twinx()

    n = len(thks)

    cmap = cm.rainbow
    cs = cmap(jnp.linspace(0, 1, n))

    for thk, speed, c1 in list(zip(thks, speeds, cs)):
        s_gnd = b + thk
        s_flt = thk*(1-0.917/1.027)
        s = jnp.maximum(s_gnd, s_flt)

        base = s-thk
        ax1.plot(s, c=c1)
        # ax1.plot(base, label="base")
        ax1.plot(base, c=c1)

        ax2.plot(speed, color=c1, marker=".", linewidth=0)

    #add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=upper_lim))
    sm._A = []
    cbar = fig.colorbar(sm, ax=ax1, orientation='horizontal', pad=0.15)
    cbar.set_label('iterate')

    ax1.plot(b, label="bed", c="k")
    
    ##legend
    ##ax1.legend(loc='lower left')
    ##slightly lower
    #ax2.legend(loc='center left')
    ##stop legends overlapping

    #axis labels
    ax1.set_xlabel("x")
    ax1.set_ylabel("elevation")
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




def plotboth(thk, speed, title=None, savepath=None, axis_limits=None, show_plots=True):
    s_gnd = b + thk
    s_flt = thk*(1-0.917/1.027)
    s = jnp.maximum(s_gnd, s_flt)

    base = s-thk


    fig, ax1 = plt.subplots(figsize=(10,5))
    ax2 = ax1.twinx()

    ax1.plot(s, label="surface")
    # ax1.plot(base, label="base")
    ax1.plot(base, label="base")
    ax1.plot(b, label="bed")

    ax2.plot(speed, color='k', marker=".", linewidth=0, label="speed")

    #legend
    ax1.legend(loc='lower left')
    #slightly lower
    ax2.legend(loc='center left')
    #stop legends overlapping

    #axis labels
    ax1.set_xlabel("x")
    ax1.set_ylabel("elevation")
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


lx = 1
n = 100
dx = lx/n
x = jnp.linspace(0,lx,n)


mu_base = 0.1
mu = jnp.zeros((n,)) + 1
mu = mu.at[:].set(mu*mu_base)



#accumulation = jnp.zeros((n+1,))
#accumulation = accumulation.at[:n].set(500)



#OVERDEEPENED BED


h = 20*jnp.exp(-0.5*x*x*x*x)
h = h.at[-1].set(0)


b_intermediate = jnp.zeros((n,))-12
# b = b.at[:n].set(b[:n] - 4*jnp.exp(-(5*x-2)**2))

s_gnd = b_intermediate + h
s_flt = h*(1-0.917/1.027)
s = jnp.maximum(s_gnd, s_flt)
s = s.at[-1].set(0)

b = jnp.zeros((n,))-12
# b = b.at[:n].set(b[:n] - 4*jnp.exp(-(5*x-2)**2))
# b = b.at[:n].set(b[:n] - 4*jnp.exp(-(5*x-3)**2))
b = b.at[:].set((x**0.5)*(b - 5*jnp.exp(-(5*x-3)**2)))

h = jnp.minimum(s-b, s/(1-0.917/1.027))
h = h.at[-1].set(0)


# #linear sliding, constant beta:
# beta = jnp.where(s_gnd>s_flt, 1, 0)

# #linear sliding, ramped beta:
p_W = 1.027 * jnp.maximum(0, h-s)
p_I = 0.917 * h
#phi = 1 - (p_W / p_I)
phi = 1
beta = 300 * phi
beta = jnp.where(s_gnd>s_flt, beta, 0)
# beta = beta.at[0].set(100)

#linear slding, artificial ramp:
# beta = jnp.where(s_gnd>s_flt, jnp.maximum(0, 1-2.2*jnp.linspace(0,1,n+1)), 0)

base = s - h

effective_base = s.copy()
effective_base = effective_base.at[:].set(s -h*mu/mu_base)

epsilon = 1e-10




#plotgeom(h)


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
  









