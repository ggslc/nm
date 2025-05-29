import jax
import jax.numpy as jnp
from jax import jacfwd, jacrev
import jax.scipy.linalg as lalg
from jax.scipy.optimize import minimize

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


np.set_printoptions(precision=1, suppress=False, linewidth=np.inf)

def make_vto_nl(beta, mu_cc):
    #NOTE: taken out the nonlinearity

    mu_face = jnp.zeros((n+1,))
    mu_face = mu_face.at[1:-1].set(0.5 * (mu_cc[:-1] + mu_cc[1:]))
    mu_face = mu_face.at[-2].set(mu_face[-3])
    mu_face = mu_face.at[0].set(mu_cc[1])

    
    def vto(u, h):
        s_gnd = h + b
        s_flt = h*(1-0.917/1.027)
        s = jnp.maximum(s_gnd, s_flt)
        s = s.at[-1].set(0)

        dudx = jnp.zeros((n+1,))
        dudx = dudx.at[1:n].set((u[1:n] - u[:n-1])/dx)
        #dudx = dudx.at[-2].set(dudx[-3])
        dudx = dudx.at[-2].set(0)
        ##set (or 'use' I guess) reflection boundary condition
        dudx = dudx.at[0].set(2*u[0]/dx)
       

        sliding = beta * u * dx
        #making sure the Jacobian is full rank!
        sliding = sliding.at[:].set(jnp.where(h>0, jnp.where(s_gnd>s_flt, sliding, 0), u * dx))

        
        h_face = jnp.zeros((n+1,))
        h_face = h_face.at[1:n-1].set(0.5 * (h[1:n-1] + h[:n-2]))
        h_face = h_face.at[-2].set(0)
        h_face = h_face.at[0].set(h[0])


        mu_face_nl = 5e-2 * mu_face * (jnp.abs(dudx)+epsilon)**(-2/3)
        #mu_face_nl = mu_face.copy()


        flux = h_face * mu_face_nl * dudx


        #flux = flux.at[-2].set(0)


        h_grad_s = jnp.zeros((n,))
        h_grad_s = h_grad_s.at[1:n-1].set(h[1:n-1] * 0.5 * (s[2:n] - s[:n-2]))
        #h_grad_s = h_grad_s.at[-2].set(-0.1)
        h_grad_s = h_grad_s.at[0].set(h[0] * 0.5 * (s[1] - s[0]))
      
        #print(flux)
        #print(sliding)
        #print(h_grad_s)
        
        return flux[1:] - flux[:-1] - h_grad_s - sliding

    return vto


def make_linear_vto(beta):
    
    def vto(u, h, mu_face):
        s_gnd = h + b
        s_flt = h*(1-0.917/1.027)
        s = jnp.maximum(s_gnd, s_flt)
        s = s.at[-1].set(0)

        dudx = jnp.zeros((n+1,))
        dudx = dudx.at[1:n].set((u[1:n] - u[:n-1])/dx)
        #dudx = dudx.at[-2].set(dudx[-3])
        dudx = dudx.at[-2].set(0)
        ##set (or 'use' I guess) reflection boundary condition
        dudx = dudx.at[0].set(2*u[0]/dx)
       

        sliding = beta * u * dx
        #making sure the Jacobian is full rank!
        sliding = sliding.at[:].set(jnp.where(h>0, jnp.where(s_gnd>s_flt, sliding, 0), u * dx))

        
        h_face = jnp.zeros((n+1,))
        h_face = h_face.at[1:n-1].set(0.5 * (h[1:n-1] + h[:n-2]))
        h_face = h_face.at[-2].set(0)
        h_face = h_face.at[0].set(h[0])


        flux = h_face * mu_face * dudx


        h_grad_s = jnp.zeros((n,))
        h_grad_s = h_grad_s.at[1:n-1].set(h[1:n-1] * 0.5 * (s[2:n] - s[:n-2]))
        #h_grad_s = h_grad_s.at[-2].set(-1)
        h_grad_s = h_grad_s.at[0].set(h[0] * 0.5 * (s[1] - s[0]))
      

        return flux[1:] - flux[:-1] - h_grad_s - sliding

    return vto


def make_picard_iterator_for_u(mu_centres_zero, h, beta, iterations):

    mu_faces_zero = jnp.zeros((n+1,))
    mu_faces_zero = mu_faces_zero.at[1:-2].set(0.5 * (mu_centres_zero[:-2] + mu_centres_zero[1:-1]))
    mu_faces_zero = mu_faces_zero.at[0].set(mu_centres_zero[1])
   
    def new_mu(u):
    
        dudx = jnp.zeros((n+1,))
        dudx = dudx.at[1:-1].set((u[1:] - u[:-1])/dx)
        #set reflection boundary condition
        dudx = dudx.at[0].set(2*u[0]/dx)
    
        mu_nl = mu_faces_zero * (jnp.abs(dudx)+epsilon)**(-2/3)
    
        return mu_nl
    

    def iterator(u_init):    

        vto = make_linear_vto(beta)

        vto_nl = make_vto_nl(beta, mu_centres_zero)
        
        jac_vto_fn = jacfwd(vto, argnums=0)


        mu = mu_faces_zero.copy()
        u = u_init.copy()

        prev_residual = 1

        us = [u]

        for i in range(iterations):
            
            residual = jnp.max(jnp.abs(vto(u, h, mu))) 
            print(residual)

            mu = new_mu(u) 
            
            jac_vto = jac_vto_fn(u, h, mu) 

            u = u + lalg.solve(jac_vto, -vto(u, h, mu)) 

            us.append(u)
            
            #if (residual < 1e-17) or (jnp.abs(residual-prev_residual) < 1e-18):
            #    break

            prev_residual = residual

        residual = jnp.max(jnp.abs(vto(u, h, mu)))
        print(residual)

        return u, us
       
    return iterator



def make_adv_expt(dt, mu):

    def adv(u, h_old, accumulation, basal_melt_rate):

        
        dudx = jnp.zeros((n,))
        dudx = dudx.at[1:n-2].set((u[2:n-1] - u[:n-3])/dx)
        #leaving the last two to be zero
        dudx = dudx.at[0].set((u[1] - u[0])/dx)
        #the last one doesn't matter I don't think
        

        accumulation = accumulation.at[-1].set(0)
        accumulation = accumulation.at[:].set(np.where(h>0, accumulation, 0))

        h_face = jnp.zeros((n+1,))
        h_face = h_face.at[1:n].set(h_old[:n-1]) #upwind values

        u_face = jnp.zeros((n+1,))
        u_face = u_face.at[1:n-1].set(0.5*(u[1:n-1]+u[:n-2]))
        u_face = u_face.at[-2].set(u[-2])

        h_flux = h_face * u_face
        h_flux = h_flux.at[-2].set(h_flux[-3]) #stop everythin piling up at the end.
        h_flux = h_flux.at[-1].set(h_flux[-2])
        h_flux = h_flux.at[0].set(0)


        h_new = h_old - (dt / dx) * ( h_flux[1:(n+1)] - h_flux[:n] )

        return h_new
        
    return adv



def make_adv_operator(dt, accumulation):
    
    def adv_op(u, h, h_old, basal_melt_rate):
        s_gnd = h + b
        s_flt = h*(1-0.917/1.027)

        acc = jnp.where(s_gnd<s_flt, accumulation-basal_melt_rate, accumulation)
        acc = acc.at[-1].set(0)
        acc = acc.at[:].set(np.where(h>0, acc, 0))

        h_face = jnp.zeros((n+1,))
        h_face = h_face.at[1:n].set(h[:n-1]) #upwind values
        h_face = h_face.at[0].set(h[0])

        u_face = jnp.zeros((n+1,))
        u_face = u_face.at[1:n-1].set(0.5*(u[1:n-1]+u[:n-2]))
        u_face = u_face.at[-2].set(u[-2])

        h_flux = h_face * u_face
        h_flux = h_flux.at[-2].set(h_flux[-3]) #stop everythin piling up at the end.
        h_flux = h_flux.at[-1].set(h_flux[-2])
        h_flux = h_flux.at[0].set(0)


        #return  (h - h_old)/dt - ( h_flux[1:(n+1)] - h_flux[:n] )/dx - acc
        #I think the above is a sign error in the flux!
        return  (h - h_old)/dt + ( h_flux[1:(n+1)] - h_flux[:n] )/dx - acc

    return adv_op    



def implicit_coupled_solver(u_trial, h_trial, accumulation, dt, num_iterations, num_timesteps):

    def newton_solve(mu, initial_basal_melt_rate=0):

        vto = make_vto_nl(beta, mu)
        adv = make_adv_operator(dt, accumulation)

        vto_jac_fn = jacfwd(vto, argnums=(0,1))
        adv_jac_fn = jacfwd(adv, argnums=(0,1))

        u = u_trial.copy()
        h = h_trial.copy()
        h_old = h_trial.copy()

        hs = []
        us = []

        for j in range(num_timesteps):
            basal_melt_rate = initial_basal_melt_rate

            print(j)
            for i in range(num_iterations):
                vto_jac = vto_jac_fn(u, h)
                adv_jac = adv_jac_fn(u, h, h_old, basal_melt_rate)
        

                full_jacobian = jnp.block(
                                          [[vto_jac[0], vto_jac[1]],
                                          [adv_jac[0], adv_jac[1]]]
                                          )[:2*n-1, :2*n-1]

                #print(np.array(vto_jac[0]))
                #print("-------------------")
                #print("-------------------")
                #print("-------------------")
                #print(np.array(adv_jac[0]))
                #print("-------------------")
                #print("-------------------")
                #print("-------------------")
                #print(np.array(full_jacobian))
                #raise


                rhs = jnp.concatenate((-vto(u, h), -adv(u, h, h_old, basal_melt_rate)[:n-1]))

                dvar = lalg.solve(full_jacobian, rhs)


                u = u.at[:].set(u+dvar[:n])
                h = h.at[:-1].set(h[:-1]+dvar[n:2*n-1])

                print(jnp.max(jnp.abs(vto(u, h))), jnp.max(jnp.abs(adv(u, h, h_old, basal_melt_rate))))


            #plotboth(h, u)


            hs.append(h)
            us.append(u)


            h_old = h.copy()


        return u, h, hs, us

    return newton_solve


def make_u_solver(h, beta, num_iterations):

    def newton_solve(u_trial, mu):

        vto = make_vto_nl(beta, mu)
        #vto = make_vto_nl_dfct(beta, mu)
        vto_jac_fn = jacfwd(vto, argnums=0)

        u = u_trial.copy()

        us = []
        
        ##For debugging:
        #vto(u, h)
        #raise

        for i in range(num_iterations):
            vto_jac = vto_jac_fn(u, h)
        #    print(vto_jac)
            rhs = -vto(u, h)
        #    print(rhs)
            #raise

            du = lalg.solve(vto_jac, rhs)

            #print(vto_jac @ du - rhs)

            u = u.at[:].set(u+du)
            
            print(jnp.max(jnp.abs(vto(u, h))))


        plotboth(h, u)

        us.append(u)

        final_vto_jac = vto_jac_fn(u, h)

        return u, us, final_vto_jac

    return newton_solve


def plotgeom(thk):

    s_gnd = b + thk
    s_flt = thk*(1-0.917/1.027)
    s = jnp.maximum(s_gnd, s_flt)

    base = s-thk

    #plot b, s and base on lhs y axis, and C on rhs y axis
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






rho = 1
g = 1

lx = 1
n = 100
dx = lx/(n-1)
x = jnp.linspace(0,lx,n)

mu = jnp.zeros((n,)) + 1


#OVERDEEPENED BED

h = jnp.exp(-2*x*x*x) #grounded sections
#h = jnp.ones_like(x)*0.5 #floating uniform thk

#h = 1.75*jnp.exp(-2*x*x) #just a bit of an odd shaped ice shelf
#h = (1+jnp.zeros((n,)))*(1-x/2)
h = h.at[-1].set(0)


b_intermediate = jnp.zeros((n,))-0.5

s_gnd = b_intermediate + h
s_flt = h*(1-0.917/1.027)
s = jnp.maximum(s_gnd, s_flt)
s = s.at[-1].set(0)


b = jnp.zeros((n,))-0.5

h = jnp.minimum(s-b, s/(1-0.917/1.027))
h = h.at[-1].set(0)


# #linear sliding, constant beta:
# beta = jnp.where(s_gnd>s_flt, 1, 0)

p_W = 1.027 * jnp.maximum(0, h-s)
p_I = 0.917 * h
phi = 1 - (p_W / p_I)
#phi = 1
#beta = 300 * phi
#beta = phi
beta = 100
beta = jnp.where(s_gnd>s_flt, beta, 0)
#print(beta)
#raise

base = s - h

epsilon = 1e-5



accumulation = jnp.zeros_like(h)

#plotgeom(h)
#raise


#u_trial = jnp.exp(x)-1
#u_trial = x.copy()
u_trial = jnp.zeros_like(x)
h_trial = h.copy()

n_timesteps = 5
newton_solve = implicit_coupled_solver(u_trial, h_trial, accumulation, 1, 20, n_timesteps)
u_end, h_end, hs, us = newton_solve(mu)
plotboths(hs[:], us[:], n_timesteps) 


##NEWTON
#newton_solve = make_u_solver(h, beta, 20)
#u_end, us, vto_jac = newton_solve(u_trial, mu)


#evals, evecs = jnp.linalg.eig(vto_jac)

#indices = jnp.argsort(evals)
#evals_ordered = evals[indices]
#evecs_ordered = evecs[:,indices]

#plt.imshow(jnp.real(evecs_ordered), vmin=-0.2, vmax=0.2)
#plt.show()

raise



print(evals_ordered)

plt.figure(figsize=(10,5))
for ev in evecs_ordered[:,-5:]:
    plt.plot(ev)
plt.show()

##PICARD:
#iterator = make_picard_iterator_for_u(mu, h, beta, 30)
#u_end, us = iterator(u_trial)

#plotboth(h, u_end)

#plt.figure(figsize=(10,5))
#for i, u_ in enumerate(us):
#    plt.plot(u_, label=str(i))
#plt.legend()
#plt.show()




