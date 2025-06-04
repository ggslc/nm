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
        accumulation = accumulation.at[:].set(jnp.where(h>0, accumulation, 0))

        h_face = jnp.zeros((n+1,))
        h_face = h_face.at[1:n].set(h_old[:n-1]) #upwind values

        u_face = jnp.zeros((n+1,))
        u_face = u_face.at[1:n-1].set(0.5*(u[1:n-1]+u[:n-2]))
        u_face = u_face.at[-2].set(u[-2])

        h_flux = h_face * u_face
        h_flux = h_flux.at[-2].set(h_flux[-3]) #stop everythin piling up at the end.
        #h_flux = h_flux.at[-1].set(h_flux[-2])
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
        acc = acc.at[:].set(jnp.where(h>0, acc, 0))

        h_face = jnp.zeros((n+1,))
        h_face = h_face.at[1:n].set(h[:n-1]) #upwind values
        h_face = h_face.at[0].set(h[0])

        u_face = jnp.zeros((n+1,))
        u_face = u_face.at[1:n-1].set(0.5*(u[1:n-1]+u[:n-2]))
        u_face = u_face.at[-2].set(u[-2])

        h_flux = h_face * u_face
        #the two lines below were supposed to stop everything piling up in the last
        #cell but they had the opposite effect for some reason...
        h_flux = h_flux.at[-2].set(h_flux[-3]) #stop everythin piling up at the end.
        h_flux = h_flux.at[-1].set(h_flux[-2])
        h_flux = h_flux.at[0].set(0)


        #return  (h - h_old)/dt - ( h_flux[1:(n+1)] - h_flux[:n] )/dx - acc
        #I think the above is a sign error in the flux!
        return  (h - h_old)/dt + ( h_flux[1:(n+1)] - h_flux[:n] )/dx - acc

    return adv_op    


def make_adv_rhs(dt, accumulation):
    
    def adv(u, h, basal_melt_rate):
        s_gnd = h + b
        s_flt = h*(1-0.917/1.027)

        acc = jnp.where(s_gnd<s_flt, accumulation-basal_melt_rate, accumulation)
        acc = acc.at[-1].set(0)
        acc = acc.at[:].set(jnp.where(h>0, acc, 0))

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

        
        return  - ( h_flux[1:(n+1)] - h_flux[:n] )/dx + acc
    
    return adv 


def construct_tangent_propagator(dHdh, dHdu, dGdh, dGdu):
    #Naive approach
    dGdu_inv = jnp.linalg.inv(dGdu)
    int_ = dGdu_inv @ dGdh

    #L = (-dHdu @ int_ + dHdh)
    #print(L-dHdh)
    
    return (-dHdu @ int_ + dHdh)


def implicit_coupled_solver_compiled(mu, beta, \
                            accumulation, dt, \
                            num_iterations, num_timesteps, \
                            compute_eigenvalues=False):


    vto = make_vto_nl(beta, mu)
    adv = make_adv_operator(dt, accumulation)

    visc_jac_fn = jacfwd(vto, argnums=(0,1))
    adv_jac_fn = jacfwd(adv, argnums=(0,1))


    def timestep_condition(state):
        t = state[-1]
        return t<num_timesteps


    def newton_condition(state):
        i = state[-1]
        return i<num_iterations


    def make_newton_iterate(bmr, h_old):
        def newton_iterate(state):
            u, h, i = state
            
            visc_jac = visc_jac_fn(u, h)
            adv_jac = adv_jac_fn(u, h, h_old, bmr)
    
            full_jacobian = jnp.block(
                                      [ [visc_jac[0], visc_jac[1]],
                                        [adv_jac[0] , adv_jac[1]] ]
                                      )[:2*n-1, :2*n-1]
    
            rhs = jnp.concatenate((-vto(u, h), -adv(u, h, h_old, bmr)[:n-1]))
    
            dvar = lalg.solve(full_jacobian, rhs)
    
            u = u.at[:].set(u+dvar[:n])
            h = h.at[:-1].set(h[:-1]+dvar[n:2*n-1])
    
            #print(jnp.max(jnp.abs(vto(u, h))), jnp.max(jnp.abs(adv(u, h, h_old, basal_melt_rate))))
    
            return u, h, i+1
        return newton_iterate


    def timestep(state):
        u, h, bmr, us, hs, bmrs, t = state

        jax.debug.print("t = {}", t)

        bmr_new = bmr.copy() #could make this some time-dependent function y'see.

        newton_iterate = make_newton_iterate(bmr_new, h.copy())

        initial_state = (u, h, 0)
        u_new, h_new, i = jax.lax.while_loop(newton_condition, newton_iterate, initial_state)

        residual = (jnp.max(vto(u_new, h_new)), jnp.max(adv(u_new, h_new, h, bmr)))
        jax.debug.print("residual = {}", residual)

        us = us.at[t].set(u_new)
        hs = hs.at[t].set(h_new)
        bmrs = bmrs.at[t].set(bmr_new)

        return u_new, h_new, bmr_new, us, hs, bmrs, t+1
        

    @jax.jit
    def iterator(u_init, h_init, bmr_init):

        #have to pre-allocate these things because we can't use python-side
        #mutations like appending to lists in a lax while_loop!
        us = jnp.zeros((num_timesteps, n))
        hs = jnp.zeros((num_timesteps, n))
        bmrs = jnp.zeros((num_timesteps, n))

        initial_state = (u_init, h_init, bmr_init, us, hs, bmrs, 0)

        u, h, bmr, us, hs, bmrs, t = jax.lax.while_loop(timestep_condition, timestep, initial_state)


        if compute_eigenvalues:
            
            H = make_adv_rhs(dt, accumulation)
            H_jac = jacfwd(H, argnums=(0,1))(u, h, bmr)

            visc_jac = visc_jac_fn(u, h)

            L = construct_tangent_propagator(H_jac[1][:n-1, :n-1],\
                                             H_jac[0][:n-1,:],\
                                             visc_jac[1][:,:n-1],\
                                             visc_jac[0])

            #evals, evecs = jnp.linalg.eig(L)
            evecs, evals, _ = jnp.linalg.svd(L) #I know they're not eigen-...!
            
            #evecs_ptl, evals_ptl, _ = jnp.linalg.svd(H_jac[1][:n-1, :n-1])

            indices = jnp.argsort(evals)
            evals_ordered = evals[indices]
            evecs_ordered = evecs[:,indices]


            #indices_ptl = jnp.argsort(evals_ptl)
            #evals_ordered_ptl = evals[indices_ptl]
            #evecs_ordered_ptl = evecs[:,indices_ptl]

            return u, h, us, hs, bmrs, evals_ordered, evecs_ordered
        else:
            return u, h, us, hs, bmrs

    return iterator



def implicit_coupled_solver(u_trial, h_trial,\
                            accumulation, dt, \
                            num_iterations, num_timesteps, \
                            compute_evals=False):

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
        largest_evals = []
        all_evals = []
        largest_evals_ptl = []
        all_evals_ptl = []

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
            
            if compute_evals:
                H = make_adv_rhs(dt, accumulation)
                H_jac = jacfwd(H, argnums=(0,1))(u, h, basal_melt_rate)

                L = construct_tangent_propagator(H_jac[1][:n-1, :n-1],\
                                                 H_jac[0][:n-1,:],\
                                                 vto_jac[1][:,:n-1],\
                                                 vto_jac[0])

                #evals, evecs = jnp.linalg.eig(L)
                evecs, evals, _ = jnp.linalg.svd(L) #I know they're not eigen-...!
                
                evecs_ptl, evals_ptl, _ = jnp.linalg.svd(H_jac[1][:n-1, :n-1])


           
                indices = jnp.argsort(evals)
                evals_ordered = evals[indices]
                evecs_ordered = evecs[:,indices]


                indices_ptl = jnp.argsort(evals_ptl)
                evals_ordered_ptl = evals[indices_ptl]
                evecs_ordered_ptl = evecs[:,indices_ptl]


                #plt.imshow(jnp.real(evecs_ordered), vmin=-0.2, vmax=0.2)
                #plt.show()

                all_evals.append(evals)
                largest_evals.append(evals_ordered[-1])
                
                all_evals_ptl.append(evals_ptl)
                largest_evals_ptl.append(evals_ordered_ptl[-1])

            hs.append(h)
            us.append(u)


            h_old = h.copy()


        return u, h, hs, us, largest_evals, jnp.array(all_evals), largest_evals_ptl, jnp.array(all_evals_ptl)

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

    if isinstance(us, (jnp.ndarray, np.ndarray)):
        us_list = [np.array(u) for u in us]
    if isinstance(hs, (jnp.ndarray, np.ndarray)):
        hs_list = [np.array(h) for h in hs]

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
    cbar.set_label('Timestep')

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

h = 1.2*jnp.exp(-2*x*x*x) #grounded sections
#h = jnp.ones_like(x)*0.5 #floating uniform thk

#h = 1.75*jnp.exp(-2*x*x) #just a bit of an odd shaped ice shelf
#h = (1+jnp.zeros((n,)))*(1-x/2)
h = h.at[-1].set(0)


b_intermediate = jnp.zeros((n,))-0.5

s_gnd = b_intermediate + h
s_flt = h*(1-0.917/1.027)
s = jnp.maximum(s_gnd, s_flt)
s = s.at[-1].set(0)


#nice overdeepening on a flat bed:
b = jnp.zeros((n,))-0.5
b = b.at[:n].set(b[:n] - 0.15*jnp.exp(-(5*x-2)**2))

##not so flat bed:
#b = -0.1 - 0.4*x
#b = b.at[:n].set(b[:n] - 0.25*jnp.exp(-(5*x-2)**2))

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



#accumulation = jnp.zeros_like(h)+0.05
accumulation = jnp.zeros_like(h)

#plotgeom(h)
#raise


#u_trial = jnp.exp(x)-1
#u_trial = x.copy()
u_trial = jnp.zeros_like(x)
h_trial = h.copy()








###########TRYING TO GET THE JIT COMPILATION GOING:

n_timesteps = 5
n_iterations = 30
timestep = 0.02

bmr_init = jnp.zeros_like(accumulation)#+0.085

solve_and_evolve = implicit_coupled_solver_compiled(mu, beta, accumulation, timestep, n_iterations, n_timesteps, compute_eigenvalues=False)
u_end, h_end, us, hs, bmrs, evals, evecs = solve_and_evolve(u_trial, h_trial, bmr_init)

print(evals)

plotboths(hs[:, :], us[:, :], n_timesteps)

raise

#Plotting evolution of largest eigenvalues for steady-state solutions:
largest_evals = []
for bmr_init in [jnp.zeros_like(accumulation)+0.05*i for i in range(10)]:

    solve_and_evolve = implicit_coupled_solver_compiled(mu, beta, accumulation, timestep, n_iterations, n_timesteps, compute_eigenvalues=True)
    u_end, h_end, us, hs, bmrs, evals, evecs = solve_and_evolve(u_trial, h_trial, bmr_init)

    largest_evals.append(evals[-1])

    plotboths(hs[::10, :], us[::10, :], n_timesteps)

plt.plot(largest_evals)
plt.show()





raise
#######WORKING STUFF LOOKING AT EIGENVALUES, NO JIT COMPILATION SO IT'S SLOW:


n_timesteps = 5
newton_solve = implicit_coupled_solver(u_trial, h_trial, accumulation, 1, 50, n_timesteps, compute_evals=True)
u_end, h_end, hs, us, evals, all_evals, evals_ptl, all_evals_ptl = newton_solve(mu)


#plt.imshow(jnp.transpose(all_evals), vmin=-10, vmax=10, cmap="RdBu_r")
#plt.show()
#
#plt.imshow(jnp.transpose(all_evals_ptl), vmin=-10, vmax=10, cmap="RdBu_r")
#plt.show()

#plt.imshow(jnp.transpose(all_evals)-jnp.transpose(all_evals_ptl), vmin=-3, vmax=3, cmap="RdBu_r")
#plt.colorbar()
#plt.show()

#plt.plot(evals)
#plt.show()


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




