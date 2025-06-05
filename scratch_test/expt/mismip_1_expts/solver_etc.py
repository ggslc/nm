import jax
import jax.numpy as jnp
from jax import jacfwd, jacrev
import jax.scipy.linalg as lalg
from jax.scipy.optimize import minimize

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


np.set_printoptions(precision=10, suppress=False, linewidth=np.inf, threshold=np.inf)

def make_nonlinear_momentum_residual():
    #NOTE: taken out the nonlinearity

    def mom_res(u, h):
        s_gnd = h + b
        s_flt = h*(1-0.917/1.027)
        s = jnp.maximum(s_gnd, s_flt)

        dudx = jnp.zeros((n+1,))
        dudx = dudx.at[1:n].set((u[1:n] - u[:n-1])/dx)
        #dudx = dudx.at[-2].set(dudx[-3])
        dudx = dudx.at[-1].set(0)
        ##set (or 'use' I guess) reflection boundary condition
        dudx = dudx.at[0].set(2*u[0]/dx)
       

        sliding = C * (u**(1/3)) * dx
        #making sure the Jacobian is full rank!
        sliding = sliding.at[:].set(jnp.where(h>0, jnp.where(s_gnd>s_flt, sliding, 0), u * dx))

        
        h_face = jnp.zeros((n+1,))
        h_face = h_face.at[1:n].set(0.5 * (h[1:n] + h[:n-1]))
        h_face = h_face.at[-1].set(0)
        h_face = h_face.at[0].set(h[0])


        mu_face_nl =  B * (jnp.abs(dudx)+epsilon_visc)**(-2/3)
        #mu_face_nl = mu_face.copy()


        flux = mu_face_nl * h_face * dudx


        h_grad_s = jnp.zeros((n,))
        h_grad_s = h_grad_s.at[1:n-1].set(rho * g * h[1:n-1] * 0.5 * (s[2:n] - s[:n-2]))
        h_grad_s = h_grad_s.at[-1].set(-rho * g * h[-1] * 0.5 * s[-2])
        #h_grad_s = h_grad_s.at[-2].set(-0.1)
        h_grad_s = h_grad_s.at[0].set(rho * g * h[0] * 0.5 * (s[1] - s[0]))
      
        print(flux)
        print(sliding)
        print(h_grad_s)

        
        return flux[1:] - flux[:-1] - h_grad_s - sliding

    return mom_res

def make_linear_momentum_residual(beta):

    def mom_res(u, h, mu_face):
        s_gnd = h + b
        s_flt = h*(1-0.917/1.027)
        s = jnp.maximum(s_gnd, s_flt)
        s = s.at[-1].set(0)

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
      
        return flux[1:] - flux[:-1] - h_grad_s - sliding

    return mom_res


def make_picard_iterator_for_u(mu_centres_zero, h, beta, iterations):

    mu_faces_zero = jnp.zeros((n+1,))
    mu_faces_zero = mu_faces_zero.at[1:n].set(0.5 * (mu_centres_zero[:n-1] + mu_centres_zero[1:n]))
    mu_faces_zero = mu_faces_zero.at[0].set(mu_centres_zero[1])
   
    def new_mu(u):
    
        dudx = jnp.zeros((n+1,))
        dudx = dudx.at[1:-1].set((u[1:] - u[:-1])/dx)
        #set reflection boundary condition
        dudx = dudx.at[0].set(2*u[0]/dx)
    
        mu_nl = 5e-2 * mu_faces_zero * (jnp.abs(dudx)+epsilon)**(-2/3)
    
        return mu_nl
    

    def iterator(u_init):    

        mom_res = make_linear_momentum_residual(beta)

        mom_res_nl = make_nonlinear_momentum_residual(beta, mu_centres_zero)
        
        jac_mom_res_fn = jacfwd(mom_res, argnums=0)


        mu = mu_faces_zero.copy()
        u = u_init.copy()

        prev_residual = 1

        us = [u]

        for i in range(iterations):
            
            #residual = jnp.max(jnp.abs(mom_res_nl(u, h, mu))) 
            residual = jnp.max(jnp.abs(mom_res_nl(u, h)))
            print(residual)

            mu = new_mu(u) 
            
            jac_mom_res = jac_mom_res_fn(u, h, mu) 

            u = u + lalg.solve(jac_mom_res, -mom_res(u, h, mu)) 

            us.append(u)
            
            #if (residual < 1e-17) or (jnp.abs(residual-prev_residual) < 1e-18):
            #    break

            prev_residual = residual

        #residual = jnp.max(jnp.abs(mom_res(u, h, mu)))
        #print(residual)

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



def make_adv_operator(dt):
    
    def adv_op(u, h, h_old):

        h_face = jnp.zeros((n+1,))
        h_face = h_face.at[1:n+1].set(h[:n]) #upwind values
        h_face = h_face.at[0].set(h[0].copy())

        u_face = jnp.zeros((n+1,))
        u_face = u_face.at[1:n].set(0.5*(u[1:n]+u[:n-1]))
        u_face = u_face.at[-1].set(u[-1])

        h_flux = h_face * u_face
        #the two lines below were supposed to stop everything piling up in the last
        #cell but they had the opposite effect for some reason...
        #h_flux = h_flux.at[-2].set(h_flux[-3]) #stop everythin piling up at the end.
        #NOTE: This changes things a lot:
        #h_flux = h_flux.at[-1].set(h_flux[-2].copy())
        h_flux = h_flux.at[0].set(0)


        #return  (h - h_old)/dt - ( h_flux[1:(n+1)] - h_flux[:n] )/dx - acc
        #I think the above is a sign error in the flux!
        return  (h - h_old)/dt + ( h_flux[1:(n+1)] - h_flux[:n] )/dx - accumulation

    return adv_op    



def make_adv_rhs(accumulation):
    
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
        #h_flux = h_flux.at[-2].set(h_flux[-3]) #stop everythin piling up at the end.
        #h_flux = h_flux.at[-1].set(h_flux[-2])
        h_flux = h_flux.at[0].set(0)

        
        return  - ( h_flux[1:(n+1)] - h_flux[:n] )/dx + acc
    
    return adv 

def construct_tangent_propagator(dHdh, dHdu, dGdh, dGdu):
    #Naive approach
    dGdu_inv = jnp.linalg.inv(dGdu)
    int_ = dGdu_inv @ dGdh
    
    feedback_term = -dHdu @ int_
    
    #L = (-dHdu @ int_ + dHdh)
    #print(L-dHdh)
    
    return feedback_term + dHdh, feedback_term, dHdh


def implicit_coupled_solver_compiled(mu, beta, \
                            accumulation, dt, \
                            num_iterations, num_timesteps, \
                            compute_eigenvalues=False):


    mom_res = make_nonlinear_momentum_residual(beta, mu)
    adv = make_adv_operator(dt, accumulation)
    #adv = make_adv_operator_acc_dependent_on_old_h(dt, accumulation)

    visc_jac_fn = jacfwd(mom_res, argnums=(0,1))
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
                                      )
    
            rhs = jnp.concatenate((-mom_res(u, h), -adv(u, h, h_old, bmr)))
    
            dvar = lalg.solve(full_jacobian, rhs)
    
            u = u.at[:].set(u+dvar[:n])
            h = h.at[:].set(h+dvar[n:])
    
            #print(jnp.max(jnp.abs(mom_res(u, h))), jnp.max(jnp.abs(adv(u, h, h_old, basal_melt_rate))))
    
            return u, h, i+1
        return newton_iterate


    def timestep(state):
        u, h, bmr, us, hs, bmrs, t = state

        #jax.debug.print("t = {}", t)

        bmr_new = bmr.copy() #could make this some time-dependent function y'see.

        newton_iterate = make_newton_iterate(bmr_new, h.copy())

        initial_state = (u, h, 0)
        u_new, h_new, i = jax.lax.while_loop(newton_condition, newton_iterate, initial_state)

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

        return u, h, us, hs, bmrs

    return iterator



def implicit_coupled_solver(dt, num_iterations, num_timesteps):

    def newton_solve(u_trial, h_trial):

        mom_res = make_nonlinear_momentum_residual()
        adv = make_adv_operator(dt)
        #adv = make_adv_operator_acc_dependent_on_old_h(dt, accumulation)


        ##for debugging:
        #mom_res(u_trial, h_trial)
        #raise


        mom_res_jac_fn = jacfwd(mom_res, argnums=(0,1))
        adv_jac_fn = jacfwd(adv, argnums=(0,1))

        u = u_trial.copy()
        h = h_trial.copy()
        h_old = h_trial.copy()

        hs = []
        us = []
        
        for j in range(num_timesteps):

            print(j)
            for i in range(num_iterations):
                mom_res_jac = mom_res_jac_fn(u, h)
                adv_jac = adv_jac_fn(u, h, h_old)
        
                full_jacobian = jnp.block(
                                          [[mom_res_jac[0], mom_res_jac[1]],
                                          [adv_jac[0], adv_jac[1]]]
                                          )

         #       print(np.array(mom_res_jac[0]))
         #       print("-------------------")
         #       print("-------------------")
         #       print("-------------------")
         #       print(np.array(adv_jac[0]))
         #       print("-------------------")
         #       print("-------------------")
         #       print("-------------------")
         #       print(np.array(mom_res_jac[1]))
         #       print("-------------------")
         #       print("-------------------")
         #       print("-------------------")
         #       print(np.array(adv_jac[1]))
         #       print("-------------------")
         #       print("-------------------")
         #       print("-------------------")
         #       print(np.array(full_jacobian))
         #       raise


                rhs = jnp.concatenate((-mom_res(u, h), -adv(u, h, h_old)))

                dvar = lalg.solve(full_jacobian, rhs)


                u = u.at[:].set(u+dvar[:n])
                h = h.at[:].set(h+dvar[n:])

                print(jnp.max(jnp.abs(mom_res(u, h))), jnp.max(jnp.abs(adv(u, h, h_old))))


        return u, h, hs, us

    return newton_solve


def make_u_solver(mu, h, beta, num_iterations):

    def newton_solve(u_trial):

        mom_res = make_nonlinear_momentum_residual(beta, mu)
        #mom_res = make_nonlinear_momentum_residual_dfct(beta, mu)
        mom_res_jac_fn = jacfwd(mom_res, argnums=0)

        u = u_trial.copy()

        us = []
        
        ##For debugging:
        #mom_res(u, h)
        #raise

        for i in range(num_iterations):
            mom_res_jac = mom_res_jac_fn(u, h)
        #    print(mom_res_jac)
            rhs = -mom_res(u, h)
        #    print(rhs)
            #raise

            du = lalg.solve(mom_res_jac, rhs)

            #print(mom_res_jac @ du - rhs)

            #u = u.at[:].set(u+du)
            u = u+du
            
            print(jnp.max(jnp.abs(mom_res(u, h))))


        #plotboth(h, u)

        us.append(u)

        final_mom_res_jac = mom_res_jac_fn(u, h)

        return u, us

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






n = 11
x = jnp.linspace(0, 1_800_000, n)
dx = x[1]-x[0]

rho = 900
rho_w = 1000

g = 9.8

accumulation = jnp.zeros_like(x)+0.3/(3.15e7)


C = 7.624e6

A = 4.6146e-24

B = 2 * (A**(-1/3))

epsilon_visc = 1e-5/(3.15e7)


b = 720 - 778.5*x/750_000


h_init = jnp.zeros_like(x)+100


u_trial = jnp.zeros_like(x)+0.0001
h_trial = h_init.copy()




solver = implicit_coupled_solver(100_000, 10, 1)
u, h, us, hs = solver(u_trial, h_trial)






