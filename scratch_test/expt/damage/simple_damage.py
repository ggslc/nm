
import jax
import jax.numpy as jnp
from jax import jacfwd, jacrev
import jax.scipy.linalg as lalg
from jax.scipy.optimize import minimize
from jax.lax import while_loop

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


np.set_printoptions(precision=2, suppress=False, linewidth=np.inf)


def make_vto_nl(h, beta, mu_cc):

    mu_face = jnp.zeros((n+1,))
    mu_face = mu_face.at[1:-2].set(0.5 * (mu_cc[:-2] + mu_cc[1:-1]))
    mu_face = mu_face.at[0].set(mu_cc[1])

    s_gnd = h + b
    s_flt = h*(1-0.917/1.027)
    s = jnp.maximum(s_gnd, s_flt)
    s = s.at[-1].set(0)


    def vto(u, d):

        d_face = jnp.zeros((n+1,))
        d_face = d_face.at[1:-2].set(0.5 * (d[1:-1] + d[:-2]))
        d_face = d_face.at[-2].set(d_face[-3])
        d_face = d_face.at[0].set(d[0])


        dudx = jnp.zeros((n+1,))
        dudx = dudx.at[1:-2].set((u[1:-1] - u[:-2])/dx)
        #dudx = dudx.at[-2].set(dudx[-3]) #commented out means leave as zero
        #set reflection boundary condition
        dudx = dudx.at[0].set(2*u[0]/dx)
        
        sliding = beta * u * dx

        sliding = sliding.at[:].set(jnp.where(h>0, jnp.where(s_gnd>s_flt, sliding, 0), u * dx))

        
        h_face = jnp.zeros((n+1,))
        h_face = h_face.at[1:-2].set(0.5 * (h[1:-1] + h[:-2]))
        #h_face = h_face.at[-1].set(h[-1])
        h_face = h_face.at[0].set(h[0])


        mu_face_nl = mu_face * (jnp.abs(dudx)+epsilon)**(-2/3)
        #print(mu_face)
        #print(mu_face_nl)
        #raise


        #d_face = d_face.at[-10].set(0.75)
        #print(d_face)

        flux = h_face * (1-d_face) * mu_face_nl * dudx


        h_grad_s = jnp.zeros((n,))
        #h_grad_s = h_grad_s.at[1:-2].set(0.917 * h[1:-2] * 0.5 * (s[2:-1] - s[:-3]))
        #h_grad_s = h_grad_s.at[-2].set(-0.917 * h[-2] * 0.5*(s[-3] + s[-2]))
        h_grad_s = h_grad_s.at[1:-1].set(0.917 * h[1:-1] * 0.5 * (s[2:] - s[:-2]))
        h_grad_s = h_grad_s.at[0].set(0.917 * h[0] * 0.5 * (s[1] - s[0]))
       
        #print(h_grad_s)
        #print(h_grad_s[n-1])
        #print(h_grad_s.shape)
        #print(h_grad_s[:n-1].shape)
        #raise


        #not sure what to do here... should the various bits have different weights?.
        #return flux[1:(n+1)] - flux[:n] - 10*h_grad_s - 100*sliding
        return flux[1:] - flux[:-1] - h_grad_s - sliding
        

    return vto


def make_linear_vto(h, beta):


    s_gnd = h + b
    s_flt = h*(1-0.917/1.027)
    s = jnp.maximum(s_gnd, s_flt)
    s = s.at[-1].set(0)


    def vto(u, d, mu_face):

        d_face = jnp.zeros((n+1,))
        d_face = d_face.at[1:-2].set(0.5 * (d[1:-1] + d[:-2]))
        d_face = d_face.at[-2].set(d_face[-3])
        d_face = d_face.at[0].set(d[0])


        dudx = jnp.zeros((n+1,))
        dudx = dudx.at[1:-2].set((u[1:-1] - u[:-2])/dx)
        #dudx = dudx.at[-2].set(dudx[-3]) #commented out means leave as zero
        #set reflection boundary condition
        dudx = dudx.at[0].set(2*u[0]/dx)
        
        sliding = beta * u * dx

        sliding = sliding.at[:].set(jnp.where(h>0, jnp.where(s_gnd>s_flt, sliding, 0), u * dx))

        
        h_face = jnp.zeros((n+1,))
        h_face = h_face.at[1:-2].set(0.5 * (h[1:-1] + h[:-2]))
        #h_face = h_face.at[-1].set(h[-1])
        h_face = h_face.at[0].set(h[0])


        flux = h_face * (1-d_face) * mu_face * dudx


        h_grad_s = jnp.zeros((n,))
        #h_grad_s = h_grad_s.at[1:-2].set(0.917 * h[1:-2] * 0.5 * (s[2:-1] - s[:-3]))
        #h_grad_s = h_grad_s.at[-2].set(-0.917 * h[-2] * 0.5*(s[-3] + s[-2]))
        h_grad_s = h_grad_s.at[1:-1].set(0.917 * h[1:-1] * 0.5 * (s[2:] - s[:-2]))
        h_grad_s = h_grad_s.at[0].set(0.917 * h[0] * 0.5 * (s[1] - s[0]))
       

        #not sure what to do here... should the various bits have different weights?.
        #return flux[1:(n+1)] - flux[:n] - 10*h_grad_s - 100*sliding
        return flux[1:] - flux[:-1] - h_grad_s - sliding
        

    return vto


def make_adv_expt(h, dt, gamma=1e5, A=1):

    def adv(u, d_old):

        hd_old = h*d_old
        
        dudx = jnp.zeros((n,))
        dudx = dudx.at[1:n-2].set((u[2:n-1] - u[:n-3])/dx)
        #leaving the last two to be zero
        dudx = dudx.at[0].set((u[1] - u[0])/dx)
        #the last one doesn't matter I don't think
        

        mu_nl = mu * (jnp.abs(dudx)+epsilon)**(-2/3)

        #tau_xx = 0.5 * (1-d) * mu_nl * dudx

        #source = 0.002*gamma * A * dx * ((0.5 * mu_nl * dudx - 0*rho * g * hd)**4) #this "works" well
        #source = 0.002*gamma * A * dx * ((0.5 * mu_nl * dudx - 0.01*rho * g * hd*(1-d))**4) #just messing about adding the (1-d) and changing rho etc.
        source = gamma * A * dx * ((0.5 * mu_nl * dudx - rho * g * hd_old)**4) #this reaches steady state nicely
        #there might be something about source term linearisation I've forgotten about... Patankar said somethibg...
        source = source.at[-1].set(0)
        #print(source)

        hd_face = jnp.zeros((n+1,))
        hd_face = hd_face.at[1:n].set(hd_old[:n-1]) #upwind values

        u_face = jnp.zeros((n+1,))
        u_face = u_face.at[1:n-1].set(0.5*(u[1:n-1]+u[:n-2]))
        u_face = u_face.at[-2].set(u[-2])

        hd_flux = hd_face * u_face
        hd_flux = hd_flux.at[-2].set(hd_flux[-3]) #stop everythin piling up at the end.
        hd_flux = hd_flux.at[-1].set(hd_flux[-2])
        hd_flux = hd_flux.at[0].set(0)


        d_new = d_old - (dt / dx) * ( hd_flux[1:(n+1)] - hd_flux[:n] )


        return d_new
        
    return adv


def make_adv(h, dt, gamma=1e5, A=1):

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

        #source = 0.002*gamma * A * dx * ((0.5 * mu_nl * dudx - 0*rho * g * hd)**4) #this "works" well
        #source = 0.002*gamma * A * dx * ((0.5 * mu_nl * dudx - 0.01*rho * g * hd*(1-d))**4) #just messing about adding the (1-d) and changing rho etc.
        source = gamma * A * dx * ((0.5 * mu_nl * dudx - rho * g * hd)**4) #this reaches steady state nicely
        #there might be something about source term linearisation I've forgotten about... Patankar said somethibg...
        source = source.at[-1].set(0)
        #print(source)

        hd_face = jnp.zeros((n+1,))
        hd_face = hd_face.at[1:n].set(hd[:n-1]) #upwind values

        u_face = jnp.zeros((n+1,))
        u_face = u_face.at[1:n-1].set(0.5*(u[1:n-1]+u[:n-2]))
        u_face = u_face.at[-2].set(u[-2])

        hd_flux = hd_face * u_face
        hd_flux = hd_flux.at[-2].set(hd_flux[-3]) #stop everythin piling up at the end.
        hd_flux = hd_flux.at[-1].set(hd_flux[-2])
        hd_flux = hd_flux.at[0].set(0)

        dhd_dt = (hd - hd_old) * dx / dt

        #return hd_flux[1:(n+1)] - hd_flux[:n] + dhd_dt - source #full
        return dhd_dt - source #no advection
        #return hd_flux[1:(n+1)] - hd_flux[:n] + dhd_dt #no source
        
    return adv



def make_u_solver(u_trial, num_iterations, d):

    def u_solve():

        vto = make_vto_nl(h, beta, mu)

        #For debugging only, run this
        #vto(u_trial, jnp.zeros_like(u_trial))
        #raise

        vto_jac_fn = jacfwd(vto, argnums=0)

        u = u_trial.copy()

        for i in range(num_iterations):
            vto_jac = vto_jac_fn(u, d)
            
            #print(vto_jac)
            #raise

            du = lalg.solve(vto_jac, -vto(u, d))

            u = u.at[:].set(u + du)


            print(jnp.max(jnp.abs(vto(u, d))))

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


def make_picard_iterator(mu_centres_zero, h, beta, d, iterations):

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

        vto = make_linear_vto(h, beta)
        jac_vto_fn = jacfwd(vto, argnums=0)


        mu = mu_faces_zero.copy()
        u = u_init.copy()

        prev_residual = 1

        for i in range(iterations):
            
            residual = jnp.max(jnp.abs(vto(u, d, mu)))
            print(residual)

            mu = new_mu(u)
            
            jac_vto = jac_vto_fn(u, d, mu)
            u = u + lalg.solve(jac_vto, -vto(u, d, mu))
            
            
            if (residual < 1e-7) or (jnp.abs(residual-prev_residual) < 1e-8):
                break

            prev_residual = residual

        residual = jnp.max(jnp.abs(vto(u, d, mu)))
        print(residual)

        return u
       
    return iterator



def make_picard_adv_only(u, mu_centres_zero, h, beta, dt, iterations):

    mu_faces_zero = jnp.zeros((n+1,))
    mu_faces_zero = mu_faces_zero.at[1:-2].set(0.5 * (mu_centres_zero[:-2] + mu_centres_zero[1:-1]))
    mu_faces_zero = mu_faces_zero.at[0].set(mu_centres_zero[1])
   

    def iterator(d_init):

        adv = make_adv(h, dt)
        jac_adv_fn = jacfwd(adv, argnums=1)

        d = d_init.copy()
        d_old = d_init.copy()

#        prev_residual_combo = 1

        for i in range(iterations):
            
#            residual1 = jnp.max(jnp.abs(vto(u, d, mu)))
#            residual2 = jnp.max(jnp.abs(adv(u, d, d_old)))
#            print(residual1, residual2)

            adv_vto = jac_adv_fn(u, d, d_old)
            d = d.at[:-1].set(d[:-1] + lalg.solve(adv_vto[:-1, :-1], -adv(u, d, d_old)[:-1]))
            d = jnp.minimum(d, 0.95)
            d = jnp.maximum(d, 0)

#TODO: MAKE THIS JIT COMPATIBLE
#            if ((residual1 < 1e-7) and (residual2 < 1e-7)) or\
#               (jnp.abs((residual1+residual2)-prev_residual_combo) < 1e-8):
#                break

#            prev_residual_combo = residual1+residual2
#
#        residual_1 = jnp.max(jnp.abs(vto(u, d, mu)))
#        residual_2 = jnp.max(jnp.abs(adv(u, d, d_old)))
#        print(residual1, residual2)

        return d
       
    return iterator
    



def make_picard_iterator_full(mu_centres_zero, h, beta, dt, iterations):

    mu_faces_zero = jnp.zeros((n+1,))
    mu_faces_zero = mu_faces_zero.at[1:-2].set(0.5 * (mu_centres_zero[:-2] + mu_centres_zero[1:-1]))
    mu_faces_zero = mu_faces_zero.at[0].set(mu_centres_zero[1])
   
    vto = make_linear_vto(h, beta)
    jac_vto_fn = jacfwd(vto, argnums=0)

    adv = make_adv(h, dt)
    jac_adv_fn = jacfwd(adv, argnums=1)



    def new_mu(u):
    
        dudx = jnp.zeros((n+1,))
        dudx = dudx.at[1:-1].set((u[1:] - u[:-1])/dx)
        #set reflection boundary condition
        dudx = dudx.at[0].set(2*u[0]/dx)
    
        mu_nl = mu_faces_zero * (jnp.abs(dudx)+epsilon)**(-2/3)
    
        return mu_nl
    
    ##for debugging:
    #jac_vto = jac_vto_fn(u_trial, d_trial, new_mu(u_trial))
    #jac_adv = jac_adv_fn(u_trial, d_trial, d_trial)
    #print(jac_vto)
    #print(jac_adv)
    #raise

    def condition(state):
        _,_,_, residual_combo, i = state
        return (residual_combo > 1e-5) & (i < iterations)


    def iterate(state):
        u, d, d_old, residual_combo, i = state

        mu = new_mu(u)

        jac_vto = jac_vto_fn(u, d, mu)
        u = u + lalg.solve(jac_vto, -vto(u, d, mu))

        adv_vto = jac_adv_fn(u, d, d_old)
        d = d.at[:-1].set(d[:-1] + lalg.solve(adv_vto[:-1, :-1], -adv(u, d, d_old)[:-1])) #relaxation solves nothing
        #d = d.at[:-1].set(d[:-1] + lalg.solve(adv_vto[:-1, :-1], -adv(u, d, d_old)[:-1]))
        d = jnp.minimum(d, 0.95)
        d = jnp.maximum(d, 0)

        residual1 = jnp.max(jnp.abs(vto(u, d, mu)))
        residual2 = jnp.max(jnp.abs(adv(u, d, d_old)))
        residual_combo = residual1+residual2

        return u, d, d_old, residual_combo, i+1


    def iterator(u_init, d_init):

        u_end, d_end, d_old_end, res, itns = while_loop(condition, iterate, (u_init, d_init, d_init, 1, 0))

        return u_end, d_end


    def iterator_dfct(u_init, d_init):

        u = u_init.copy()
        d = d_init.copy()
        d_old = d_init.copy()

#        prev_residual_combo = 1

        for i in range(iterations):
            
#            residual1 = jnp.max(jnp.abs(vto(u, d, mu)))
#            residual2 = jnp.max(jnp.abs(adv(u, d, d_old)))
#            print(residual1, residual2)

            mu = new_mu(u)
            
            jac_vto = jac_vto_fn(u, d, mu)
            u = u + lalg.solve(jac_vto, -vto(u, d, mu))

            adv_vto = jac_adv_fn(u, d, d_old)
            d = d.at[:-1].set(d[:-1] + lalg.solve(adv_vto[:-1, :-1], -adv(u, d, d_old)[:-1]))
            d = jnp.minimum(d, 0.95)
            d = jnp.maximum(d, 0)

#TODO: MAKE THIS JIT COMPATIBLE
#            if ((residual1 < 1e-7) and (residual2 < 1e-7)) or\
#               (jnp.abs((residual1+residual2)-prev_residual_combo) < 1e-8):
#                break

#            prev_residual_combo = residual1+residual2
#
#        residual_1 = jnp.max(jnp.abs(vto(u, d, mu)))
#        residual_2 = jnp.max(jnp.abs(adv(u, d, d_old)))
#        print(residual1, residual2)

        return u, d
       
    return iterator


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


def plotboth(d, speed, title=None, savepath=None, axis_limits=None, show_plots=True):

    fig, ax1 = plt.subplots(figsize=(10,6))
    ax2 = ax1.twinx()

    ax1.plot(d, c='k')
    ax2.plot(speed, color='blue', marker=".", linewidth=0)

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


mu = jnp.zeros((n,)) + 1


#OVERDEEPENED BED


h = 5*jnp.exp(-2*x*x*x*x)
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

p_W = 1.027 * jnp.maximum(0, h-s)
p_I = 0.917 * h
phi = 1 - (p_W / p_I)
#phi = 1
beta = 3000 * phi
beta = jnp.where(s_gnd>s_flt, beta, 0)


base = s - h


epsilon = 1e-6




#plotgeom(h)
#raise






#################################
#Bits for advection tests:

#u_test = jnp.array([4.06286308e-05, 1.22049154e-04, 2.03960386e-04, 2.86693161e-04, 3.70583002e-04, 4.55971254e-04, 5.43205708e-04, 6.32642419e-04, 7.24646670e-04, 8.19593377e-04, 9.17867990e-04, 1.01986690e-03, 1.12599786e-03, 1.23667973e-03, 1.35234231e-03, 1.47342647e-03, 1.60038169e-03, 1.73366640e-03, 1.87374617e-03, 2.02109152e-03, 2.17617722e-03, 2.33947788e-03, 2.51146685e-03, 2.69261398e-03, 2.88338074e-03, 3.08421929e-03, 3.29556619e-03, 3.51784076e-03, 3.75144207e-03, 3.99674522e-03, 4.25409665e-03, 4.52381093e-03, 4.80616884e-03, 5.10141253e-03, 5.40974690e-03, 5.73133165e-03, 6.06628321e-03, 6.41467143e-03, 6.77652191e-03, 7.15181092e-03, 7.54047092e-03, 7.94238970e-03, 8.35741218e-03, 8.78534093e-03, 9.22594592e-03, 9.67896264e-03, 1.01441033e-02, 1.06210560e-02, 1.11094965e-02, 1.16090979e-02, 1.21195298e-02, 1.26404697e-02, 1.31716151e-02, 1.37126809e-02, 1.42634129e-02, 1.48235867e-02, 1.53930075e-02, 1.59715042e-02, 1.65589228e-02, 1.71551052e-02, 1.77598596e-02, 1.83729082e-02, 1.89938471e-02, 1.96220297e-02, 2.02564690e-02, 2.08956860e-02, 2.15375088e-02, 2.21788380e-02, 2.28153709e-02, 2.34412868e-02, 2.40489636e-02, 2.46287491e-02, 2.51689311e-02, 2.56560873e-02, 2.60760337e-02, 2.64156479e-02, 2.66657956e-02, 2.68252064e-02, 2.69045290e-02, 2.69324314e-02, 2.69446224e-02, 2.69553512e-02, 2.69647520e-02, 2.69729421e-02, 2.69800462e-02, 2.69861743e-02, 2.69914325e-02, 2.69959215e-02, 2.69997343e-02, 2.70029511e-02, 2.70056520e-02, 2.70079058e-02, 2.70097777e-02, 2.70113200e-02, 2.70125866e-02, 2.70136185e-02, 2.70144548e-02, 2.70151310e-02, 2.70156711e-02, 0.00000000e+00])
#
##plt.plot(u_test)
##plt.show()
#
##u_test = 0.04*(x**2)
#
#u_const = jnp.ones_like(x)/100
#h_const = jnp.ones_like(x)
##set end to zero:
#h_const = h_const.at[-1].set(0)




###################################
##Explicit version:


#d_start = jnp.zeros((n,))
#d_start = d_start.at[-80:-75].set(0.9)
#
#d_t = d_start.copy()
#
#ds = [d_t]
#
#advector = make_adv_expt(h_const, 0.1)
#
#for i in range(101):
#    d_t = advector(u_test, d_t)
#
#    if not i%10:
#        print(i)
#        ds.append(d_t)
#
#plt.figure(figsize=(5,5))
#for d_t in ds:
#    plt.plot(d_t)
#plt.show()
#
#
#
#raise





###################################
## IMPLICIT VERSION



#iterator = jax.jit(make_picard_adv_only(u_test, jnp.ones_like(u_test), h_const, beta, 1, 10))
##iterator = jax.jit(make_picard_adv_only(u_const, jnp.ones_like(u_test), h_const, beta, 0.1, 100))
#
#d_start = jnp.zeros((n,))
#d_start = d_start.at[-80:-75].set(0.9)
#
#d_t = d_start.copy()
#
#ds = [d_t]
#
#
#for i in range(101):
#    d_t = iterator(d_t)
#
#    if not i%10:
#        print(i)
#        ds.append(d_t)
#
#plt.figure(figsize=(5,5))
#for d_t in ds:
#    plt.plot(d_t)
#plt.show()
#
#raise



################################

###PICARD JOINT PROBLEM:
#-----------------------#

u_trial = 0*(jnp.exp(x)-1)

d_trial = jnp.zeros((n,))
d_test = jnp.zeros((n,))
d_test = d_test.at[-80:-70].set(0.8)

iterator = jax.jit(make_picard_iterator_full(jnp.ones_like(u_trial), h, beta, 0.005, 30))
#iterator = make_picard_iterator_full(jnp.ones_like(u_trial), h, beta, 0.2, 40)

#NOTE: something very wrong with advection probably as we're waaaaaaaay below cfl limit and
#it looks unstable. Test advection only.
#advection seems to work! Maybe something wrong in vel solve...

u_t = u_trial.copy()
#d_t = d_test.copy() #useful for testing advection
d_t = d_trial.copy()

us = [u_t]
ds = [d_t]
for i in range(1001):
    u_t, d_t = iterator(u_t, d_t)
    #print(u_t)
    #plotboth(d_t, u_t, title=None, savepath=None, axis_limits=None, show_plots=True)
    
    if i%100==0:
        print(i)
        us.append(u_t)
        ds.append(d_t)

plotboths(ds, us, 1000, title=None, savepath=None, axis_limits=None, show_plots=True)


raise




###################################
###PICARD SOLVE FOR U ONLY. Sort of works, convergence a bit shite. Not sure if something's up.

#u_trial = 1*(jnp.exp(x)-1)
u_trial = jnp.zeros((n,))

d_test = jnp.zeros((n,))
#d_test = d_test.at[-10:-8].set(0.9)

iterator = make_picard_iterator(jnp.ones_like(u_trial), h, beta, d_test, 20)
u_end = iterator(u_trial)

print(u_end)

plt.plot(u_end)
plt.show()

raise




###################################
##Just u using Newton's method. Doesn't reliably converge!!!

d_test = jnp.zeros((n,))
#d_test = d_test.at[80:81].set(0.75)

#u_trial = 6*(jnp.exp(x)-1)
u_trial = jnp.zeros((n,))

u_solve = make_u_solver(u_trial, 30, d_test)

u_end = u_solve()

print(u_end)

plt.plot(u_end)
plt.show()


raise




###################################
##Coupled, time-dependent problem, Newton:

dt = 0.01
num_iterations = 30
num_timesteps = 5

u_trial = 6*(jnp.exp(x)-1)
d_trial = jnp.zeros((n,))

ns = solver(u_trial, d_trial, dt, num_iterations, num_timesteps)

u_end, d_end, ds, us = ns()

plotboths(ds, us, num_timesteps, axis_limits=[[-0.2, 1.2], [0,1.5]])

raise



