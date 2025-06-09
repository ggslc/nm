import jax
import jax.numpy as jnp
from jax import jacfwd, jacrev
import jax.scipy.linalg as lalg
from jax.scipy.optimize import minimize

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


np.set_printoptions(precision=4, suppress=False, linewidth=np.inf, threshold=np.inf)

def make_nonlinear_momentum_residual(beta, mu_cc):
    #NOTE: taken out the nonlinearity

    mu_face = jnp.zeros((n+1,))
    mu_face = mu_face.at[1:n].set(0.5 * (mu_cc[:n-1] + mu_cc[1:n]))
    #mu_face = mu_face.at[-1].set(mu_face[-2])
    mu_face = mu_face.at[0].set(mu_cc[1])

    
    def mom_res(u, h, rheology_factor=5e-2):
        s_gnd = h + b
        s_flt = h*(1-0.917/1.027)
        s = jnp.maximum(s_gnd, s_flt)

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


        mu_face_nl = rheology_factor * mu_face * (jnp.abs(dudx)+epsilon)**(-2/3)
        #mu_face_nl = 1e-1 * mu_face * (jnp.abs(dudx)+epsilon)**(-2/3)
        #mu_face_nl = mu_face.copy()


        flux = h_face * mu_face_nl * dudx


        #flux = flux.at[-2].set(0)


        h_grad_s = jnp.zeros((n,))
        h_grad_s = h_grad_s.at[1:n-1].set(h[1:n-1] * 0.5 * (s[2:n] - s[:n-2]))
        h_grad_s = h_grad_s.at[-1].set(-h[-1] * 0.5 * s[-2])
        #h_grad_s = h_grad_s.at[-2].set(-0.1)
        h_grad_s = h_grad_s.at[0].set(h[0] * 0.5 * (s[1] - s[0]))
      
        #print(flux)
        #print(sliding)
        #print(h_grad_s)
      
        
        return flux[1:] - flux[:-1] - h_grad_s - sliding
        #return - h_grad_s - sliding
        #return flux[1:] - flux[:-1] - sliding

    return mom_res

def make_nonlinear_momentum_residual_stress_split_at_gl(beta, mu_cc):
    #NOTE: taken out the nonlinearity

    mu_face = jnp.zeros((n+1,))
    mu_face = mu_face.at[1:n].set(0.5 * (mu_cc[:n-1] + mu_cc[1:n]))
    #mu_face = mu_face.at[-1].set(mu_face[-2])
    mu_face = mu_face.at[0].set(mu_cc[1])

    
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
       

        sliding = beta * u * dx
        #making sure the Jacobian is full rank!
        sliding = sliding.at[:].set(jnp.where(h>0, jnp.where(s_gnd>s_flt, sliding, 0), u * dx))

        
        h_face = jnp.zeros((n+1,))
        h_face = h_face.at[1:n].set(0.5 * (h[1:n] + h[:n-1]))
        h_face = h_face.at[-1].set(0)
        h_face = h_face.at[0].set(h[0])


        mu_face_nl = 5e-2 * mu_face * (jnp.abs(dudx)+epsilon)**(-2/3)
        #mu_face_nl = mu_face.copy()


        flux = h_face * mu_face_nl * dudx


        #flux = flux.at[-2].set(0)


        h_grad_s = jnp.zeros((n,))
        h_grad_s = h_grad_s.at[1:n-1].set(h[1:n-1] * 0.5 * (s[2:n] - s[:n-2]))
        h_grad_s = h_grad_s.at[-1].set(-h[-1] * 0.5 * s[-2])
        #h_grad_s = h_grad_s.at[-2].set(-0.1)
        h_grad_s = h_grad_s.at[0].set(h[0] * 0.5 * (s[1] - s[0]))
      
        #print(flux)
        #print(sliding)
        #print(h_grad_s)
      
        
        return flux[1:] - flux[:-1] - h_grad_s - sliding
        #return - h_grad_s - sliding
        #return flux[1:] - flux[:-1] - sliding

    return mom_res

def make_linear_momentum_residual(beta):
    #NOTE: taken out the nonlinearity

    def mom_res(u, h, mu_face):
        s_gnd = h + b
        s_flt = h*(1-0.917/1.027)
        s = jnp.maximum(s_gnd, s_flt)

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


        #flux = flux.at[-2].set(0)


        h_grad_s = jnp.zeros((n,))
        h_grad_s = h_grad_s.at[1:n-1].set(h[1:n-1] * 0.5 * (s[2:n] - s[:n-2]))
        h_grad_s = h_grad_s.at[-1].set(-h[-1] * 0.5 * s[-2])
        #h_grad_s = h_grad_s.at[-2].set(-0.1)
        h_grad_s = h_grad_s.at[0].set(h[0] * 0.5 * (s[1] - s[0]))
      
        #print(flux)
        #print(sliding)
        #print(h_grad_s)
        
        return flux[1:] - flux[:-1] - h_grad_s - sliding

    return mom_res


def make_linear_momentum_residual_dfct(beta):

    def mom_res(u, h, mu_face):
        s_gnd = h + b
        s_flt = h*(1-0.917/1.027)
        s = jnp.maximum(s_gnd, s_flt)

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

        #mom_res_nl = make_nonlinear_momentum_residual(beta, mu_centres_zero)
        
        jac_mom_res_fn = jacfwd(mom_res, argnums=0)


        mu = mu_faces_zero.copy()
        u = u_init.copy()

        prev_residual = 1

        us = [u]

        for i in range(iterations):
            
            mu = new_mu(u) 
            
            residual = jnp.max(jnp.abs(mom_res(u, h, mu))) 
            #residual = jnp.max(jnp.abs(mom_res_nl(u, h)))
            print(residual, prev_residual/residual)

            
            jac_mom_res = jac_mom_res_fn(u, h, mu) 

            du = lalg.solve(jac_mom_res, -mom_res(u, h, mu))
            print(jac_mom_res@du + -mom_res(u, h, mu))

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



def make_adv_operator(dt, accumulation):
    
    def adv_op(u, h, h_old, basal_melt_rate):
        s_gnd = h + b
        s_flt = h*(1-0.917/1.027)

        acc = jnp.where(s_gnd<s_flt, accumulation-basal_melt_rate, accumulation)
        acc = acc.at[:].set(jnp.where(h>0, acc, 0))

        h_face = jnp.zeros((n+1,))
        h_face = h_face.at[1:n+1].set(h[:n]) #upwind values
        h_face = h_face.at[0].set(h[0].copy())

        u_face = jnp.zeros((n+1,))
        u_face = u_face.at[1:n].set(0.5*(u[1:n]+u[:n-1]))
        u_face = u_face.at[-1].set(2*u[-1] - u[-2]) #extrapolating u (linear)

        h_flux = h_face * u_face
        #the two lines below were supposed to stop everything piling up in the last
        #cell but they had the opposite effect for some reason...
        #h_flux = h_flux.at[-2].set(h_flux[-3]) #stop everythin piling up at the end.
        #NOTE: This changes things a lot:
        #h_flux = h_flux.at[-1].set(h_flux[-2].copy())
        h_flux = h_flux.at[0].set(0)


        #return  (h - h_old)/dt - ( h_flux[1:(n+1)] - h_flux[:n] )/dx - acc
        #I think the above is a sign error in the flux!
        return  (h - h_old)*dx + dt*( h_flux[1:(n+1)] - h_flux[:n] ) - dt*dx*acc

    return adv_op    

def make_adv_operator_acc_dependent_on_old_h(dt, accumulation):
    
    def adv_op(u, h, h_old, basal_melt_rate):
        s_gnd = h_old + b
        s_flt = h_old*(1-0.917/1.027)

        acc = jnp.where(s_gnd<s_flt, accumulation-basal_melt_rate, accumulation)
        acc = acc.at[:].set(jnp.where(h>0, acc, 0))

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
        return  (h - h_old)/dt + ( h_flux[1:(n+1)] - h_flux[:n] )/dx - acc

    return adv_op    


def make_adv_rhs(accumulation):
    
    def adv(u, h, basal_melt_rate):
        s_gnd = h + b
        s_flt = h*(1-0.917/1.027)

        acc = jnp.where(s_gnd<s_flt, accumulation-basal_melt_rate, accumulation)
        acc = acc.at[-1].set(0)
        acc = acc.at[:].set(jnp.where(h>0, acc, 0))

        h_face = jnp.zeros((n+1,))
        h_face = h_face.at[1:n+1].set(h) #upwind values
        h_face = h_face.at[0].set(h[0])

        u_face = jnp.zeros((n+1,))
        u_face = u_face.at[1:n].set(0.5*(u[1:n]+u[:n-1]))
        u_face = u_face.at[-1].set(2*u[-1]-u[-2])

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
                            compute_eigenvalues=False, \
                            compute_singular_values=False):



    mom_res = make_nonlinear_momentum_residual(beta, mu)
    adv = make_adv_operator(dt, accumulation)
    #adv = make_adv_operator_acc_dependent_on_old_h(dt, accumulation)
    adv_rhs = make_adv_rhs(accumulation)

    visc_jac_fn = jacfwd(mom_res, argnums=(0,1))
    adv_jac_fn = jacfwd(adv, argnums=(0,1))


    def timestep_condition(state):
        t = state[-1]
        return t<num_timesteps


    def newton_condition(state):
        i = state[-1]
        return i<num_iterations


    def make_newton_iterate(bmr, rf, h_old):
        def newton_iterate(state):
            u, h, i = state
            
            visc_jac = visc_jac_fn(u, h, rf)
            adv_jac = adv_jac_fn(u, h, h_old, bmr)
    
            full_jacobian = jnp.block(
                                      [ [visc_jac[0], visc_jac[1]],
                                        [adv_jac[0] , adv_jac[1]] ]
                                      )
    
            rhs = jnp.concatenate((-mom_res(u, h, rf), -adv(u, h, h_old, bmr)))
    
            dvar = lalg.solve(full_jacobian, rhs)
    
            u = u.at[:].set(u+dvar[:n])
            h = h.at[:].set(h+dvar[n:])
    
            #print(jnp.max(jnp.abs(mom_res(u, h))), jnp.max(jnp.abs(adv(u, h, h_old, basal_melt_rate))))
    
            return u, h, i+1
        return newton_iterate


    def timestep(state):
        u, h, us, hs, bmr, rf, t = state

        #jax.debug.print("t = {}", t)

        #bmr_new = bmr.copy() #could make this some time-dependent function y'see.

        newton_iterate = make_newton_iterate(bmr, rf, h.copy())

        initial_state = (u, h, 0)
        u_new, h_new, i = jax.lax.while_loop(newton_condition, newton_iterate, initial_state)

        us = us.at[t].set(u_new)
        hs = hs.at[t].set(h_new)
        #bmrs = bmrs.at[t].set(bmr_new)

        return u_new, h_new, us, hs, bmr, rf, t+1
        

    @jax.jit
    def iterator(u_init, h_init, bmr, rheology_factor):

        #have to pre-allocate these things because we can't use python-side
        #mutations like appending to lists in a lax while_loop!
        us = jnp.zeros((num_timesteps, n))
        hs = jnp.zeros((num_timesteps, n))
        #bmrs = jnp.zeros((num_timesteps, n))

        initial_state = (u_init, h_init, us, hs, bmr, rheology_factor, 0)

        u, h, us, hs, bmr, rheology_factor, t = jax.lax.while_loop(timestep_condition, timestep, initial_state)


        if compute_eigenvalues or compute_singular_values:
            
            H_jac = jacfwd(adv_rhs, argnums=(0,1))(u, h, bmr)

            adv_residual_jac = adv_jac_fn(u, h, hs[-2], bmr)

            visc_jac = visc_jac_fn(u, h)

            L, fdbk, dHdh = construct_tangent_propagator(H_jac[1],\
                                             H_jac[0],\
                                             visc_jac[1],\
                                             visc_jac[0])

            #This has the most expected behaviour, despite being nothing!
            #It's not even a real thing. It's got the visc rhs on the bottom..!?
            #J = jnp.block(
            #                          [ [visc_jac[0], visc_jac[1]],
            #                            [H_jac[0] , H_jac[1]] ]
            #                          )

            J = jnp.block(
                                      [ [visc_jac[0], visc_jac[1]],
                                        [adv_residual_jac[0] , adv_residual_jac[1]] ]
                                      )

            s_gnd = h + b
            s_flt = h*(1-0.917/1.027)


            mask = jnp.where(s_gnd>s_flt, 1, 0)
            mask = jnp.outer(mask, mask)

            L_cr = L * mask
            fdbk_cr = fdbk * mask
            ptl_cr = dHdh * mask

            if compute_eigenvalues:

                #ffi = np.where(s_gnd<s_flt)[0][0]
                #L_cr = L[:ffi, :ffi]
    


                #Exhibits critical slowing down:
                evals, evecs = jnp.linalg.eig(J)
                order_indices = jnp.argsort(jnp.real(evals))
                evals_ord = evals[order_indices]
                evecs_ord = evecs[:, order_indices]
    
#                lsvecs, svals, rsvecs_t = jnp.linalg.svd(L_cr)
#                evals_ord = jnp.argsort(svals)
#                evecs_ord = rsvecs_t

                #evals, evecs = jnp.linalg.eig(fdbk)
                #evecs, evals, _ = jnp.linalg.svd(L) #I know they're not eigen-...!
                
                #evecs_ptl, evals_ptl, _ = jnp.linalg.svd(H_jac[1][:n-1, :n-1])
    
                #indices = jnp.argsort(evals)
                #evals_ordered = evals[indices]
                #evecs_ordered = evecs[:,indices]
    
                #indices_ptl = jnp.argsort(evals_ptl)
                #evals_ordered_ptl = evals[indices_ptl]
                #evecs_ordered_ptl = evecs[:,indices_ptl]
    
                return u, h, us, hs, bmr, rheology_factor, evals_ord, evecs_ord, mask

            elif compute_singular_values:

                lsvecs, svals, rsvecs_t = jnp.linalg.svd(L_cr)
                order_indices = jnp.argsort(svals)
                svals_ord = svals[order_indices]
                rsvecs_ord = jnp.transpose(rsvecs_t)[:, order_indices]


                lsvecs_ptl, svals_ptl, rsvecs_t_ptl = jnp.linalg.svd(ptl_cr)
                order_indices_ptl = jnp.argsort(svals_ptl)
                svals_ord_ptl = svals_ptl[order_indices_ptl]
                rsvecs_ord_ptl = jnp.transpose(rsvecs_t_ptl)[:, order_indices_ptl]


                lsvecs_fdbk, svals_fdbk, rsvecs_t_fdbk = jnp.linalg.svd(fdbk_cr)
                order_indices_fdbk = jnp.argsort(svals_fdbk)
                svals_ord_fdbk = svals_fdbk[order_indices_fdbk]
                rsvecs_ord_fdbk = jnp.transpose(rsvecs_t_fdbk)[:, order_indices_fdbk]


                return u, h, us, hs, bmr, rheology_factor, svals_ord, rsvecs_ord,\
                       svals_ord_ptl, rsvecs_ord_ptl, svals_ord_fdbk, rsvecs_ord_fdbk 
        else:
            return u, h, us, hs, bmr, rheology_factor

    return iterator



def implicit_coupled_solver(u_trial, h_trial,\
                            accumulation, dt, \
                            basal_melt_rate, \
                            num_iterations, num_timesteps, \
                            compute_evals=False, \
                            compute_evals_at_ss=False):

    def newton_solve(mu):

        mom_res = make_nonlinear_momentum_residual(beta, mu)
        adv = make_adv_operator(dt, accumulation)
        #adv = make_adv_operator_acc_dependent_on_old_h(dt, accumulation)

        mom_res_jac_fn = jacfwd(mom_res, argnums=(0,1))
        adv_jac_fn = jacfwd(adv, argnums=(0,1))

        u = u_trial.copy()
        h = h_trial.copy()
        h_old = h_trial.copy()

        
        #for debugging:
        #mom_res(u, h)
        #raise


        hs = []
        us = []
        largest_evals = []
        all_evals = []
        largest_evals_ptl = []
        all_evals_ptl = []
        largest_evals_fdbk = []
        all_evals_fdbk = []

        leading_evecs = []
        leading_evecs_ptl = []
        leading_evecs_fdbk = []


        for j in range(num_timesteps):

            print(j)
            for i in range(num_iterations):
                mom_res_jac = mom_res_jac_fn(u, h)
                
                adv_jac = adv_jac_fn(u, h, h_old, basal_melt_rate)
        
                full_jacobian = jnp.block(
                                          [[mom_res_jac[0], mom_res_jac[1]],
                                          [adv_jac[0], adv_jac[1]]]
                                          )

                #print(np.array(mom_res_jac[0]))
                #print("-------------------")
                #print("-------------------")
                #print("-------------------")
                ##print(np.array(adv_jac[0]))
                #print("-------------------")
                #print("-------------------")
                #print("-------------------")
                #print(np.array(mom_res_jac[1]))
                #print("-------------------")
                #print("-------------------")
                #print("-------------------")
                ##print(np.array(adv_jac[1]))
#               # print("-------------------")
#               # print("-------------------")
#               # print("-------------------")
#               # print(np.array(full_jacobian))
                ##raise


                rhs = jnp.concatenate((-mom_res(u, h), -adv(u, h, h_old, basal_melt_rate)))

                dvar = lalg.solve(full_jacobian, rhs)


                u = u.at[:].set(u+dvar[:n])
                h = h.at[:].set(h+dvar[n:])

                print(jnp.max(jnp.abs(mom_res(u, h))), jnp.max(jnp.abs(adv(u, h, h_old, basal_melt_rate))))


            #plotboth(h, u)
            
            if compute_evals:
                H = make_adv_rhs(accumulation)
                H_jac = jacfwd(H, argnums=(0,1))(u, h, basal_melt_rate)

                L, fdbk, dHdh = construct_tangent_propagator(H_jac[1],\
                                                             H_jac[0],\
                                                             mom_res_jac[1],\
                                                             mom_res_jac[0])

                

                s_gnd = h + b
                s_flt = h*(1-0.917/1.027)

                ffi = np.where(s_gnd<s_flt)[0][0]



                #NOTE:
                
                L_cr = L[:ffi, :ffi]
                ptl_cr = dHdh[:ffi, :ffi]
                fdbk_cr = fdbk[:ffi, :ffi]
               
                ##looking at which perturbations across the whole space
                ##are most amplified in the grounded ice thickness:
                #L_cr = L[:ffi, :]
                #ptl_cr = dHdh[:ffi, :]
                #fdbk_cr = fdbk[:ffi, :]
                
                #L_cr = L
                #ptl_cr = dHdh
                #fdbk_cr = fdbk


                

                #singular values:
                lsvecs, svals, rsvecs_t = jnp.linalg.svd(L_cr)
                order_indices = jnp.argsort(svals)
                svals_ord = svals[order_indices]
                rsvecs_ord = jnp.transpose(rsvecs_t)[:, order_indices]

                largest_evals.append(svals_ord[-1])
                leading_evecs.append(rsvecs_ord[:,-1])

                ##NOTE: I'm just overwriting this with eigenvalues as I wanted to have a look!
                #evals, evecs = jnp.linalg.eig(L_cr)
                #order_indices = jnp.argsort(evals)
                #largest_evals.append(evals[order_indices][-1])
                #leading_evecs.append(evecs[:, order_indices][-1])



                lsvecs_ptl, svals_ptl, rsvecs_t_ptl = jnp.linalg.svd(ptl_cr)
                order_indices_ptl = jnp.argsort(svals_ptl)
                svals_ord_ptl = svals_ptl[order_indices_ptl]
                rsvecs_ord_ptl = jnp.transpose(rsvecs_t_ptl)[:, order_indices_ptl]

                largest_evals_ptl.append(svals_ord_ptl[-1])
                leading_evecs_ptl.append(rsvecs_ord_ptl[:,-1])



                lsvecs_fdbk, svals_fdbk, rsvecs_t_fdbk = jnp.linalg.svd(fdbk_cr)
                order_indices_fdbk = jnp.argsort(svals_fdbk)
                svals_ord_fdbk = svals_fdbk[order_indices_fdbk]
                rsvecs_ord_fdbk = jnp.transpose(rsvecs_t_fdbk)[:, order_indices_fdbk]

                largest_evals_fdbk.append(svals_ord_fdbk[-1])
                leading_evecs_fdbk.append(rsvecs_ord_fdbk[:,-1])




#            print(svals_ord[-1])
    #            plt.imshow(jnp.rot90(rsvecs_ord), vmin=-1, vmax=1, cmap="RdBu_r")
    #            plt.show()


                ##eigenvalues
                #evals, evecs = jnp.linalg.eig(L_cr)
                #order_indices = jnp.argsort(evals)
                #evals_ord = evals[order_indices]
                #evecs_ord = evecs[:, order_indices]

                #print(evals_ord[-1])
                #largest_evals.append(evals_ord[-1])


                #print(evals_ord[-1])
                #plt.imshow(jnp.rot90(jnp.real(evecs_ord)))
                #plt.show()
                #raise




                #max_speed = jnp.max(u)

                #evecs, evals, Vh = jnp.linalg.svd(L_cr)#/max_speed) #I know they're not eigen-...!
                #evecs_ptl, evals_ptl, Vh_ptl = jnp.linalg.svd(H_jac[1])#/max_speed)
                #evecs_fdbk, evals_fdbk, Vh_fdbk = jnp.linalg.svd(fdbk)#/max_speed)

                ##evals, evecs = jnp.linalg.eig(L) #I know they're not eigen-...!
                ##evals_ptl, evecs_ptl = jnp.linalg.eig(H_jac[1])

           
                #plotboth(h,u)


                #indices = jnp.argsort(evals)
                #evals_ordered = evals[indices]
                #right_evecs_ordered = Vh.T[:,indices]
                #left_evecs_ordered = evecs[:,indices]
                
                #print(evals_ordered[0])
                #plt.plot(evecs_ordered[0])
                #plt.show()

                #print(evals_ordered[-1])
                #plt.plot(evecs_ordered[-1])
                #plt.show()
                #raise

                
                #plt.imshow(jnp.rot90(right_evecs_ordered), vmin=-0.5, vmax=0.5, cmap="RdBu_r")
                #plt.show()
                #
                #plt.imshow(jnp.rot90(left_evecs_ordered), vmin=-0.5, vmax=0.5, cmap="RdBu_r")
                #plt.show()

                ##raise


                #indices_ptl = jnp.argsort(evals_ptl)
                #evals_ordered_ptl = evals_ptl[indices_ptl]
                #evecs_ordered_ptl = evecs_ptl[:,indices_ptl]


                #indices_fdbk = jnp.argsort(evals_fdbk)
                #evals_ordered_fdbk = evals_fdbk[indices_fdbk]
                #evecs_ordered_fdbk = evecs_fdbk[:,indices_fdbk]
                #
                #plt.imshow(jnp.transpose(evecs_ordered_fdbk), vmin=-0.5, vmax=0.5, cmap="RdBu_r")
                #plt.show()
                ##raise

                ##plt.imshow(jnp.real(evecs_ordered), vmin=-0.2, vmax=0.2)
                ##plt.show()


                ##NOTE: I'm actually looking at the smallest eigenvalues now!

                #all_evals.append(evals)
                #largest_evals.append(evals_ordered[-1]) #largest
                ##largest_evals.append(evals_ordered[0]) #smallest
                #
                #all_evals_ptl.append(evals_ptl)
                #largest_evals_ptl.append(evals_ordered_ptl[-1])
                ##largest_evals_ptl.append(evals_ordered_ptl[0])

                #all_evals_fdbk.append(evals_fdbk)
                #largest_evals_fdbk.append(evals_ordered_fdbk[-1])
                ##largest_evals_ptl.append(evals_ordered_ptl[0])
            
            hs.append(h)
            us.append(u)


            h_old = h.copy()


        if compute_evals_at_ss:
            H = make_adv_rhs(accumulation)
            H_jac = jacfwd(H, argnums=(0,1))(u, h, basal_melt_rate)

            L, fdbk, dHdh = construct_tangent_propagator(H_jac[1],\
                                                         H_jac[0],\
                                                         mom_res_jac[1],\
                                                         mom_res_jac[0])

            

            s_gnd = h + b
            s_flt = h*(1-0.917/1.027)

            ffi = np.where(s_gnd<s_flt)[0][0]


            L_cr = L[:ffi, :ffi]



            #eigenvalues
            evals, evecs = jnp.linalg.eig(L_cr)
            order_indices = jnp.argsort(evals)
            evals_ord = evals[order_indices]
            evecs_ord = evecs[:, order_indices]

        #return u, h, hs, us, evals_ord, evecs_ord

        #return u, h, hs, us, largest_evals, jnp.array(all_evals), \
        #       largest_evals_ptl, jnp.array(all_evals_ptl), \
        #       largest_evals_fdbk, jnp.array(all_evals_fdbk)
        return u, h, hs, us, largest_evals, leading_evecs, \
               largest_evals_ptl, leading_evecs_ptl, \
               largest_evals_fdbk, leading_evecs_fdbk

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



def plotgeoms(thks, upper_lim, title=None, savepath=None, axis_limits=None, show_plots=True):

    if isinstance(hs, (jnp.ndarray, np.ndarray)):
        hs_list = [np.array(h) for h in hs]

    fig, ax1 = plt.subplots(figsize=(10,6))
    ax2 = ax1.twinx()

    n = len(thks)

    cmap = cm.rainbow
    cs = cmap(jnp.linspace(0, 1, n))

    for thk,  c1 in list(zip(thks, cs)):
        s_gnd = b + thk
        s_flt = thk*(1-0.917/1.027)
        s = jnp.maximum(s_gnd, s_flt)

        base = s-thk
        ax1.plot(s, c=c1)
        # ax1.plot(base, label="base")
        ax1.plot(base, c=c1)

    #add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=upper_lim))
    sm._A = []
    cbar = fig.colorbar(sm, ax=ax1, orientation='horizontal', pad=0.15)
    cbar.set_label('Timestep')

    ax1.plot(b, label="bed", c="k")
    
    #axis labels
    ax1.set_xlabel("x")
    ax1.set_ylabel("elevation")

    if axis_limits is not None:
        ax1.set_ylim(axis_limits[0])

    if title is not None:
        plt.title(title)
    if savepath is not None:
        plt.savefig(savepath)

    if show_plots:
        plt.show()


def pad_arrays_with_nan(arrays):
    max_len = max(len(arr) for arr in arrays)
    result = np.full((len(arrays), max_len), np.nan)

    for i, arr in enumerate(arrays):
        result[i, :len(arr)] = arr

    return result





rho = 1
g = 1

lx = 1
n = 21
dx = lx/(n-1)
x = jnp.linspace(0,lx,n)

mu = jnp.zeros((n,)) + 1


#OVERDEEPENED BED

h = 1.5*jnp.exp(-2*x*x*x*x*x*x*x) #grounded sections
#h = 1.5*jnp.exp(-2*x*x*x*x) #grounded sections
#h = jnp.ones_like(x)*0.5 #floating uniform thk

#h = 1.75*jnp.exp(-2*x*x) #just a bit of an odd shaped ice shelf
#h = (1+jnp.zeros((n,)))*(1-x/2)
#h = h.at[-1].set(0)


#b_intermediate = jnp.zeros((n,))-0.5
b_intermediate = 0.1 - 0.8*x.copy()

s_gnd = b_intermediate + h
s_flt = h*(1-0.917/1.027)
s = jnp.maximum(s_gnd, s_flt)
#s = s.at[-1].set(0)


#b = jnp.zeros((n,))-0.5
#b = b.at[:n].set(b[:n] - 0.15*jnp.exp(-(5*x-2)**2))

b = 0.1 - 0.8*x.copy()
#b = b.at[:n].set(b[:n] - 0.15*jnp.exp(-(5*x-2)**2))
#NOTE: this is the standard:
#b = b.at[:n].set(b[:n] - 0.35*jnp.exp(-(5*x-2)**2))
#b = b.at[:n].set(b[:n] - 0.3*jnp.exp(-(10*x-5)**4))
#NOTE: maybe works well with the steeper slope
b = b.at[:n].set(b[:n] - 0.5*jnp.exp(-(5*x-2)**2))


h = jnp.minimum(s-b, s/(1-0.917/1.027))


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



accumulation = jnp.zeros_like(h)+0.05
#accumulation = jnp.zeros_like(h)

#plotgeom(h)
#raise


#u_trial = jnp.exp(x)-1
#u_trial = x.copy()
u_trial = jnp.zeros_like(x)
h_trial = h.copy()







#######TESTING JACOBIANS:


h_test = jnp.array([2.434,2.4389,2.4437,2.4486,2.4534,2.4582,2.4631,2.4679,2.4728,2.4777,2.4827,2.4877,2.4928,2.4979,2.5032,2.5085,2.5139,2.5194,2.525,2.5308,2.5367,2.5427,2.5489,2.5553,2.5618,2.5685,2.5753,2.5824,2.5896,2.5971,2.6047,2.6125,2.6206,2.6288,2.6373,2.646,2.6548,2.6639,2.6731,2.6826,2.6922,2.702,2.7119,2.722,2.7322,2.7425,2.753,2.7635,2.7741,2.7847,2.7954,2.806,2.8166,2.8272,2.8377,2.8481,2.8583,2.8684,2.8783,2.8879,2.8973,2.9064,2.9152,2.9237,2.9318,2.9394,2.9467,2.9534,2.9597,2.9655,2.9707,2.9754,2.9794,2.9829,2.9857,2.9879,2.9895,2.9903,2.9905,2.99,2.9888,2.9869,2.9843,2.981,2.977,2.9723,2.9669,2.9608,2.9541,2.9467,2.9386,2.93,2.9207,2.9109,2.9004,2.8895,2.878,2.866,2.8535,2.8406,2.8272,2.8135,2.7994,2.7849,2.7701,2.755,2.7397,2.7241,2.7082,2.6922,2.676,2.6596,2.6431,2.6264,2.6096,2.5928,2.5758,2.5588,2.5418,2.5247,2.5075,2.4903,2.4731,2.4559,2.4386,2.4213,2.404,2.3867,2.3693,2.3519,2.3344,2.3169,2.2993,2.2816,2.2638,2.246,2.228,2.2098,2.1915,2.173,2.1543,2.1354,2.1162,2.0968,2.077,2.0569,2.0364,2.0156,1.9942,1.9724,1.9501,1.9272,1.9037,1.8795,1.8546,1.8288,1.8022,1.7746,1.7459,1.7161,1.685,1.6525,1.6184,1.5825,1.5446,1.5044,1.4617,1.4158,1.3665,1.3127,1.2543,1.1878,1.1167,1.0246,0.9388,0.7703,0.6634,0.6517,0.6413,0.6319,0.6233,0.6155,0.6083,0.6017,0.5956,0.59,0.5847,0.5798,0.5752,0.5709,0.5668,0.563,0.5594,0.556,0.5528,0.5497,0.5468,0.5441,0.5414,0.539,0.5341])
u_test = jnp.array([2.5724e-04,7.6986e-04,1.2803e-03,1.7888e-03,2.2952e-03,2.7997e-03,3.3022e-03,3.8027e-03,4.3012e-03,4.7977e-03,5.2922e-03,5.7845e-03,6.2748e-03,6.7628e-03,7.2487e-03,7.7323e-03,8.2136e-03,8.6925e-03,9.1688e-03,9.6427e-03,1.0114e-02,1.0582e-02,1.1048e-02,1.1511e-02,1.1970e-02,1.2427e-02,1.2880e-02,1.3330e-02,1.3777e-02,1.4220e-02,1.4659e-02,1.5095e-02,1.5527e-02,1.5955e-02,1.6379e-02,1.6799e-02,1.7215e-02,1.7627e-02,1.8035e-02,1.8439e-02,1.8839e-02,1.9235e-02,1.9626e-02,2.0014e-02,2.0398e-02,2.0778e-02,2.1154e-02,2.1527e-02,2.1896e-02,2.2262e-02,2.2626e-02,2.2986e-02,2.3343e-02,2.3699e-02,2.4052e-02,2.4403e-02,2.4753e-02,2.5102e-02,2.5450e-02,2.5797e-02,2.6144e-02,2.6491e-02,2.6839e-02,2.7188e-02,2.7538e-02,2.7890e-02,2.8244e-02,2.8600e-02,2.8960e-02,2.9323e-02,2.9689e-02,3.0060e-02,3.0436e-02,3.0817e-02,3.1203e-02,3.1595e-02,3.1994e-02,3.2399e-02,3.2811e-02,3.3231e-02,3.3659e-02,3.4094e-02,3.4539e-02,3.4992e-02,3.5455e-02,3.5927e-02,3.6409e-02,3.6901e-02,3.7403e-02,3.7916e-02,3.8441e-02,3.8976e-02,3.9523e-02,4.0081e-02,4.0651e-02,4.1233e-02,4.1827e-02,4.2434e-02,4.3052e-02,4.3683e-02,4.4327e-02,4.4983e-02,4.5652e-02,4.6333e-02,4.7027e-02,4.7734e-02,4.8453e-02,4.9186e-02,4.9931e-02,5.0688e-02,5.1459e-02,5.2242e-02,5.3038e-02,5.3846e-02,5.4667e-02,5.5501e-02,5.6348e-02,5.7207e-02,5.8080e-02,5.8965e-02,5.9863e-02,6.0774e-02,6.1699e-02,6.2638e-02,6.3590e-02,6.4556e-02,6.5537e-02,6.6533e-02,6.7544e-02,6.8572e-02,6.9616e-02,7.0677e-02,7.1756e-02,7.2854e-02,7.3972e-02,7.5111e-02,7.6272e-02,7.7456e-02,7.8665e-02,7.9901e-02,8.1165e-02,8.2459e-02,8.3786e-02,8.5147e-02,8.6545e-02,8.7984e-02,8.9466e-02,9.0995e-02,9.2576e-02,9.4211e-02,9.5908e-02,9.7670e-02,9.9505e-02,1.0142e-01,1.0342e-01,1.0552e-01,1.0773e-01,1.1006e-01,1.1253e-01,1.1515e-01,1.1794e-01,1.2093e-01,1.2415e-01,1.2764e-01,1.3144e-01,1.3561e-01,1.4024e-01,1.4540e-01,1.5125e-01,1.5793e-01,1.6583e-01,1.7501e-01,1.8700e-01,2.0029e-01,2.2427e-01,2.4175e-01,3.2948e-01,3.3756e-01,3.4525e-01,3.5258e-01,3.5961e-01,3.6636e-01,3.7288e-01,3.7918e-01,3.8528e-01,3.9120e-01,3.9696e-01,4.0258e-01,4.0805e-01,4.1340e-01,4.1864e-01,4.2376e-01,4.2879e-01,4.3372e-01,4.3857e-01,4.4333e-01,4.4802e-01,4.5263e-01,4.5718e-01,4.6166e-01,4.6605e-01])

momres_test = make_nonlinear_momentum_residual(beta, mu)
momres_test_jac = jacfwd(momres_test, argnums=(0,1))(u_test, h_test)
dgdu = momres_test_jac[0]
dgdh = momres_test_jac[1]

pert = jnp.zeros_like(u_test)+1e-5

ad_g_pert_u = dgdu @ pert
ad_g_pert_h = dgdh @ pert


fd_g_pert_u = momres_test(u_test+pert, h_test)-momres_test(u_test, h_test)
fd_g_pert_h = momres_test(u_test, h_test+pert)-momres_test(u_test, h_test)

print(ad_g_pert_u)
print(fd_g_pert_u)
print(ad_g_pert_u - fd_g_pert_u)
print("----------------")
print(ad_g_pert_h)
print(fd_g_pert_h)
print(ad_g_pert_h - fd_g_pert_h)
raise


###TESTING PICARD ITERATION:
#iterator = make_picard_iterator_for_u(mu, h_trial, beta, 20)
#u_end, us = iterator(u_trial)
#print("change to newton")
#iterator = make_u_solver(mu, h_trial, beta, 5)
#u_end_2, us_2 = iterator(u_end)
#
#plotboth(h, u_end)
#
#plt.figure(figsize=(10,5))
#for i, u_ in enumerate(us+us_2):
#    plt.plot(u_, label=str(i))
#plt.legend()
#plt.show()
#
#raise

##TESTING NEWTON FOR SPEED
#bmr = jnp.zeros_like(x)
#timestep = 0
#n_iterations = 10
#n_timesteps = 1
#
#acc = jnp.zeros_like(x)+0.08
#
##accumulation_scaled = acc/timestep
#
#solve_and_evolve = implicit_coupled_solver_compiled(mu, beta, acc, timestep, n_iterations, n_timesteps, compute_eigenvalues=True)
#u_ss, h_ss, us, hs, _,_,_,_ = solve_and_evolve(u_trial, h_trial, bmr)
#   
#plotboths(hs, us, n_timesteps)
#
#raise

#
#############TRYING TO GET THE JIT COMPILATION GOING:
##
#n_timesteps = 30
#n_iterations = 20
#timestep = 1
##
##bmr_init = jnp.zeros_like(accumulation)+0.005
##
##accumulation = accumulation/timestep
##
##
###bmr_init = bmr_init/timestep
###
###solve_and_evolve = implicit_coupled_solver_compiled(mu, beta, accumulation, timestep, n_iterations, n_timesteps, compute_eigenvalues=True)
###u_end, h_end, us, hs, bmrs, evals, evecs = solve_and_evolve(u_trial, h_trial, bmr_init)
###
###print(evals)
###
###plotboths(hs[:, :], us[:, :], n_timesteps)
###
###raise
##
##
##
##
##bmr_init = jnp.zeros_like(accumulation)
##
##solve_and_evolve = implicit_coupled_solver_compiled(mu, beta, accumulation, timestep, n_iterations, n_timesteps, compute_eigenvalues=True)
##u_end, h_end, us, hs, bmrs, evals, evecs = solve_and_evolve(u_trial, h_trial, bmr_init)
##
##largest_eval = evals[-1]
##largest_evec = evecs[-1]
##
##plotboths(hs[:, :], us[:, :], n_timesteps)
##
##u_end_2, h_end_2, us_2, hs_2, bmrs_2, evals_2, evecs_2 = solve_and_evolve(u_end, h_end+0.2*jnp.real(largest_evec), bmr_init)
##
##plotboths(hs_2[:,:], us_2[:,:], n_timesteps)
##
##plt.plot(h_end)
##plt.plot(h_end_2)
##plt.show()
##
##
##raise


#Plotting evolution of largest eigenvalues for steady-ish state solutions, changing the rate factor:
largest_evals = []
all_evals = []
smallest_evals = []
steady_state_us = []
steady_state_hs = []
bmr = jnp.zeros_like(x)
timestep = 0.2
n_iterations = 10
n_timesteps = 1
initial_n_timesteps = 4

accumulation_scaled = (jnp.zeros_like(x) + 0.05)/timestep


solve_and_evolve_initial = implicit_coupled_solver_compiled(mu, beta, accumulation_scaled,\
                                                            timestep, n_iterations, 1,\
                                                            compute_eigenvalues=True)

solve_and_evolve = implicit_coupled_solver_compiled(mu, beta, accumulation_scaled,\
                                                            timestep, n_iterations, n_timesteps,\
                                                            compute_eigenvalues=True)

n_different_As = 2

for k in range(n_different_As):

    rf = 3e-2*(1 - (k)/800+ 1e-4)
    #rf = 8e-3*(1 - k/350 + 1e-4)
    print(rf)

    if k==0:
        #print(h_end)
        #print(u_end)
        h_trial = jnp.array([2.434,2.4389,2.4437,2.4486,2.4534,2.4582,2.4631,2.4679,2.4728,2.4777,2.4827,2.4877,2.4928,2.4979,2.5032,2.5085,2.5139,2.5194,2.525,2.5308,2.5367,2.5427,2.5489,2.5553,2.5618,2.5685,2.5753,2.5824,2.5896,2.5971,2.6047,2.6125,2.6206,2.6288,2.6373,2.646,2.6548,2.6639,2.6731,2.6826,2.6922,2.702,2.7119,2.722,2.7322,2.7425,2.753,2.7635,2.7741,2.7847,2.7954,2.806,2.8166,2.8272,2.8377,2.8481,2.8583,2.8684,2.8783,2.8879,2.8973,2.9064,2.9152,2.9237,2.9318,2.9394,2.9467,2.9534,2.9597,2.9655,2.9707,2.9754,2.9794,2.9829,2.9857,2.9879,2.9895,2.9903,2.9905,2.99,2.9888,2.9869,2.9843,2.981,2.977,2.9723,2.9669,2.9608,2.9541,2.9467,2.9386,2.93,2.9207,2.9109,2.9004,2.8895,2.878,2.866,2.8535,2.8406,2.8272,2.8135,2.7994,2.7849,2.7701,2.755,2.7397,2.7241,2.7082,2.6922,2.676,2.6596,2.6431,2.6264,2.6096,2.5928,2.5758,2.5588,2.5418,2.5247,2.5075,2.4903,2.4731,2.4559,2.4386,2.4213,2.404,2.3867,2.3693,2.3519,2.3344,2.3169,2.2993,2.2816,2.2638,2.246,2.228,2.2098,2.1915,2.173,2.1543,2.1354,2.1162,2.0968,2.077,2.0569,2.0364,2.0156,1.9942,1.9724,1.9501,1.9272,1.9037,1.8795,1.8546,1.8288,1.8022,1.7746,1.7459,1.7161,1.685,1.6525,1.6184,1.5825,1.5446,1.5044,1.4617,1.4158,1.3665,1.3127,1.2543,1.1878,1.1167,1.0246,0.9388,0.7703,0.6634,0.6517,0.6413,0.6319,0.6233,0.6155,0.6083,0.6017,0.5956,0.59,0.5847,0.5798,0.5752,0.5709,0.5668,0.563,0.5594,0.556,0.5528,0.5497,0.5468,0.5441,0.5414,0.539,0.5341])
        u_trial = jnp.array([2.5724e-04,7.6986e-04,1.2803e-03,1.7888e-03,2.2952e-03,2.7997e-03,3.3022e-03,3.8027e-03,4.3012e-03,4.7977e-03,5.2922e-03,5.7845e-03,6.2748e-03,6.7628e-03,7.2487e-03,7.7323e-03,8.2136e-03,8.6925e-03,9.1688e-03,9.6427e-03,1.0114e-02,1.0582e-02,1.1048e-02,1.1511e-02,1.1970e-02,1.2427e-02,1.2880e-02,1.3330e-02,1.3777e-02,1.4220e-02,1.4659e-02,1.5095e-02,1.5527e-02,1.5955e-02,1.6379e-02,1.6799e-02,1.7215e-02,1.7627e-02,1.8035e-02,1.8439e-02,1.8839e-02,1.9235e-02,1.9626e-02,2.0014e-02,2.0398e-02,2.0778e-02,2.1154e-02,2.1527e-02,2.1896e-02,2.2262e-02,2.2626e-02,2.2986e-02,2.3343e-02,2.3699e-02,2.4052e-02,2.4403e-02,2.4753e-02,2.5102e-02,2.5450e-02,2.5797e-02,2.6144e-02,2.6491e-02,2.6839e-02,2.7188e-02,2.7538e-02,2.7890e-02,2.8244e-02,2.8600e-02,2.8960e-02,2.9323e-02,2.9689e-02,3.0060e-02,3.0436e-02,3.0817e-02,3.1203e-02,3.1595e-02,3.1994e-02,3.2399e-02,3.2811e-02,3.3231e-02,3.3659e-02,3.4094e-02,3.4539e-02,3.4992e-02,3.5455e-02,3.5927e-02,3.6409e-02,3.6901e-02,3.7403e-02,3.7916e-02,3.8441e-02,3.8976e-02,3.9523e-02,4.0081e-02,4.0651e-02,4.1233e-02,4.1827e-02,4.2434e-02,4.3052e-02,4.3683e-02,4.4327e-02,4.4983e-02,4.5652e-02,4.6333e-02,4.7027e-02,4.7734e-02,4.8453e-02,4.9186e-02,4.9931e-02,5.0688e-02,5.1459e-02,5.2242e-02,5.3038e-02,5.3846e-02,5.4667e-02,5.5501e-02,5.6348e-02,5.7207e-02,5.8080e-02,5.8965e-02,5.9863e-02,6.0774e-02,6.1699e-02,6.2638e-02,6.3590e-02,6.4556e-02,6.5537e-02,6.6533e-02,6.7544e-02,6.8572e-02,6.9616e-02,7.0677e-02,7.1756e-02,7.2854e-02,7.3972e-02,7.5111e-02,7.6272e-02,7.7456e-02,7.8665e-02,7.9901e-02,8.1165e-02,8.2459e-02,8.3786e-02,8.5147e-02,8.6545e-02,8.7984e-02,8.9466e-02,9.0995e-02,9.2576e-02,9.4211e-02,9.5908e-02,9.7670e-02,9.9505e-02,1.0142e-01,1.0342e-01,1.0552e-01,1.0773e-01,1.1006e-01,1.1253e-01,1.1515e-01,1.1794e-01,1.2093e-01,1.2415e-01,1.2764e-01,1.3144e-01,1.3561e-01,1.4024e-01,1.4540e-01,1.5125e-01,1.5793e-01,1.6583e-01,1.7501e-01,1.8700e-01,2.0029e-01,2.2427e-01,2.4175e-01,3.2948e-01,3.3756e-01,3.4525e-01,3.5258e-01,3.5961e-01,3.6636e-01,3.7288e-01,3.7918e-01,3.8528e-01,3.9120e-01,3.9696e-01,4.0258e-01,4.0805e-01,4.1340e-01,4.1864e-01,4.2376e-01,4.2879e-01,4.3372e-01,4.3857e-01,4.4333e-01,4.4802e-01,4.5263e-01,4.5718e-01,4.6166e-01,4.6605e-01])
        
        u_end, h_end, us, hs, bmr, _, evals, evecs, gnd_mask = solve_and_evolve_initial(u_trial, h_trial, bmr, rf)

    else:
        u_end, h_end, us, hs, bmr, _, evals, evecs, gnd_mask = solve_and_evolve(u_trial, h_trial, bmr, rf)

    u_trial = u_end.copy()
    h_trial = h_end.copy()

    s_gnd = h_end + b
    s_flt = h_end*(1-0.917/1.027)

    ffi = np.where(s_gnd<s_flt)[0][0]

    #evals = evals[:ffi]
    #evecs = evecs[:ffi, :ffi]


    if len(evals)==0:
        plt.plot(largest_evals)
        plt.show()
        plt.plot(smallest_evals)
        plt.show()

        #plotboths(steady_state_hs, steady_state_us, k)
        plotgeoms(steady_state_hs, k)
    
    #print(evals[-1])

    steady_state_us.append(u_end)
    steady_state_hs.append(h_end)

    u_trial = u_end.copy()
    h_trial = h_end.copy()
    
    all_evals.append(evals)
    largest_evals.append(evals[-1])
    smallest_evals.append(evals[0])


    #plt.imshow(jnp.rot90(jnp.real(evecs)), vmin=-0.1, vmax=0.1, cmap="RdBu_r")
    #plt.show()


    #plotboths(hs[::2, :], us[::2, :], n_timesteps)

#print(h_end)
#print(u_end)
#raise


#plt.imshow(jnp.real(pad_arrays_with_nan(all_evals)))
#plt.show()

plotgeoms(steady_state_hs, k)
plt.plot(largest_evals)
plt.show()
plt.plot(smallest_evals)
plt.show()





raise






##Plotting evolution of largest singular values for steady-state solutions:
#
##try a sharper retrograde bed:
#
#
#largest_evals = []
#smallest_evals = []
#steady_state_us = []
#steady_state_hs = []
#bmr = jnp.zeros_like(x)
#timestep = 1
#n_iterations = 10
#n_timesteps = 4
#
#accumulation_scaled = (jnp.zeros_like(x)+0.5)/timestep
#
#bmr = jnp.zeros_like(x)+0.0
#
#for k in range(12):
##for k, acc in enumerate([(jnp.zeros_like(x)+0.05-0.001*i)/timestep for i in range(50)]):
#    print(k)
#
#    #bmr = (accumulation_scaled.copy()*k)/2
#    acc_scaled_new = accumulation_scaled*(1-(k/10))
#
#    solve_and_evolve = implicit_coupled_solver_compiled(mu, beta, acc_scaled_new, timestep, n_iterations, n_timesteps, compute_eigenvalues=False, compute_singular_values=True)
#    u_end, h_end, us, hs, bmrs, svs, rsvecs, _,_,_,_ = solve_and_evolve(u_trial, h_trial, bmr)
#    #u_end, h_end, us, hs, bmrs= solve_and_evolve(u_trial, h_trial, bmr)
#
#
#    #plotboths(hs, us, n_timesteps)
#    #raise
#   
#    u_trial = u_end.copy()
#    h_trial = h_end.copy()
#
#
#    steady_state_hs.append(h_end)
#    steady_state_us.append(u_end)
#    
#    if True:
#        s_gnd = h_end + b
#        s_flt = h_end*(1-0.917/1.027)
#    
#        ffi = np.where(s_gnd<s_flt)[0][0]
#    
#        svs = svs[:ffi]
#        rsvecs = rsvecs[:ffi, :ffi]
#        
#    
#    
#        if len(svs)==0:
#            plt.plot(largest_evals)
#            plt.show()
#    
#            plotgeoms(steady_state_hs, k)
#        
#        print(svs[-1])
#    
#        u_trial = u_end.copy()
#        h_trial = h_end.copy()
#    
#        largest_evals.append(svs[-1])
#        smallest_evals.append(svs[0])
#    
#    
#        #plt.imshow(jnp.rot90(jnp.real(evecs)), vmin=-0.1, vmax=0.1, cmap="RdBu_r")
#        #plt.show()
#    
#    
#        #plotboths(hs[::2, :], us[::2, :], n_timesteps)
#
#
#plotgeoms(steady_state_hs, k)
#
#plt.plot(largest_evals)
#plt.show()
#plt.plot(smallest_evals)
#plt.show()
#
#
#
#
#
#raise










##Plotting evolution of largest eigenvalues for steady-state solutions:
#largest_evals = []
#smallest_evals = []
#steady_state_us = []
#steady_state_hs = []
#bmr = jnp.zeros_like(x)
#timestep = 1
#n_iterations = 10
#n_timesteps = 100
#
#for k, acc in enumerate([(jnp.zeros_like(x)+0.1-0.001*i)/timestep for i in range(50)]):
#    print(k)
#
#    accumulation_scaled = acc/timestep
#
#    solve_and_evolve = implicit_coupled_solver_compiled(mu, beta, accumulation_scaled, timestep, n_iterations, n_timesteps, compute_eigenvalues=True)
#    u_end, h_end, us, hs, bmrs, evals, evecs, gnd_mask = solve_and_evolve(u_trial, h_trial, bmr)
#   
#    u_trial = u_end.copy()
#    h_trial = h_end.copy()
#
#
#    s_gnd = h_end + b
#    s_flt = h_end*(1-0.917/1.027)
#
#    ffi = np.where(s_gnd<s_flt)[0][0]
#
#    evals = evals[:ffi]
#    evecs = evecs[:ffi, :ffi]
#
#
#    if len(evals)==0:
#        plt.plot(largest_evals)
#        plt.show()
#        plt.plot(smallest_evals)
#        plt.show()
#
#        #plotboths(steady_state_hs, steady_state_us, k)
#        plotgeoms(steady_state_hs, k)
#    
#    print(evals[-1])
#
#    steady_state_us.append(u_end)
#    steady_state_hs.append(h_end)
#
#    u_trial = u_end.copy()
#    h_trial = h_end.copy()
#    
#
#    largest_evals.append(evals[-1])
#    smallest_evals.append(evals[0])
#
#
#    #plt.imshow(jnp.rot90(jnp.real(evecs)), vmin=-0.1, vmax=0.1, cmap="RdBu_r")
#    #plt.show()
#
#
#    #plotboths(hs[::2, :], us[::2, :], n_timesteps)
#
#plotgeoms(steady_state_hs, k)
#plt.plot(largest_evals)
#plt.show()
#plt.plot(smallest_evals)
#plt.show()
#
#
#
#
#
#raise





#######WORKING STUFF LOOKING AT EIGENVALUES, NO JIT COMPILATION SO IT'S SLOW:




#initialise the thing in steady state:


#bmr = jnp.zeros_like(x)
#timestep = 0
#n_iterations = 10
#n_timesteps = 5
#
#
#acc = jnp.zeros_like(x)+0.08
#
#accumulation_scaled = acc/timestep
#
#solve_and_evolve = implicit_coupled_solver_compiled(mu, beta, accumulation_scaled, timestep, n_iterations, n_timesteps, compute_eigenvalues=True)
#u_ss, h_ss, us, hs, _,_,_,_ = solve_and_evolve(u_trial, h_trial, bmr)
#   
#plotboths(hs, us, n_timesteps)
#
#raise








#######STEADY STATE ON PROGRADE SLOPE:
#
#
#timestep = 1
##basal_melt_rate = 0.1/timestep
#basal_melt_rate = 0
#accumulation = jnp.zeros_like(h)+0.0025
#
#n_timesteps = 1000
#n_iterations = 5
#
#h_trial = jnp.zeros_like(x)+0.5
#b = 0.2 - 0.5*x
#
##plotgeom(h_trial)
#
#solve_and_evolve = implicit_coupled_solver_compiled(mu, beta, accumulation, timestep, n_iterations, n_timesteps, compute_eigenvalues=False)
#u_ss, h_ss, us, hs, _ = solve_and_evolve(u_trial, h_trial, jnp.zeros_like(x))
##newton_solve = implicit_coupled_solver(u_ss, h_ss, accumulation, timestep, basal_melt_rate, 20, n_timesteps, compute_evals=True)
##newton_solve = implicit_coupled_solver(u_trial, h_trial, accumulation, timestep, basal_melt_rate, 20, n_timesteps, compute_evals=False)
##u_end, h_end, hs, us, evals, leading_evecs, evals_ptl, leading_evecs_ptl, evals_fdbk, leading_evecs_fdbk = newton_solve(mu)
#
##plotgeom(h_ss)
#plotgeoms(hs[::100], n_timesteps)
##plotboths(hs, us, n_timesteps)
#
#raise






######LOOKING AT SINGULAR VALUE EVOLUTION

timestep = 1
#basal_melt_rate = 0.1/timestep
basal_melt_rate = 0
accumulation = jnp.zeros_like(h)+0.01

n_timesteps = 51


#newton_solve = implicit_coupled_solver(u_ss, h_ss, accumulation, timestep, basal_melt_rate, 20, n_timesteps, compute_evals=True)
newton_solve = implicit_coupled_solver(u_trial, h_trial, accumulation, timestep, basal_melt_rate, 10, n_timesteps, compute_evals=True)
u_end, h_end, hs, us, evals, leading_evecs, evals_ptl, leading_evecs_ptl, evals_fdbk, leading_evecs_fdbk = newton_solve(mu)



#leading_evec_angles = [jnp.dot(e1, e2) for e1, e2 in zip(leading_evecs[1:], leading_evecs[:-1])]
#plt.plot(leading_evec_angles)
#plt.show()

cmap = cm.RdBu_r.copy()
cmap.set_bad(color='fuchsia')
plt.imshow(pad_arrays_with_nan(leading_evecs), vmin=-1, vmax=1, cmap=cmap)
plt.show()


plt.plot(evals)
plt.show()

plt.plot(evals_ptl)
plt.show()

plt.plot(evals_fdbk)
plt.show()

plotgeoms(hs, n_timesteps)
#plotboths(hs[::10], us[::10], n_timesteps)

raise


all_evals = jnp.real(all_evals)
all_evals_ptl = jnp.real(all_evals_ptl)



plt.imshow(jnp.transpose(all_evals), vmin=-100, vmax=100, cmap="RdBu_r")
plt.colorbar()
plt.show()

plt.imshow(jnp.transpose(all_evals_ptl), vmin=-100, vmax=100, cmap="RdBu_r")
plt.colorbar()
plt.show()

plt.imshow(jnp.transpose(all_evals_fdbk), vmin=-100, vmax=100, cmap="RdBu_r")
plt.colorbar()
plt.show()

plt.imshow(jnp.transpose(all_evals)-jnp.transpose(all_evals_ptl), vmin=-30, vmax=30, cmap="RdBu_r")
plt.colorbar()
plt.show()

plt.plot(evals)
plt.show()

plt.plot(evals_ptl)
plt.show()

plt.plot(evals_fdbk)
plt.show()

plotboths(hs[::5], us[::5], n_timesteps) 


##NEWTON
#newton_solve = make_u_solver(h, beta, 20)
#u_end, us, mom_res_jac = newton_solve(u_trial, mu)


#evals, evecs = jnp.linalg.eig(mom_res_jac)

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




