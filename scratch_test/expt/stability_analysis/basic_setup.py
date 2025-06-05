import jax
import jax.numpy as jnp
from jax import jacfwd, jacrev
import jax.scipy.linalg as lalg
from jax.scipy.optimize import minimize

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


#np.set_printoptions(precision=3, suppress=False, linewidth=np.inf)

def make_nonlinear_momentum_residual(beta, mu_cc):
    #NOTE: taken out the nonlinearity

    mu_face = jnp.zeros((n+1,))
    mu_face = mu_face.at[1:n].set(0.5 * (mu_cc[:n-1] + mu_cc[1:n]))
    #mu_face = mu_face.at[-1].set(mu_face[-2])
    mu_face = mu_face.at[0].set(mu_cc[1])

    
    def mom_res(u, h):
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


        if compute_eigenvalues:
            
            H = make_adv_rhs(accumulation)
            H_jac = jacfwd(H, argnums=(0,1))(u, h, bmr)

            visc_jac = visc_jac_fn(u, h)

            L, fdbk, dHdh = construct_tangent_propagator(H_jac[1],\
                                             H_jac[0],\
                                             visc_jac[1],\
                                             visc_jac[0])



            s_gnd = h + b
            s_flt = h*(1-0.917/1.027)


            mask = jnp.where(s_gnd>s_flt, 1, 0)
            mask = jnp.outer(mask, mask)

            L_cr = L * mask

            #ffi = np.where(s_gnd<s_flt)[0][0]
            #L_cr = L[:ffi, :ffi]

            
            evals, evecs = jnp.linalg.eig(L_cr)
            order_indices = jnp.argsort(evals)
            evals_ord = evals[order_indices]
            evecs_ord = evecs[:, order_indices]

            
            #evals, evecs = jnp.linalg.eig(fdbk)
            #evecs, evals, _ = jnp.linalg.svd(L) #I know they're not eigen-...!
            
            #evecs_ptl, evals_ptl, _ = jnp.linalg.svd(H_jac[1][:n-1, :n-1])

            #indices = jnp.argsort(evals)
            #evals_ordered = evals[indices]
            #evecs_ordered = evecs[:,indices]

            #indices_ptl = jnp.argsort(evals_ptl)
            #evals_ordered_ptl = evals[indices_ptl]
            #evecs_ordered_ptl = evecs[:,indices_ptl]

            return u, h, us, hs, bmrs, evals_ord, evecs_ord, mask
        else:
            return u, h, us, hs, bmrs

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

        hs = []
        us = []
        largest_evals = []
        all_evals = []
        largest_evals_ptl = []
        all_evals_ptl = []
        largest_evals_fdbk = []
        all_evals_fdbk = []

        for j in range(num_timesteps):

            print(j)
            for i in range(num_iterations):
                mom_res_jac = mom_res_jac_fn(u, h)
                adv_jac = adv_jac_fn(u, h, h_old, basal_melt_rate)
        
                full_jacobian = jnp.block(
                                          [[mom_res_jac[0], mom_res_jac[1]],
                                          [adv_jac[0], adv_jac[1]]]
                                          )

#                print(np.array(mom_res_jac[0]))
#                print("-------------------")
#                print("-------------------")
#                print("-------------------")
#                print(np.array(adv_jac[0]))
#                print("-------------------")
#                print("-------------------")
#                print("-------------------")
#                print(np.array(mom_res_jac[1]))
#                print("-------------------")
#                print("-------------------")
#                print("-------------------")
#                print(np.array(adv_jac[1]))
#                print("-------------------")
#                print("-------------------")
#                print("-------------------")
#                print(np.array(full_jacobian))
#                raise


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


                L_cr = L[:ffi, :ffi]



                #eigenvalues
                evals, evecs = jnp.linalg.eig(L_cr)
                order_indices = jnp.argsort(evals)
                evals_ord = evals[order_indices]
                evecs_ord = evecs[:, order_indices]


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

        return u, h, hs, us, evals_ord, evecs_ord

        #return u, h, hs, us, largest_evals, jnp.array(all_evals), \
        #       largest_evals_ptl, jnp.array(all_evals_ptl), \
        #       largest_evals_fdbk, jnp.array(all_evals_fdbk)

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







rho = 1
g = 1

lx = 1
n = 100
dx = lx/(n-1)
x = jnp.linspace(0,lx,n)

mu = jnp.zeros((n,)) + 1


#OVERDEEPENED BED

h = 1.5*jnp.exp(-2*x*x*x) #grounded sections
#h = jnp.ones_like(x)*0.5 #floating uniform thk

#h = 1.75*jnp.exp(-2*x*x) #just a bit of an odd shaped ice shelf
#h = (1+jnp.zeros((n,)))*(1-x/2)
#h = h.at[-1].set(0)


#b_intermediate = jnp.zeros((n,))-0.5
b_intermediate = -0.3 - 0.3*x.copy()

s_gnd = b_intermediate + h
s_flt = h*(1-0.917/1.027)
s = jnp.maximum(s_gnd, s_flt)
s = s.at[-1].set(0)


#b = jnp.zeros((n,))-0.5
#b = b.at[:n].set(b[:n] - 0.15*jnp.exp(-(5*x-2)**2))

b = -0.3 - 0.3*x.copy()
#b = b.at[:n].set(b[:n] - 0.15*jnp.exp(-(5*x-2)**2))
b = b.at[:n].set(b[:n] - 0.35*jnp.exp(-(5*x-2)**2))


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



accumulation = jnp.zeros_like(h)+0.05
#accumulation = jnp.zeros_like(h)

#plotgeom(h)
#raise


#u_trial = jnp.exp(x)-1
#u_trial = x.copy()
u_trial = jnp.zeros_like(x)
h_trial = h.copy()



##TESTING PICARD ITERATION:
iterator = make_picard_iterator_for_u(mu, h_trial, beta, 20)
u_end, us = iterator(u_trial)
#print("change to newton")
#iterator = make_u_solver(mu, h_trial, beta, 5)
#u_end_2, us_2 = iterator(u_trial)

plotboth(h, u_end)

plt.figure(figsize=(10,5))
for i, u_ in enumerate(us+us_2):
    plt.plot(u_, label=str(i))
plt.legend()
plt.show()

raise



############TRYING TO GET THE JIT COMPILATION GOING:
#
n_timesteps = 30
n_iterations = 20
timestep = 1
#
#bmr_init = jnp.zeros_like(accumulation)+0.005
#
#accumulation = accumulation/timestep
#
#
##bmr_init = bmr_init/timestep
##
##solve_and_evolve = implicit_coupled_solver_compiled(mu, beta, accumulation, timestep, n_iterations, n_timesteps, compute_eigenvalues=True)
##u_end, h_end, us, hs, bmrs, evals, evecs = solve_and_evolve(u_trial, h_trial, bmr_init)
##
##print(evals)
##
##plotboths(hs[:, :], us[:, :], n_timesteps)
##
##raise
#
#
#
#
#bmr_init = jnp.zeros_like(accumulation)
#
#solve_and_evolve = implicit_coupled_solver_compiled(mu, beta, accumulation, timestep, n_iterations, n_timesteps, compute_eigenvalues=True)
#u_end, h_end, us, hs, bmrs, evals, evecs = solve_and_evolve(u_trial, h_trial, bmr_init)
#
#largest_eval = evals[-1]
#largest_evec = evecs[-1]
#
#plotboths(hs[:, :], us[:, :], n_timesteps)
#
#u_end_2, h_end_2, us_2, hs_2, bmrs_2, evals_2, evecs_2 = solve_and_evolve(u_end, h_end+0.2*jnp.real(largest_evec), bmr_init)
#
#plotboths(hs_2[:,:], us_2[:,:], n_timesteps)
#
#plt.plot(h_end)
#plt.plot(h_end_2)
#plt.show()
#
#
#raise


##Plotting evolution of largest eigenvalues for steady-state solutions:
largest_evals = []
smallest_evals = []
steady_state_us = []
steady_state_hs = []
bmr = jnp.zeros_like(x)
for k, acc in enumerate([(jnp.zeros_like(x)+0.05-0.004*i)/timestep for i in range(30)]):
    print(k)

    accumulation_scaled = acc/timestep

    solve_and_evolve = implicit_coupled_solver_compiled(mu, beta, accumulation_scaled, timestep, n_iterations, n_timesteps, compute_eigenvalues=True)
    u_end, h_end, us, hs, bmrs, evals, evecs, gnd_mask = solve_and_evolve(u_trial, h_trial, bmr)
   
    u_trial = u_end.copy()
    h_trial = h_end.copy()


    s_gnd = h_end + b
    s_flt = h_end*(1-0.917/1.027)

    ffi = np.where(s_gnd<s_flt)[0][0]

    evals = evals[:ffi]
    evecs = evecs[:ffi, :ffi]

    if len(evals)==0:
        plt.plot(largest_evals)
        plt.show()
        plt.plot(smallest_evals)
        plt.show()

        plotboths(steady_state_hs, steady_state_us, k)

    steady_state_us.append(u_end)
    steady_state_hs.append(h_end)

    u_trial = u_end.copy()
    h_trial = h_end.copy()
    
    print(evals[-1])

    largest_evals.append(evals[-1])
    smallest_evals.append(evals[0])


    #plt.imshow(jnp.rot90(jnp.real(evecs)), vmin=-0.1, vmax=0.1, cmap="RdBu_r")
    #plt.show()


    #plotboths(hs[::2, :], us[::2, :], n_timesteps)


plt.plot(largest_evals)
plt.show()
plt.plot(smallest_evals)
plt.show()





raise




#######WORKING STUFF LOOKING AT EIGENVALUES, NO JIT COMPILATION SO IT'S SLOW:

timestep = 1
#basal_melt_rate = 0.1/timestep
basal_melt_rate = 0
accumulation = jnp.zeros_like(h)+0.01

n_timesteps = 25
newton_solve = implicit_coupled_solver(u_trial, h_trial, accumulation, timestep, basal_melt_rate, 20, n_timesteps, compute_evals=True)
u_end, h_end, hs, us, evals, all_evals, evals_ptl, all_evals_ptl, evals_fdbk, all_evals_fdbk = newton_solve(mu)


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

plotboths(hs[:], us[:], n_timesteps) 


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




