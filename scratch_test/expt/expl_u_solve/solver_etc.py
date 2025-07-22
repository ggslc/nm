import jax
import jax.numpy as jnp
from jax import jacfwd, jacrev
import jax.scipy.linalg as lalg
from jax.scipy.optimize import minimize

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


np.set_printoptions(precision=10, suppress=False, linewidth=np.inf, threshold=np.inf)

def make_linear_momentum_residual_osd_at_gl():

    def mom_res(u, h, mu_face, beta):
        s_gnd = h + b
        s_flt = h*(1-rho/rho_w)
        s = jnp.maximum(s_gnd, s_flt)
        #ffci = jnp.where(s_flt>s_gnd)[0][0] #first_floating_cell_index
        
        is_floating = s_flt > s_gnd
        ffci = jnp.where(jnp.any(is_floating), jnp.argmax(is_floating), n)


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
      
        #one-sided differences at gl
        h_grad_s = h_grad_s.at[ffci].set(h[ffci] * (s[ffci+1] - s[ffci]))
        h_grad_s = h_grad_s.at[ffci-1].set(h[ffci-1] * (s[ffci-1] - s[ffci-2]))
     
        #scale
        h_grad_s = rho * g * h_grad_s

        #print(flux)
        #print(sliding)
        #print(h_grad_s)

        #plt.plot(-h_grad_s)
        #plt.plot(sliding)
        #plt.plot(flux)
        #plt.show()
        
        return flux[1:] - flux[:-1] - h_grad_s - sliding
        #return - h_grad_s - sliding
        #return flux[1:] - flux[:-1] - sliding

    return mom_res


def make_nonlinear_momentum_residual_osd_at_gl(C):

    def mom_res(u, h):
        s_gnd = h + b
        s_flt = h*(1-rho/rho_w)
        s = jnp.maximum(s_gnd, s_flt)
        #ffci = jnp.where(s_flt>s_gnd)[0][0] #first_floating_cell_index
        
        is_floating = s_flt > s_gnd
        ffci = jnp.where(jnp.any(is_floating), jnp.argmax(is_floating), n)

        
        grounded_mask = jnp.where((b+h)<(h*(1-rho/rho_w)), 0, 1)
        beta_int = C * (1/(jnp.abs(u)**(2/3) + (1e-8)**(2/3))) * grounded_mask

        dudx = jnp.zeros((n+1,))
        dudx = dudx.at[1:n].set((u[1:n] - u[:n-1])/dx)
        #dudx = dudx.at[-2].set(dudx[-3])
        dudx = dudx.at[-1].set(0)
        ##set (or 'use' I guess) reflection boundary condition
        dudx = dudx.at[0].set(2*u[0]/dx)


        mu_face = B * (jnp.abs(dudx)+epsilon_visc)**(-2/3)


        sliding = beta_int * u * dx
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
      
        #one-sided differences at gl
        h_grad_s = h_grad_s.at[ffci].set(h[ffci] * (s[ffci+1] - s[ffci]))
        h_grad_s = h_grad_s.at[ffci-1].set(h[ffci-1] * (s[ffci-1] - s[ffci-2]))
     
        #scale
        h_grad_s = rho * g * h_grad_s

        #print(flux)
        #print(sliding)
        #print(h_grad_s)
        #raise

        #plt.plot(-h_grad_s)
        #plt.plot(sliding)
        #plt.plot(flux)
        #plt.show()
        
        return flux[1:] - flux[:-1] - h_grad_s - sliding
        #return - h_grad_s - sliding
        #return flux[1:] - flux[:-1] - sliding

    return mom_res


def make_picard_iterator_for_u(C, iterations):

    def new_mu(u):
    
        dudx = jnp.zeros((n+1,))
        dudx = dudx.at[1:-1].set((u[1:] - u[:-1])/dx)
        dudx = dudx.at[-1].set(dudx[-2])
        #set reflection boundary condition
        dudx = dudx.at[0].set(2*u[0]/dx)
    
        mu_nl = B * (jnp.abs(dudx)+epsilon_visc)**(-2/3)

        #mu_nl = B * (epsilon_visc**(-2/3))

        return mu_nl

    def new_beta(u, h):

        grounded_mask = jnp.where((b+h)<(h*(1-rho/rho_w)), 0, 1)

        #beta = C * ((jnp.abs(u))**(-2/3)) * grounded_mask
        beta = C * (1/(jnp.abs(u)**(2/3) + (1e-8)**(2/3))) * grounded_mask
        #beta = C * (1e-6)**(-2/3) * grounded_mask

        return beta
    

    def iterator(u_init, h):    

        mom_res = make_linear_momentum_residual_osd_at_gl()

        #for debugging:
        #beta = new_beta(u_init)
        #mu = new_mu(u_init)
        #mom_res(u_init, h, mu, beta)
        #raise
        #mom_res_nl = make_nonlinear_momentum_residual(beta, mu_centres_zero)
        
        jac_mom_res_fn = jacfwd(mom_res, argnums=0)


        u = u_init.copy()

        prev_residual = 1

        us = [u]

        for i in range(iterations):

            beta = new_beta(u, h)
            #print(beta)
            
            mu = new_mu(u)
            #print(mu)
            
            residual = jnp.max(jnp.abs(mom_res(u, h, mu, beta))) 
            #residual = jnp.max(jnp.abs(mom_res_nl(u, h)))
            print(residual, prev_residual/residual)

            
            jac_mom_res = jac_mom_res_fn(u, h, mu, beta)


            du = lalg.solve(jac_mom_res, -mom_res(u, h, mu, beta))

            #print(jac_mom_res@du + -mom_res(u, h, mu))

            u = u + du

            us.append(u)
            
            #if (residual < 1e-17) or (jnp.abs(residual-prev_residual) < 1e-18):
            #    break

            prev_residual = residual

        #residual = jnp.max(jnp.abs(mom_res(u, h, mu)))
        #print(residual)

        return u, us
       
    return iterator


def make_adv_residual(dt, accumulation):
    
    def adv_res(u, h, h_old, basal_melt_rate):
        s_gnd = h + b
        s_flt = h*(1-rho/rho_w)

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

    return adv_res    

def make_picard_iterator_for_joint_impl_problem_alt_compiled(C, iterations, dt, bmr):
    
    mom_res = make_linear_momentum_residual_osd_at_gl()
    adv_res = make_adv_residual(dt, accumulation)

    jac_mom_res_fn = jacfwd(mom_res, argnums=(0,1))
    jac_adv_res_fn = jacfwd(adv_res, argnums=(0,1))
    
    def new_mu(u):
    
        dudx = jnp.zeros((n+1,))
        dudx = dudx.at[1:-1].set((u[1:] - u[:-1])/dx)
        dudx = dudx.at[-1].set(dudx[-2])
        #set reflection boundary condition
        dudx = dudx.at[0].set(2*u[0]/dx)
    
        mu_nl = B * (jnp.abs(dudx)+epsilon_visc)**(-2/3)

        #mu_nl = B * (epsilon_visc**(-2/3))

        return mu_nl

    def new_beta(u, h):

        grounded_mask = jnp.where((b+h)<(h*(1-rho/rho_w)), 0, 1)

        #beta = C * ((jnp.abs(u))**(-2/3)) * grounded_mask
        beta = C * (1/(jnp.abs(u)**(2/3) + (1e-8)**(2/3))) * grounded_mask
        #beta = C * (1e-6)**(-2/3) * grounded_mask

        return beta

    def continue_condition(state):
        _,_,_, i,res,resrat = state
        return i<iterations


    def step(state):
        u, h, h_init, i, prev_res, prev_resrat = state

        beta = new_beta(u, h)
        mu = new_mu(u)

        jac_mom_res = jac_mom_res_fn(u, h, mu, beta)
        jac_adv_res = jac_adv_res_fn(u, h, h_init, bmr)

        full_jacobian = jnp.block(
                                  [ [jac_mom_res[0], jac_mom_res[1]],
                                    [jac_adv_res[0], jac_adv_res[1]] ]
                                  )
    
        rhs = jnp.concatenate((-mom_res(u, h, mu, beta), -adv_res(u, h, h_init, bmr)))
    
        dvar = lalg.solve(full_jacobian, rhs)
    
        u = u.at[:].set(u+dvar[:n])
        h = h.at[:].set(h+dvar[n:])

        #TODO: add one for the adv residual too...
        res = jnp.max(jnp.abs(mom_res(u, h, mu, beta))) 

        return u, h, h_init, i+1, res, prev_res/res


    def iterator(u_init, h_init):    

        resrat = np.inf
        res = np.inf

        initial_state = u_init, h_init, h_init, 0, res, resrat

        u, h, h_init, itn, res, resrat = jax.lax.while_loop(continue_condition, step, initial_state)

        return u, h, res
       
    return iterator


def make_picard_iterator_for_joint_impl_problem_alt(C, iterations, dt, bmr):

    def new_mu(u):
    
        dudx = jnp.zeros((n+1,))
        dudx = dudx.at[1:-1].set((u[1:] - u[:-1])/dx)
        dudx = dudx.at[-1].set(dudx[-2])
        #set reflection boundary condition
        dudx = dudx.at[0].set(2*u[0]/dx)
    
        mu_nl = B * (jnp.abs(dudx)+epsilon_visc)**(-2/3)

        #mu_nl = B * (epsilon_visc**(-2/3))

        return mu_nl

    def new_beta(u, h):

        grounded_mask = jnp.where((b+h)<(h*(1-rho/rho_w)), 0, 1)

        #beta = C * ((jnp.abs(u))**(-2/3)) * grounded_mask
        beta = C * (1/(jnp.abs(u)**(2/3) + (1e-8)**(2/3))) * grounded_mask
        #beta = C * (1e-6)**(-2/3) * grounded_mask

        return beta
    

    def iterator(u_init, h_init):    

        mom_res = make_linear_momentum_residual_osd_at_gl()
        adv_res = make_adv_residual(dt, accumulation)

        #for debugging:
        #beta = new_beta(u_init)
        #mu = new_mu(u_init)
        #mom_res(u_init, h, mu, beta)
        #raise
        #mom_res_nl = make_nonlinear_momentum_residual(beta, mu_centres_zero)
        
        jac_mom_res_fn = jacfwd(mom_res, argnums=(0,1))
        jac_adv_res_fn = jacfwd(adv_res, argnums=(0,1))

        u = u_init.copy()
        h = h_init.copy()

        prev_res = jnp.inf
        prev_res_adv = jnp.inf

        us = [u]
        hs = [h]

        for i in range(iterations):

            #update beta
            beta = new_beta(u, h)
            #print(beta)
            
            #update mu
            mu = new_mu(u)
            #print(mu)


            jac_mom_res = jac_mom_res_fn(u, h, mu, beta)
            jac_adv_res = jac_adv_res_fn(u, h, h_init, bmr)

            full_jacobian = jnp.block(
                                      [ [jac_mom_res[0], jac_mom_res[1]],
                                        [jac_adv_res[0], jac_adv_res[1]] ]
                                      )
    
            rhs = jnp.concatenate((-mom_res(u, h, mu, beta), -adv_res(u, h, h_init, bmr)))
    
            dvar = lalg.solve(full_jacobian, rhs)
    
            u = u.at[:].set(u+dvar[:n])
            h = h.at[:].set(h+dvar[n:])
            

            #calculate u given those
            residual = jnp.max(jnp.abs(mom_res(u, h, mu, beta))) 
            print(residual, prev_res/residual)
            
            #if (residual < 1e-17) or (jnp.abs(residual-prev_residual) < 1e-18):
            #    break

            prev_res = residual

        #residual = jnp.max(jnp.abs(mom_res(u, h, mu)))
        #print(residual)

        return u, us, h, hs
       
    return iterator

def make_newton_iterator_for_joint_impl_problem(C, iterations, dt, bmr, accumulation):

    mom_res = make_nonlinear_momentum_residual_osd_at_gl(C)
    adv_res = make_adv_residual(dt, accumulation)
        
    jac_mom_res_fn = jacfwd(mom_res, argnums=(0,1))
    jac_adv_res_fn = jacfwd(adv_res, argnums=(0,1))
    
    def iterator(u_init, h_init):

        u = u_init.copy()
        h = h_init.copy()

        us = [u]
        hs = [h]

        for i in range(iterations):

            jac_mom_res = jac_mom_res_fn(u, h)
            jac_adv_res = jac_adv_res_fn(u, h, h_init, bmr)

            full_jacobian = jnp.block(
                                      [ [jac_mom_res[0], jac_mom_res[1]],
                                        [jac_adv_res[0], jac_adv_res[1]] ]
                                      )

            print(jac_mom_res[0])
            print(jac_mom_res[1])
            print(jac_adv_res[0])
            print(jac_adv_res[1])
            raise
    
            rhs = jnp.concatenate((-mom_res(u, h), -adv_res(u, h, h_init, bmr)))
    
            dvar = lalg.solve(full_jacobian, rhs)
    
            u = u.at[:].set(u+dvar[:n])
            h = h.at[:].set(h+dvar[n:])
                
            print(jnp.max(jnp.abs(mom_res(u, h))), jnp.max(jnp.abs(adv_res(u, h, h_init, bmr))))

            us.append(u)
            hs.append(h)

        return u, us, h, hs
       
    return iterator

def make_picard_iterator_for_joint_impl_problem(C, iterations, dt, bmr):

    def new_mu(u):
    
        dudx = jnp.zeros((n+1,))
        dudx = dudx.at[1:-1].set((u[1:] - u[:-1])/dx)
        dudx = dudx.at[-1].set(dudx[-2])
        #set reflection boundary condition
        dudx = dudx.at[0].set(2*u[0]/dx)
    
        mu_nl = B * (jnp.abs(dudx)+epsilon_visc)**(-2/3)

        #mu_nl = B * (epsilon_visc**(-2/3))

        return mu_nl

    def new_beta(u, h):

        grounded_mask = jnp.where((b+h)<(h*(1-rho/rho_w)), 0, 1)

        #beta = C * ((jnp.abs(u))**(-2/3)) * grounded_mask
        beta = C * (1/(jnp.abs(u)**(2/3) + (1e-8)**(2/3))) * grounded_mask
        #beta = C * (1e-6)**(-2/3) * grounded_mask

        return beta
    

    def iterator(u_init, h_init):    

        mom_res = make_linear_momentum_residual_osd_at_gl()
        adv_res = make_adv_residual(dt, accumulation)

        #for debugging:
        #beta = new_beta(u_init)
        #mu = new_mu(u_init)
        #mom_res(u_init, h, mu, beta)
        #raise
        #mom_res_nl = make_nonlinear_momentum_residual(beta, mu_centres_zero)
        
        jac_mom_res_fn = jacfwd(mom_res, argnums=0)
        jac_adv_res_fn = jacfwd(adv_res, argnums=1)

        u = u_init.copy()
        h = h_init.copy()

        prev_res = jnp.inf
        prev_res_adv = jnp.inf

        us = [u]
        hs = [h]

        for i in range(iterations):

            #update beta
            beta = new_beta(u, h)
            #print(beta)
            
            #update mu
            mu = new_mu(u)
            #print(mu)
            

            #calculate u given those
            residual = jnp.max(jnp.abs(mom_res(u, h, mu, beta))) 
            print(residual, prev_res/residual)
            
            jac_mom_res = jac_mom_res_fn(u, h, mu, beta)
            du = lalg.solve(jac_mom_res, -mom_res(u, h, mu, beta))
            u = u + du
            us.append(u)


            #calculate h given u
            jac_adv_res = jac_adv_res_fn(u, h, h_init, bmr)
            dh = lalg.solve(jac_adv_res, -adv_res(u, h, h_init, bmr))
            h = h + dh
            hs.append(h)


            
            #if (residual < 1e-17) or (jnp.abs(residual-prev_residual) < 1e-18):
            #    break

            prev_res = residual

        #residual = jnp.max(jnp.abs(mom_res(u, h, mu)))
        #print(residual)

        return u, us, h, hs
       
    return iterator


def construct_tlo(u, h, C, accumulation, bmr):

    mom_res_fn = make_nonlinear_momentum_residual_osd_at_gl(C)
    adv_rhs_fn = make_adv_rhs(accumulation)

    mom_res_jac = jacfwd(mom_res_fn, argnums=(0,1))(u,h)
    adv_rhs_jac = jacfwd(adv_rhs_fn, argnums=(0,1))(u,h,bmr)


    #Naive approach
    dGdu_inv = jnp.linalg.inv(mom_res_jac[0])
    int_ = dGdu_inv @ mom_res_jac[1]
    
    feedback_term = - adv_rhs_jac[0] @ int_
    
    #L = (-dHdu @ int_ + dHdh)
    #print(L-dHdh)
    
    return feedback_term + adv_rhs_jac[1], feedback_term, adv_rhs_jac[1]


def make_picard_iterator_for_u_compiled(C, iterations):

    mom_res = make_linear_momentum_residual_osd_at_gl()
    jac_mom_res_fn = jacfwd(mom_res, argnums=0)
    

    def new_mu(u):
    
        dudx = jnp.zeros((n+1,))
        dudx = dudx.at[1:-1].set((u[1:] - u[:-1])/dx)
        dudx = dudx.at[-1].set(dudx[-2])
        #set reflection boundary condition
        dudx = dudx.at[0].set(2*u[0]/dx)
    
        mu_nl = B * (jnp.abs(dudx)+epsilon_visc)**(-2/3)

        #mu_nl = B * (epsilon_visc**(-2/3))

        return mu_nl

    def new_beta(u, h):

        grounded_mask = jnp.where((b+h)<(h*(1-rho/rho_w)), 0, 1)

        #beta = C * ((jnp.abs(u))**(-2/3)) * grounded_mask
        beta = C * (1/(jnp.abs(u)**(2/3) + (1e-8)**(2/3))) * grounded_mask
        #beta = C * (1e-6)**(-2/3) * grounded_mask

        return beta


    def continue_condition(state):
        _, i, res, res_rat, _ = state
        #return ((i<iterations) | (res_rat<=1))
        return i<iterations


    def step(state):
        u, i, prev_residual, residual_ratio, h = state
        
        beta = new_beta(u, h)
        mu = new_mu(u)

        jac_mom_res = jac_mom_res_fn(u, h, mu, beta)

        u = u + lalg.solve(jac_mom_res, -mom_res(u, h, mu, beta))
        
        residual = jnp.max(jnp.abs(mom_res(u, h, mu, beta)))

        return u, i+1, residual, prev_residual/residual, h


    @jax.jit
    def iterator(u_init, h):

        residual_ratio = jnp.inf
        residual = 1

        initial_state = u_init, 0, residual, residual_ratio, h

        u, its, res, res_rat, h = jax.lax.while_loop(continue_condition, step, initial_state)

        return u, its, res, res_rat, h
       
    return iterator


def make_adv_expt(dt, accumulation):
    
    def adv(u, h_old, basal_melt_rate):
        s_gnd = h_old + b
        s_flt = h_old*(1-rho/rho_w)

        acc = jnp.where(s_gnd<s_flt, accumulation-basal_melt_rate, accumulation)
        acc = acc.at[:].set(jnp.where(h_old>0, acc, 0))

        h_face = jnp.zeros((n+1,))
        h_face = h_face.at[1:n+1].set(h_old[:n]) #upwind values
        h_face = h_face.at[0].set(h_old[0].copy())

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

        h_new = h_old + dt * ( -( h_flux[1:(n+1)] - h_flux[:n] )/dx + acc )

        return h_new

    return adv


def make_timestepper_compiled(C, iterations, timesteps, dt, accumulation, bmr):

    picard_iterator = make_picard_iterator_for_u_compiled(C, iterations)
    adv_step = make_adv_expt(dt,accumulation)

    def continue_condition(state):
        _,_, ts, ssr = state
        return ts<timesteps #and something to do with ssr I hope

    def timestep(state):
        h_t, u_tmo, ts, steady_state_residual = state

        u_t,_,_,_,_ = picard_iterator(u_tmo, h_t)
        h_tpo = advection_step(u_t, h_t, bmr)

        steady_state_residual = None

        return h_tpo, u_t, ts+1, steady_state_residual

    @jax.jit
    def iterator(u_init, h_init):

        initial_state = h_init, u_init, 0, None

        h, u, ts, ssr = jax.lax.while_loop(continue_condition, timestep, initial_state)

        return h, u, ts, ssr

    return iterator


def make_adv_rhs(accumulation):
    
    def adv(u, h, basal_melt_rate):
        s_gnd = h + b
        s_flt = h*(1-rho/rho_w)

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


def plotgeom(thk):

    s_gnd = b + thk
    s_flt = thk*(1-rho/rho_w)
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
    s_flt = thk*(1-rho/rho_w)
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

    if isinstance(speeds, (jnp.ndarray, np.ndarray)):
        speeds = [np.array(u) for u in speeds]
    if isinstance(thks, (jnp.ndarray, np.ndarray)):
        thks = [np.array(h) for h in thks]

    fig, ax1 = plt.subplots(figsize=(10,6))
    ax2 = ax1.twinx()

    n = len(thks)

    cmap = cm.rainbow
    cs = cmap(jnp.linspace(0, 1, n))

    for thk, speed, c1 in list(zip(thks, speeds, cs)):
        s_gnd = b + thk
        s_flt = thk*(1-rho/rho_w)
        s = jnp.maximum(s_gnd, s_flt)

        base = s-thk
        ax1.plot(s, c=c1)
        # ax1.plot(base, label="base")
        ax1.plot(base, c=c1)

        ax2.plot(speed*3.15e7, color=c1, marker=".", linewidth=0)

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
    ax1.set_xlabel("x (m)")
    ax1.set_ylabel("elevation (m)")
    ax2.set_ylabel("speed (m/yr)")

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

    if isinstance(thks, (jnp.ndarray, np.ndarray)):
        thks = [np.array(h) for h in thks]

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



n = 1001
l = 1_800_000
x = jnp.linspace(0, l, n)
dx = x[1]-x[0]

rho = 900
rho_w = 1000

g = 9.8

accumulation = jnp.zeros_like(x)+0.3/(3.15e7)


C = 7.624e6
#C = 7.624e5

#A = 4.6146e-24
A = 5e-26

B = 2 * (A**(-1/3))

#epsilon_visc = 1e-5/(3.15e7)
epsilon_visc = 3e-11


#b = 720 - 778.5*x/750_000


x_s = x/l
#h_init = jnp.zeros_like(x)+100
h_init = 4000*jnp.exp(-2*((x_s)**15))
#h_init = 4000 - 3500*x_s*x_s
#h_init = 500 + 4000*jnp.exp(-2*((x_s+0.35)**15))

u_trial = jnp.zeros_like(x)
h_trial = h_init.copy()


#plotgeom(h_trial)
#raise




#####Having a look at eigenvalue spectra:
#NOTE: ALL OF BELOW (pretty much) USING:
#h_init = 4000*jnp.exp(-2*((x_s)**15))
#b = 720 - 778.5*x/750_000

u = jnp.load("./u_ss_2_1000cells.npy")
h = jnp.load("./h_ss_2_1000cells.npy")

L, fbk, ad = construct_tlo(u, h, C, accumulation, 0)

evals, evecs = jnp.linalg.eig(L)
order_indices = jnp.argsort(evals)
evals_ord = evals[order_indices]
evecs_ord = evecs[:, order_indices]

#plt.imshow(jnp.rot90(jnp.real(evecs_ord)), vmin=-0.1, vmax=0.1, cmap="RdBu_r")
#plt.show()
#raise

#plt.plot(evals_ord)
#plt.show()



raise

#####Testing implicit coupled picard solver
n_its = 60
#dt = 5e8

n_timesteps = 80

dt = 1e10
iteratorr = make_picard_iterator_for_joint_impl_problem_alt_compiled(C, n_its, dt, 0)
#dt = 1e8
#iteratorr = make_picard_iterator_for_joint_impl_problem(C, n_its, dt, 0)

u = u_trial.copy()
h = h_trial.copy()

uends = []
hends = []
for i in range(n_timesteps):
    u, h, res = iteratorr(u, h)
    #print(dx/jnp.max(u))
    uends.append(u)
    hends.append(h)

jnp.save("./u_ss_2_1000cells.npy", u)
jnp.save("./h_ss_2_1000cells.npy", h)
plotboths(hends, uends, n_timesteps)

raise


##########TESTING NEWTON'S METHOD FOR COUPLED IMPL PROBLEM
##For some reason this doesn't work, but i can't be arsed working out why.

#n_its = 60
##dt = 5e8
#
#dt = 1e8
#iteratorr = make_newton_iterator_for_joint_impl_problem(C, n_its, dt, 0, accumulation)
#
#u = u_trial.copy()
#h = h_trial.copy()
#
#uends = []
#hends = []
#for i in range(1):
#    u, us, h, hs = iteratorr(u, h)
#    #print(dx/jnp.max(u))
#    uends.append(u)
#    hends.append(h)
#plotboths(hends, uends, n_its)
#raise


######Testing implicit coupled picard solver
#n_its = 60
##dt = 5e8
#
#dt = 1e10
#iteratorr = make_picard_iterator_for_joint_impl_problem_alt(C, n_its, dt, 0)
#
##dt = 1e8
##iteratorr = make_picard_iterator_for_joint_impl_problem(C, n_its, dt, 0)
#
#
#u = u_trial.copy()
#h = h_trial.copy()
#
#uends = []
#hends = []
#for i in range(10):
#    u, us, h, hs = iteratorr(u, h)
#    #print(dx/jnp.max(u))
#    uends.append(u)
#    hends.append(h)
#plotboths(hends, uends, n_its)
#
#
#
#
#raise

######Testing explicit timestepping and picard u solver. Seems to work!

n_its = 60
#dt = 2.6e6 #roughly a month
#dt = dt/10
dt = 1e8
n_timesteps = 20

u = u_trial.copy()
h = h_trial.copy()

picard_iterator = make_picard_iterator_for_u(C, n_its)
advection_step = make_adv_expt(dt, accumulation)


##FOR SOME REASON, THIS IS A LOT SLOWER THAN THE LOOP
#timestepper = make_timestepper_compiled(C, n_its, n_timesteps, dt, accumulation, 0)
#h_final, u_final, _,_ = timestepper(u, h)
#plotboth(h_final, u_final*3.15e7)
#raise

us = []
hs = []

picard_iterator_comp = make_picard_iterator_for_u_compiled(C, n_its)
for i in range(n_timesteps):
    if not i%500:
        print(i)
    #u, _ = picard_iterator(u, h)
    u, _,_,_,_ = picard_iterator_comp(u, h)

    h = advection_step(u, h, jnp.zeros_like(accumulation))


    us.append(u)
    hs.append(h)

plotboths(hs[::1], us[::1], n_timesteps)
raise





plotboths([h_trial for u in us], us, n_its)
raise

cmap = cm.rainbow
cs = cmap(jnp.linspace(0, 1, n_its))
for u, c in zip(us, cs):
    plt.plot(u, c=c)
plt.show()

raise

solver = implicit_coupled_solver(100_000, 10, 1)
u, h, us, hs = solver(u_trial, h_trial)






