"""
This module contains particle pushers
"""
# import the configuration first
from ..config import parallel, cache, fastmath
import numpy as np
import numba

@numba.njit("(f8[:,::1], f8[:,::1], f8, f8[:,::1], f8[:,::1], f8[::1], f8[::1], f8[::1], f8, f8, i8, b1[::1])", 
            parallel=parallel, cache=cache, fastmath=fastmath)
def cohen_push( E, B, qmdt, p, v, x, x_old, rg, m, dt, N, state):
    """
    Cohen particle push
    http://dx.doi.org/10.1016/j.nima.2009.03.083
    
    E        : particle E-fields
    B        : particle B-fields
    qmdt     : constant factor
    p        : particle momenta
    v        : particle velocities
    x        : particle positions
    x_old    : previous particle position
    rg       : particle reciprocal gammas
    m        : particle mass
    dt       : timestep
    N        : number of particles
    state    : particle states
    
    No returns neccessary as arrays are modified in-place.
    """

    for i in numba.prange(N):
        if state[i]:
            a = np.empty(3) # must be assigned within the loop to be private
            
            b = 0.5*qmdt*B[:,i]
            
            #a = p0 + q*E + q/2*p/gamma x B
            a[0] = p[0,i] + qmdt*E[0,i] +  rg[i]*(p[1,i]*b[2] - p[2,i]*b[1])
            a[1] = p[1,i] + qmdt*E[1,i] +  rg[i]*(p[2,i]*b[0] - p[0,i]*b[2])
            a[2] = p[2,i] + qmdt*E[2,i] +  rg[i]*(p[0,i]*b[1] - p[1,i]*b[0])
            
            a2 = a[0]**2 + a[1]**2 + a[2]**2
            b2 = b[0]**2 + b[1]**2 + b[2]**2
            
            a2b2 = 0.5*(1. + a2 - b2)
            adotb = a[0]*b[0] + a[1]*b[1] + a[2]*b[2] 
            
            gamma2 = a2b2 + np.sqrt( a2b2**2 + b2 + adotb**2 ) 
            gamma = np.sqrt(gamma2)
            
            p[0,i] = ( gamma2*a[0] + gamma*(a[1]*b[2] - a[2]*b[1]) + b[0]*adotb ) / (gamma2 + b2)
            p[1,i] = ( gamma2*a[1] + gamma*(a[2]*b[0] - a[0]*b[2]) + b[1]*adotb ) / (gamma2 + b2)
            p[2,i] = ( gamma2*a[2] + gamma*(a[0]*b[1] - a[1]*b[0]) + b[2]*adotb ) / (gamma2 + b2)
            
            rg[i] = 1./gamma
            
            # make a note of the old positions
            x_old[i] = x[i]
            
            # update v
            v[:,i] = p[:,i] * rg[i] / m
            
            # update x
            x[i] = x[i] + v[0,i] * dt
         
    return

@numba.njit("(f8[:,::1], f8[:,::1], f8, f8[:,::1], f8[:,::1], f8[::1], f8[::1], f8[::1], f8, f8, i8, b1[::1])", 
            parallel=parallel, cache=cache, fastmath=fastmath)
def vay_push( E, B, qmdt, p, v, x, x_old, rg, m, dt, N, state):
    """
    Vay particle push
    https://doi.org/10.1063/1.2837054
    
    E        : particle E-fields
    B        : particle B-fields
    qmdt     : constant factor: 2*pi*dt*q/m
    p        : particle momenta
    v        : particle velocities
    x        : particle positions
    x_old    : previous particle position
    rg       : particle reciprocal gammas
    m        : particle mass
    dt       : timestep
    N        : number of particles
    state    : particle states
    
    No returns neccessary as arrays are modified in-place.
    """
    # qmdt = 2*pi*dt*q/m
    for i in numba.prange(N):
        if state[i]:

            u = np.empty(3) # must be assigned within the loop to be private
            
            # tau = (pi*dt*q/m) * B
            tau = 0.5*qmdt*B[:,i]
            tau2 = tau[0]**2 + tau[1]**2 + tau[2]**2
            
            # p* = p0 + q*E + q/2*v x B
            u[0] = p[0,i] + qmdt*E[0,i] +  (v[1,i]*tau[2] - v[2,i]*tau[1])
            u[1] = p[1,i] + qmdt*E[1,i] +  (v[2,i]*tau[0] - v[0,i]*tau[2])
            u[2] = p[2,i] + qmdt*E[2,i] +  (v[0,i]*tau[1] - v[1,i]*tau[0])
    
            # t = tau/gamma1
            # gamma1 = sqrt( sigma + sqrt(sigma^2 + (tau^2 + w^2) ) )
            # w = dot(p*, tau)
            # sigma = (gamma'^2 - tau^2)/2
            # gamma' = sqrt(1 + p*^2)
            gamma_prime = np.sqrt( 1 + u[0]**2 + u[1]**2 + u[2]**2 )
            sigma = (gamma_prime**2 - tau2)*0.5
            w = u[0]*tau[0] + u[1]*tau[1] + u[2]*tau[2] 
            gamma1 = np.sqrt( sigma + np.sqrt(sigma**2 + (tau2 + w**2) ) )
            t = tau/gamma1
            t2 = t[0]**2 + t[1]**2 + t[2]**2
            
            # p1 = (p* + dot(p*,t)*t + cross(p*,t) ) / (1 + t^2)
            pdott = u[0]*t[0] + u[1]*t[1] + u[2]*t[2]
            p[0,i] = ( u[0] + pdott*t[0] + (u[1]*t[2] - u[2]*t[1]) ) / (1 + t2)
            p[1,i] = ( u[1] + pdott*t[1] + (u[2]*t[0] - u[0]*t[2]) ) / (1 + t2)
            p[2,i] = ( u[2] + pdott*t[2] + (u[0]*t[1] - u[1]*t[0]) ) / (1 + t2)
            
            rg[i] = 1./gamma1
            
            # make a note of the old positions
            x_old[i] = x[i]
            
            # update v
            v[:,i] = p[:,i] * rg[i] / m
    
            # update x
            x[i] = x[i] + v[0,i] * dt
         
    return

@numba.njit("(f8[:,::1], f8[:,::1], f8, f8[:,::1], f8[:,::1], f8[::1], f8[::1], f8[::1], f8, f8, i8, b1[::1])", 
            parallel=parallel, cache=cache, fastmath=fastmath)
def boris_push( E, B, qmdt2, p, v, x, x_old, rg, m, dt, N, state):
    """
    Boris particle push
    
    Boris, J. P. 1970. "Relativistic Plasma Simulation-Optimization of a Hybrid Code." 
    In Proc. Fourth Conf. Num. Sim. Plasmas, 3–67. Naval Res. Lab., Wash., D. C.
    
    E        : particle E-fields
    B        : particle B-fields
    qmdt2    : constant factor
    p        : particle momenta
    v        : particle velocities
    x        : particle positions
    x_old    : previous particle position
    rg       : particle reciprocal gammas
    m        : particle mass
    dt       : timestep
    N        : number of particles
    state    : particle states
    
    No returns neccessary as arrays are modified in-place.
    """
    
    for i in numba.prange(N):
        if state[i]:
            P = np.empty(3) # must be assigned within the loop to be private
            
            # half-push
            p[:,i] = p[:,i] + qmdt2 * E[:,i] # Pn-1/2 > P-
            rg[i] = 1./np.sqrt(1. + p[0,i]*p[0,i] + p[1,i]*p[1,i] + p[2,i]*p[2,i])
           
            # apply rotation
            T = qmdt2 * B[:,i] * rg[i] # these are automatically privatised
            S = 2. * T / (1. + T[0]*T[0] + T[1]*T[1] + T[2]*T[2]  )
            
            # p- = p + cross(p, T)            
            P[0] = p[0,i] + (p[1,i]*T[2] - p[2,i]*T[1])
            P[1] = p[1,i] + (p[2,i]*T[0] - p[0,i]*T[2])
            P[2] = p[2,i] + (p[0,i]*T[1] - p[1,i]*T[0])
            
            # p+ = p- + cross(P, S) 
            p[0,i] = p[0,i] + (P[1]*S[2] - P[2]*S[1])
            p[1,i] = p[1,i] + (P[2]*S[0] - P[0]*S[2])
            p[2,i] = p[2,i] + (P[0]*S[1] - P[1]*S[0])
            
            # complete the push
            p[:,i] = p[:,i] + qmdt2 * E[:,i]
            rg[i] = 1./np.sqrt(1. + p[0,i]*p[0,i] + p[1,i]*p[1,i] + p[2,i]*p[2,i])
            
            # make a note of the old positions
            x_old[i] = x[i]
            
            # update v
            v[:,i] = p[:,i] * rg[i] / m
            
            # update x
            x[i] = x[i] + v[0,i] * dt
         
    return