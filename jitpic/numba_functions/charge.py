"""
This module contains the charge deposition functions.
Charge is deposited on grid verices, not in cell centers! 

arguments  
    n_threads : number of parallel threads
    x         : particle positions
    idx       : inverse grid dx
    qw        : particle (w*q)
    rho       : 2D rho array
    l         : particle left cell indices
    r         : particle right cell indices
    state     : particle states
    indices   : particle index start/stop for each thread
    xg        : grid points
    Nx        : grid Nx

    No returns neccessary as arrays are modified in-place.
"""

# import the numba configuration
from ..config import parallel, cache, fastmath
import numba

from .shapes import quadratic_shape_factor, cubic_shape_factor, quartic_shape_factor

signature = "(i8, f8[::1], f8, f8[::1], f8[:,::1], u4[::1], u4[::1], b1[::1], i4[::1], f8[::1], i4)"
njit = numba.njit(signature, parallel=parallel, cache=cache, fastmath=fastmath)

@njit
def R1o(n_threads, x, idx, qw, rho, l, r, state, indices, xg, Nx ):
    """ 1st order shapes and open or periodic boundaries """
    for j in numba.prange(n_threads):
        for i in range( indices[j], indices[j+1] ):   
            if state[i]:
                xi = (x[i] - xg[l[i]])*idx
                
                rho[j,l[i]] += qw[i] * (1-xi)
                rho[j,r[i]] += qw[i] * xi      
    return

@njit
def R2o(n_threads, x, idx, qw, rho, l, r, state, indices, xg, Nx ):
    """ 2nd order shapes and open boundaries """
    for j in numba.prange(n_threads):
        for i in range( indices[j], indices[j+1] ):   
            if state[i]:
                
                xi = (x[i] - xg[l[i]])*idx
                
                if l[i] > 0:
                    rho[j,l[i]-1] += qw[i] * quadratic_shape_factor(1+xi)
                    
                rho[j,l[i]]   += qw[i] * quadratic_shape_factor(xi  )
                rho[j,r[i]]   += qw[i] * quadratic_shape_factor(1-xi)  
                
                if r[i] < Nx-2:
                    rho[j,r[i]+1]    += qw[i] * quadratic_shape_factor(2-xi)
    return

@njit
def R3o(n_threads, x, idx, qw, rho, l, r, state, indices, xg, Nx ):
    """ 3rd order shapes and open boundaries """
    for j in numba.prange(n_threads):
        for i in range( indices[j], indices[j+1] ):   
            if state[i]:
                
                xi = (x[i] - xg[l[i]])*idx
                
                if l[i] > 0:
                    rho[j,l[i]-1] += qw[i] * cubic_shape_factor(1+xi)
                    
                rho[j,l[i]]   += qw[i] * cubic_shape_factor(xi  )
                rho[j,r[i]]   += qw[i] * cubic_shape_factor(1-xi)  
                
                if r[i] < Nx-2:
                    rho[j,r[i]+1]    += qw[i] * cubic_shape_factor(2-xi)
    return

@njit
def R4o(n_threads, x, idx, qw, rho, l, r, state, indices, xg, Nx ):
    """ 4th order shapes and open boundaries """ 
    for j in numba.prange(n_threads):
        for i in range( indices[j], indices[j+1] ):   
            if state[i]:
                
                xi = (x[i] - xg[l[i]])*idx
                
                if l[i] > 1:
                    rho[j,l[i]-2] += qw[i] * quartic_shape_factor(2+xi)
                if l[i] > 0:
                    rho[j,l[i]-1] += qw[i] * quartic_shape_factor(1+xi)
                    
                rho[j,l[i]]   += qw[i] * quartic_shape_factor(xi  )
                rho[j,r[i]]   += qw[i] * quartic_shape_factor(1-xi)  
                
                if r[i] < Nx-2:
                    rho[j,r[i]+1]    += qw[i] * quartic_shape_factor(2-xi)  
                if r[i] < Nx-3:
                    rho[j,r[i]+2]    += qw[i] * quartic_shape_factor(3-xi)  
    return

# Deposition for linear particles are the same, so a simple alias is used.
R1p = R1o


@njit
def R2p(n_threads, x, idx, qw, rho, l, r, state, indices, xg, Nx ):
    """ 2nd order shapes and periodic boundaries """
    for j in numba.prange(n_threads):
        for i in range( indices[j], indices[j+1] ):   
            if state[i]:
                xi = (x[i] - xg[l[i]])*idx
                
                rp1 = (r[i] + 1)%Nx
    
                rho[j,l[i]-1] += qw[i] * quadratic_shape_factor(1+xi)
                rho[j,l[i]]   += qw[i] * quadratic_shape_factor(xi  )
                rho[j,r[i]]   += qw[i] * quadratic_shape_factor(1-xi)  
                rho[j,rp1 ]   += qw[i] * quadratic_shape_factor(2-xi)
    return

@njit
def R3p(n_threads, x, idx, qw, rho, l, r, state, indices, xg, Nx ):
    """ 3rd order shapes and periodic boundaries """
    for j in numba.prange(n_threads):
        for i in range( indices[j], indices[j+1] ):   
            if state[i]:
                xi = (x[i] - xg[l[i]])*idx
                
                rp1 = (r[i] + 1)%Nx
                
                rho[j,l[i]-1] += qw[i] * cubic_shape_factor(1+xi)
                rho[j,l[i]]   += qw[i] * cubic_shape_factor(xi  )
                rho[j,r[i]]   += qw[i] * cubic_shape_factor(1-xi)  
                rho[j,rp1 ]   += qw[i] * cubic_shape_factor(2-xi)
    return

@njit
def R4p(n_threads, x, idx, qw, rho, l, r, state, indices, xg, Nx ):
    """ 4th order shapes and periodic boundaries """
    for j in numba.prange(n_threads):
        for i in range( indices[j], indices[j+1] ):   
            if state[i]:
                xi = (x[i] - xg[l[i]])*idx
                
                rp1 = (r[i] + 1)%Nx
                rp2 = (r[i] + 2)%Nx
                
                rho[j,l[i]-2] += qw[i] * quartic_shape_factor(2+xi)
                rho[j,l[i]-1] += qw[i] * quartic_shape_factor(1+xi)
                rho[j,l[i]]   += qw[i] * quartic_shape_factor(xi  )
                rho[j,r[i]]   += qw[i] * quartic_shape_factor(1-xi)  
                rho[j,rp1 ]   += qw[i] * quartic_shape_factor(2-xi)  
                rho[j,rp2 ]   += qw[i] * quartic_shape_factor(3-xi)           
    return