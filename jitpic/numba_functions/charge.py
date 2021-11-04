"""
This module contains the charge deposition functions.
"""
import numba

from .shapes import quadratic_shape_factor, cubic_shape_factor, quartic_shape_factor
# import the numba configuration
from ..config import parallel, cache, fastmath

@numba.njit("(i8, i8, f8[::1], f8, f8[::1], f8[:,::1], u4[::1], u4[::1], i4[::1], f8[::1], i4)", 
            parallel=parallel, cache=cache, fastmath=fastmath)
def R1o(N, n_threads, x, idx, qw, rho, l, r, indices, xg, Nx ):
    """
    Charge density deposition for linear particle shapes and open boundaries
    
    N         : number of particles
    n_threads : number of parallel threads
    x         : particle positions
    idx       : inverse grid dx
    qw        : particle (w*q)
    rho       : 2D rho array
    l         : particle left cell indices
    r         : particle right cell indices
    indices   : particle index start/stop for each thread
    xg        : grid points
    Nx        : grid Nx

    No returns neccessary as arrays are modified in-place.
    """
    for j in numba.prange(n_threads):
        for i in range( indices[j], indices[j+1] ):   
            
            xi = (x[i] - xg[l[i]])*idx
            
            rho[j,l[i]] += qw[i] * (1-xi)
            rho[j,r[i]] += qw[i] * xi  
            
    return

@numba.njit("(i8, i8, f8[::1], f8, f8[::1], f8[:,::1], u4[::1], u4[::1], i4[::1], f8[::1], i4)", 
            parallel=parallel, cache=cache, fastmath=fastmath)
def R2o(N, n_threads, x, idx, qw, rho, l, r, indices, xg, Nx ):
    """
    Charge density deposition for quadratic particle shapes and open boundaries

    N         : number of particles
    n_threads : number of parallel threads
    x         : particle positions
    idx       : inverse grid dx
    qw        : particle (w*q)
    rho       : 2D rho array
    l         : particle left cell indices
    r         : particle right cell indices
    indices   : particle index start/stop for each thread
    xg        : grid points
    Nx        : grid Nx

    No returns neccessary as arrays are modified in-place.
    """ 
    for j in numba.prange(n_threads):
        for i in range( indices[j], indices[j+1] ):   
            
            xi = (x[i] - xg[l[i]])*idx
            
            if l[i] > 0:
                rho[j,l[i]-1] += qw[i] * quadratic_shape_factor(1+xi)
                
            rho[j,l[i]]   += qw[i] * quadratic_shape_factor(xi  )
            rho[j,r[i]]   += qw[i] * quadratic_shape_factor(1-xi)  
            
            if r[i] < Nx-2:
                rho[j,r[i]+1]    += qw[i] * quadratic_shape_factor(2-xi)

    return

@numba.njit("(i8, i8, f8[::1], f8, f8[::1], f8[:,::1], u4[::1], u4[::1], i4[::1], f8[::1], i4)", 
            parallel=parallel, cache=cache, fastmath=fastmath)
def R3o(N, n_threads, x, idx, qw, rho, l, r, indices, xg, Nx ):
    """
    Charge density deposition for cubic particle shapes and open boundaries

    N         : number of particles
    n_threads : number of parallel threads
    x         : particle positions
    idx       : inverse grid dx
    qw        : particle (w*q)
    rho       : 2D rho array
    l         : particle left cell indices
    r         : particle right cell indices
    indices   : particle index start/stop for each thread
    xg        : grid points
    Nx        : grid Nx

    No returns neccessary as arrays are modified in-place.
    """ 
    for j in numba.prange(n_threads):
        for i in range( indices[j], indices[j+1] ):   
            
            xi = (x[i] - xg[l[i]])*idx
            
            if l[i] > 0:
                rho[j,l[i]-1] += qw[i] * cubic_shape_factor(1+xi)
                
            rho[j,l[i]]   += qw[i] * cubic_shape_factor(xi  )
            rho[j,r[i]]   += qw[i] * cubic_shape_factor(1-xi)  
            
            if r[i] < Nx-2:
                rho[j,r[i]+1]    += qw[i] * cubic_shape_factor(2-xi)

    return

@numba.njit("(i8, i8, f8[::1], f8, f8[::1], f8[:,::1], u4[::1], u4[::1], i4[::1], f8[::1], i4)", 
            parallel=parallel, cache=cache, fastmath=fastmath)
def R4o(N, n_threads, x, idx, qw, rho, l, r, indices, xg, Nx ):
    """
    Charge density deposition for quartic particle shapes and open boundaries

    N         : number of particles
    n_threads : number of parallel threads
    x         : particle positions
    idx       : inverse grid dx
    qw        : particle (w*q)
    rho       : 2D rho array
    l         : particle left cell indices
    r         : particle right cell indices
    indices   : particle index start/stop for each thread
    xg        : grid points
    Nx        : grid Nx

    No returns neccessary as arrays are modified in-place.
    """ 
    
    for j in numba.prange(n_threads):
        for i in range( indices[j], indices[j+1] ):   

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


@numba.njit("(i8, i8, f8[::1], f8, f8[::1], f8[:,::1], u4[::1], u4[::1], i4[::1], f8[::1], i4)", 
            parallel=parallel, cache=cache, fastmath=fastmath)
def R2p(N, n_threads, x, idx, qw, rho, l, r, indices, xg, Nx ):
    """
    Charge density deposition for quadratic particle shapes and periodic boundaries

    N         : number of particles
    n_threads : number of parallel threads
    x         : particle positions
    idx       : inverse grid dx
    qw        : particle (w*q)
    rho       : 2D rho array
    l         : particle left cell indices
    r         : particle right cell indices
    indices   : particle index start/stop for each thread
    xg        : grid points
    Nx        : grid Nx

    No returns neccessary as arrays are modified in-place.
    """ 
    for j in numba.prange(n_threads):
        for i in range( indices[j], indices[j+1] ):   
            
            xi = (x[i] - xg[l[i]])*idx
            
            rp1 = (r[i] + 1)%Nx

            rho[j,l[i]-1] += qw[i] * quadratic_shape_factor(1+xi)
            rho[j,l[i]]   += qw[i] * quadratic_shape_factor(xi  )
            rho[j,r[i]]   += qw[i] * quadratic_shape_factor(1-xi)  
            rho[j,rp1 ]   += qw[i] * quadratic_shape_factor(2-xi)

    return

@numba.njit("(i8, i8, f8[::1], f8, f8[::1], f8[:,::1], u4[::1], u4[::1], i4[::1], f8[::1], i4)", 
            parallel=parallel, cache=cache, fastmath=fastmath)
def R3p(N, n_threads, x, idx, qw, rho, l, r, indices, xg, Nx ):
    """
    Charge density deposition for cubic particle shapes and periodic boundaries

    N         : number of particles
    n_threads : number of parallel threads
    x         : particle positions
    idx       : inverse grid dx
    qw        : particle (w*q)
    rho       : 2D rho array
    l         : particle left cell indices
    r         : particle right cell indices
    indices   : particle index start/stop for each thread
    xg        : grid points
    Nx        : grid Nx

    No returns neccessary as arrays are modified in-place.
    """ 
    for j in numba.prange(n_threads):
        for i in range( indices[j], indices[j+1] ):   
            
            xi = (x[i] - xg[l[i]])*idx
            
            rp1 = (r[i] + 1)%Nx
            
            rho[j,l[i]-1] += qw[i] * cubic_shape_factor(1+xi)
            rho[j,l[i]]   += qw[i] * cubic_shape_factor(xi  )
            rho[j,r[i]]   += qw[i] * cubic_shape_factor(1-xi)  
            rho[j,rp1 ]   += qw[i] * cubic_shape_factor(2-xi)

    return

@numba.njit("(i8, i8, f8[::1], f8, f8[::1], f8[:,::1], u4[::1], u4[::1], i4[::1], f8[::1], i4)", 
            parallel=parallel, cache=cache, fastmath=fastmath)
def R4p(N, n_threads, x, idx, qw, rho, l, r, indices, xg, Nx ):
    """
    Charge density deposition for quartic particle shapes and periodic boundaries

    N         : number of particles
    n_threads : number of parallel threads
    x         : particle positions
    idx       : inverse grid dx
    qw        : particle (w*q)
    rho       : 2D rho array
    l         : particle left cell indices
    r         : particle right cell indices
    indices   : particle index start/stop for each thread
    xg        : grid points
    Nx        : grid Nx

    No returns neccessary as arrays are modified in-place.
    """ 
    for j in numba.prange(n_threads):
        for i in range( indices[j], indices[j+1] ):   

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