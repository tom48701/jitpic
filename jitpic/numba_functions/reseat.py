"""
This module contains particle reseating functions

Update the left and right indices for particles
after a push, then look for particles that have 
been pushed off the grid and disable them.

Arguments  
    N     : number of particles
    x     : particle positions
    state : particle status (alive/dead)
    l     : left cell index
    r     : right cell index
    x0    : grid x0
    x1    : grid x1
    idx   : grid inverse dx
    Nx    : grid Nx
     
    No return neccessary as arrays are modified in-place.
"""
# import the numba configuration
from ..config import parallel, cache, fastmath
import numpy as np
import numba

signature = "(i8, f8[::1], b1[::1], u4[::1], u4[::1], f8, f8, f8, f8, i8)"
njit = numba.njit(signature, parallel=parallel, cache=cache, fastmath=fastmath)

@njit
def reseat_open(N,x,state,l,r,x0,x1,dx,idx,Nx ):
    """ Open boundaries """
    l[:] = np.floor((x-x0)*idx) # left cell
    r[:] = l+1 # right cell   
        
    for i in numba.prange(N):
        if x[i] >= x1 or x[i] < x0:
            state[i] = False
    return

@njit
def reseat_periodic(N,x,state,l,r,x0,x1,dx,idx,Nx ):
    """ Periodic Boundaries  """
    l[:] = np.floor((x-x0)*idx) # left cell
    r[:] = l+1 # right cell   
      
    for i in numba.prange(N): # particle BEFORE first cell
        if x[i] < x0:
            x[i] += Nx*dx
            r[i] = 0
            l[i] = Nx-1
            
        if x[i] > x1+dx: # aprticle BEYOND final 'virtual' cell
            x[i] -= Nx*dx
            l[i] = 0
            r[i] = 1
            
        if x[i] > x1: # particle IN final virtual cell
            r[i] = 0
    return