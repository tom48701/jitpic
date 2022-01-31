"""
This module contains interpolation functions

Arguments:    
    Eg    : grid E field
    Eb    : grid B field
    Es    : particle E field
    Bs    : particle B field
    l     : left cell indices
    r     : right cell indices
    x     : normalised particle positions
    N     : number of particles
    state : particle states
        
    No return neccessary as arrays are modified in-place.

"""
# import the numba configuration first
from ..config import parallel, cache, fastmath
import numba
from .shapes import quadratic_shape_factor, cubic_shape_factor, quartic_shape_factor

signature = "(f8[:,::1], f8[:,::1], f8[:,::1], f8[:,::1], u4[::1], u4[::1], f8[::1], i4, b1[::1])"
njit = numba.njit(signature, parallel=parallel, cache=cache, fastmath=fastmath)

@njit
def I1o(Eg,Bg, Es,Bs, l,r, x, N, state):
    """ 1st order shapes with open or periodic boundaries """
    for i in numba.prange(N):
        if state[i]:
            xi = x[i] - l[i]
    
            Es[:,i] = (1-xi)*Eg[:,l[i]] + xi*Eg[:,r[i]]
            Bs[:,i] = (1-xi)*Bg[:,l[i]] + xi*Bg[:,r[i]]
    return

@njit
def I2o(Eg,Bg, Es,Bs, l,r, x, N, state):
    """ 2nd order shapes with open boundaries """
    for i in numba.prange(N):
        if state[i]:
            xi = x[i] - l[i]
            
            lm1 = quadratic_shape_factor(   xi+1 )
            L   = quadratic_shape_factor(   xi   )
            R   = quadratic_shape_factor( 1-xi   )
            rp1 = quadratic_shape_factor( 2-xi   )
            
            Es[:,i] = Eg[:,l[i]-1]*lm1 + \
                      Eg[:,l[i]  ]*L   + \
                      Eg[:,r[i]  ]*R   + \
                      Eg[:,r[i]+1]*rp1
                      
            Bs[:,i] = Bg[:,l[i]-1]*lm1 + \
                      Bg[:,l[i]  ]*L   + \
                      Bg[:,r[i]  ]*R   + \
                      Bg[:,r[i]+1]*rp1 
    return

@njit
def I3o(Eg,Bg, Es,Bs, l,r, x, N, state):
    """ 3rd order shapes with open boundaries """
    for i in numba.prange(N):
        if state[i]:
            xi = x[i] - l[i]
            
            lm1 = cubic_shape_factor(   xi+1 )
            L   = cubic_shape_factor(   xi   )
            R   = cubic_shape_factor( 1-xi   )
            rp1 = cubic_shape_factor( 2-xi   )
                    
            Es[:,i] = Eg[:,l[i]-1]*lm1 + \
                      Eg[:,l[i]  ]*L   + \
                      Eg[:,r[i]  ]*R   + \
                      Eg[:,r[i]+1]*rp1
                      
            Bs[:,i] = Bg[:,l[i]-1]*lm1 + \
                      Bg[:,l[i]  ]*L   + \
                      Bg[:,r[i]  ]*R   + \
                      Bg[:,r[i]+1]*rp1           
    return

@njit
def I4o(Eg,Bg, Es,Bs, l,r, x, N, state):
    """ 4th order shapes with open boundaries """
    for i in numba.prange(N):
        if state[i]:
            xi = x[i] - l[i]
    
            lm2 = quartic_shape_factor(   xi+2 )
            lm1 = quartic_shape_factor(   xi+1 )
            L   = quartic_shape_factor(   xi   )
            R   = quartic_shape_factor( 1-xi   )
            rp1 = quartic_shape_factor( 2-xi   )
            rp2 = quartic_shape_factor( 3-xi   )
            
            Es[:,i] = Eg[:,l[i]-2]*lm2 + \
                      Eg[:,l[i]-1]*lm1 + \
                      Eg[:,l[i]  ]*L   + \
                      Eg[:,r[i]  ]*R   + \
                      Eg[:,r[i]+1]*rp1 + \
                      Eg[:,r[i]+2]*rp2 
          
            Bs[:,i] = Bg[:,l[i]-2]*lm2 + \
                      Bg[:,l[i]-1]*lm1 + \
                      Bg[:,l[i]  ]*L   + \
                      Bg[:,r[i]  ]*R   + \
                      Bg[:,r[i]+1]*rp1 + \
                      Bg[:,r[i]+2]*rp2 
    return

# Interpolation for linear particles are the same, so a simple alias is used.
I1p = I1o

@njit
def I2p(Eg,Bg, Es,Bs, l,r, x, N, state):
    """ 2nd order shapes with periodic boundaries """
    Nx = len(Eg[0])

    for i in numba.prange(N):

        xi = x[i] - l[i] 
        rp1 = r[i]+1
        
        if r[i] == Nx-1:
            rp1 = 0
        
        lm1 = quadratic_shape_factor(   xi+1 )
        L   = quadratic_shape_factor(   xi   )
        R   = quadratic_shape_factor( 1-xi   )
        Rp1 = quadratic_shape_factor( 2-xi   )
        
        Es[:,i] = Eg[:,l[i]-1]*lm1 + \
                  Eg[:,l[i]  ]*L   + \
                  Eg[:,r[i]  ]*R   + \
                  Eg[:,rp1   ]*Rp1
                  
        Bs[:,i] = Bg[:,l[i]-1]*lm1 + \
                  Bg[:,l[i]  ]*L   + \
                  Bg[:,r[i]  ]*R   + \
                  Bg[:,rp1   ]*Rp1 
    return

@njit
def I3p(Eg,Bg, Es,Bs, l,r, x, N, state):
    """ 3rd order shapes with open boundaries """
    Nx = len(Eg[0])
    
    for i in numba.prange(N):

        xi = x[i] - l[i]
        rp1 = r[i]+1
        
        if r[i] == Nx-1:
            rp1 = 0
                
        lm1 = cubic_shape_factor(   xi+1 )
        L   = cubic_shape_factor(   xi   )
        R   = cubic_shape_factor( 1-xi   )
        Rp1 = cubic_shape_factor( 2-xi   )
                
        Es[:,i] = Eg[:,l[i]-1]*lm1 + \
                  Eg[:,l[i]  ]*L   + \
                  Eg[:,r[i]  ]*R   + \
                  Eg[:,rp1   ]*Rp1
                  
        Bs[:,i] = Bg[:,l[i]-1]*lm1 + \
                  Bg[:,l[i]  ]*L   + \
                  Bg[:,r[i]  ]*R   + \
                  Bg[:,rp1   ]*Rp1 
    return

@njit
def I4p(Eg,Bg, Es,Bs, l,r, x, N, state):
    """ 4th order shapes with periodic boundaries"""
    Nx = len(Eg[0])
    
    for i in numba.prange(N):

        xi = x[i] - l[i]
        rp1 = r[i]+1
        rp2 = r[i]+2
        
        if r[i] == Nx-1:
            rp1 = 0
            rp2 = 1
        elif r[i] == Nx-2:
            rp2 = 0
                
        lm2 = quartic_shape_factor(   xi+2 )
        lm1 = quartic_shape_factor(   xi+1 )
        L   = quartic_shape_factor(   xi   )
        R   = quartic_shape_factor( 1-xi   )
        Rp1 = quartic_shape_factor( 2-xi   )
        Rp2 = quartic_shape_factor( 3-xi   )
        
        Es[:,i] = Eg[:,l[i]-2]*lm2 + \
                  Eg[:,l[i]-1]*lm1 + \
                  Eg[:,l[i]  ]*L   + \
                  Eg[:,r[i]  ]*R   + \
                  Eg[:,rp1   ]*Rp1 + \
                  Eg[:,rp2   ]*Rp2 
      
        Bs[:,i] = Bg[:,l[i]-2]*lm2 + \
                  Bg[:,l[i]-1]*lm1 + \
                  Bg[:,l[i]  ]*L   + \
                  Bg[:,r[i]  ]*R   + \
                  Bg[:,rp1   ]*Rp1 + \
                  Bg[:,rp2   ]*Rp2 
    return