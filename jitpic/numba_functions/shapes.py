"""
This module contains the piecewise functions describing various particle shapes.
The functions are optimised for performance. 
The basic shape factors are symmetric, so only the positive part is computed, arguments should reflect this
"""
# import the numba configuration first
from ..config import cache, fastmath
import numba
import numpy as np

@numba.njit("f8(f8)", cache=cache, fastmath=fastmath)
def quadratic_shape_factor(x):
    if x < 0.5:
        return 0.75-x**2
    elif x < 1.5:
        return (x**2-3*x+2.25) * 0.5
    else:
        return 0

@numba.njit("f8(f8)", cache=cache, fastmath=fastmath) 
def cubic_shape_factor(x):
    if x < 1:
        return (3*x**3-6*x**2+4) * 0.16666666666666666
    elif x < 2:
        return (-x**3+6*x**2-12*x+8) * 0.16666666666666666
    else: 
        return 0

@numba.njit("f8(f8)", cache=cache, fastmath=fastmath) 
def quartic_shape_factor(x):
    if x < 0.5:
        return (48*x**4-120*x**2+115) * 0.005208333333333333
    elif x < 1.5:
        return (-16*x**4+80*x**3-120*x**2+20*x+55)* 0.010416666666666666
    elif x < 2.5:
        return (2*x-5)**4 * 0.0026041666666666665
    else: 
        return 0
    
# integrated shape functions for current deposition
@numba.njit("f8(f8)", cache=cache, fastmath=fastmath)
def integrated_linear_shape_factor(x):
    sgn = np.sign(x)
    x = abs(x)    
    if x < 1:
        return sgn * (x-0.5*x**2)
    else:# x>1:
        return sgn * 0.5 
     
@numba.njit("f8(f8)", cache=cache, fastmath=fastmath)
def integrated_quadratic_shape_factor(x):  
    sgn = np.sign(x)
    x = abs(x)
    if x < 0.5:
        return sgn * (0.75*x-x**3 * 0.3333333333333333)
    elif x < 1.5:
        return sgn * (8*x**3-36*x**2+54*x-3) * 0.020833333333333332
    else: # x>1.5
        return sgn * 0.5

@numba.njit("f8(f8)", cache=cache, fastmath=fastmath)
def integrated_cubic_shape_factor(x):    
    sgn = np.sign(x)
    x = abs(x)
    if x < 1:
        return sgn * x*(3*x**3-8*x**2+16) * 0.041666666666666664
    elif x < 2:
        return sgn * (-x**4+8*x**3-24*x**2+32*x-4) * 0.041666666666666664
    else: # x > 2
        return sgn * 0.5

@numba.njit("f8(f8)", cache=cache, fastmath=fastmath)
def integrated_quartic_shape_factor(x):    
    sgn = np.sign(x)
    x = abs(x)
    if x < 0.5:
        return sgn * (48*x**5-200*x**3+575*x) * 0.0010416666666666667
    elif x < 1.5:
        return sgn * (-16*x**5+100*x**4-200*x**3+50*x**2+275*x+1.25) * 0.0020833333333333333
    elif x < 2.5:
        return sgn * ((2*x-5)**5 * 0.00026041666666666666 + 0.5)
    else: # x > 2.5
        return sgn * 0.5