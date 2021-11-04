"""
This module contains the current deposition functions.
"""
import numba

from .shapes import quadratic_shape_factor, cubic_shape_factor, quartic_shape_factor
from .shapes import integrated_linear_shape_factor, integrated_quadratic_shape_factor, integrated_cubic_shape_factor
# import the numba configuration
from ..config import parallel, cache, fastmath

@numba.njit("(f8[::1], f8[::1], f8[::1], f8[::1], f8[::1], i4[::1], f8[:,:,::1], i8, i4[::1], f8, f8[::1])", 
            parallel=parallel, cache=cache, fastmath=fastmath)
def J1o( xs, x_olds, ws, vys, vzs, l_olds, J,
                   n_threads, indices, q, xidx ):
    """
    Current deposition for linear particle shapes with periodic boundaries
    
    xs        : current particle positions
    x_olds    : old particle positions
    ws        : particle weights
    vys       : particle y velocities
    vzs       : particle z velocities
    l_olds    : old particle left cell indices
    J         : 3D current array
    n_threads : number of parallel threads
    indices   : particle index start/stop for each thread
    q         : particle charge
    xidx      : grid positions

    No returns neccessary as arrays are modified in-place.
    """
    for k in numba.prange(n_threads):
        for j in range( indices[k], indices[k+1] ):   
            
            x = xs[j]
            x_old = x_olds[j]
            w = ws[j]*q

            i = l_olds[j] # l for x_old
            xi = xidx[i] # left grid cell position
       
            if x != x_old: # Jx
     
                dx1 = xi - x
                dx0 = xi - x_old
                
                # integrated NGP factor implemented with intrinsic functions for speed
                J[k,0, i-1] += w*( min(max( dx0-1, -0.5), 0.5) - min(max( dx1-1, -0.5), 0.5) )
                J[k,0, i  ] += w*( min(max( dx0  , -0.5), 0.5) - min(max( dx1  , -0.5), 0.5) )
                J[k,0, i+1] += w*( min(max( dx0+1, -0.5), 0.5) - min(max( dx1+1, -0.5), 0.5) )
                J[k,0, i+2] += w*( min(max( dx0+2, -0.5), 0.5) - min(max( dx1+2, -0.5), 0.5) )
            
            vy = vys[j]
            vz = vzs[j]
            dx = xi + 0.5 - .5*(x+x_old) 
            
            # the linear shape factor is also faster with intrinsics
            if vy != 0.: # Jy
                J[k,1, i-1] += w*vy * max( 0, dx ) 
                J[k,1, i  ] += w*vy * max( 0, 1-abs(dx)   ) 
                J[k,1, i+1] += w*vy * max( 0, -dx ) 
            
            if vz != 0.: # Jz
                J[k,2, i-1] += w*vz * max( 0, dx ) 
                J[k,2, i  ] += w*vz * max( 0, 1-abs(dx)   ) 
                J[k,2, i+1] += w*vz * max( 0, -dx ) 

    return

@numba.njit("(f8[::1], f8[::1], f8[::1], f8[::1], f8[::1], i4[::1], f8[:,:,::1], i8, i4[::1], f8, f8[::1])", 
            parallel=parallel, cache=cache, fastmath=fastmath)
def J2o( xs, x_olds, ws, vys, vzs, l_olds, J,
                   n_threads, indices, q, xidx ):
    """
    Current deposition for quadratic particle shapes with periodic boundaries
    
    xs        : current particle positions
    x_olds    : old particle positions
    ws        : particle weights
    vys       : particle y velocities
    vzs       : particle z velocities
    l_olds    : old particle left cell indices
    J         : 3D current array
    n_threads : number of parallel threads
    indices   : particle index start/stop for each thread
    q         : particle charge
    xidx      : grid positions

    No returns neccessary as arrays are modified in-place.
    """
    for k in numba.prange(n_threads):
        for j in range( indices[k], indices[k+1] ):   
            
            x = xs[j]
            x_old = x_olds[j]
            w = ws[j]*q

            i = l_olds[j] # l for x_old
            xi = xidx[i] # left grid cell position
                    
            if x != x_old: # Jx
            
                dx1 = xi - x
                dx0 = xi - x_old

                J[k,0, i-1] += w*( integrated_linear_shape_factor( dx0-1 ) - integrated_linear_shape_factor( dx1-1 ) )
                J[k,0, i  ] += w*( integrated_linear_shape_factor( dx0   ) - integrated_linear_shape_factor( dx1   ) )
                J[k,0, i+1] += w*( integrated_linear_shape_factor( dx0+1 ) - integrated_linear_shape_factor( dx1+1 ) )
                J[k,0, i+2] += w*( integrated_linear_shape_factor( dx0+2 ) - integrated_linear_shape_factor( dx1+2 ) )
            
            vy = vys[j]
            vz = vzs[j]
            dx = xi + 0.5 - .5*(x+x_old) 
            

            if vy != 0.: # Jy
                J[k,1, i-2] += w*vy * quadratic_shape_factor( 2-dx ) 
                J[k,1, i-1] += w*vy * quadratic_shape_factor( 1-dx ) 
                J[k,1, i  ] += w*vy * quadratic_shape_factor(abs(dx)) 
                J[k,1, i+1] += w*vy * quadratic_shape_factor( 1+dx ) 
                J[k,1, i+2] += w*vy * quadratic_shape_factor( 2+dx ) 
            
            if vz != 0.: # Jz
                J[k,2, i-2] += w*vz * quadratic_shape_factor( 2-dx ) 
                J[k,2, i-1] += w*vz * quadratic_shape_factor( 1-dx ) 
                J[k,2, i  ] += w*vz * quadratic_shape_factor(abs(dx)) 
                J[k,2, i+1] += w*vz * quadratic_shape_factor( 1+dx ) 
                J[k,2, i+2] += w*vz * quadratic_shape_factor( 2+dx )

    return

@numba.njit("(f8[::1], f8[::1], f8[::1], f8[::1], f8[::1], i4[::1], f8[:,:,::1], i8, i4[::1], f8, f8[::1])", 
            parallel=parallel, cache=cache, fastmath=fastmath)
def J3o( xs, x_olds, ws, vys, vzs, l_olds, J,
                   n_threads, indices, q, xidx ):
    """
    Current deposition for cubic particle shapes with periodic boundaries
        
    xs        : current particle positions
    x_olds    : old particle positions
    ws        : particle weights
    vys       : particle y velocities
    vzs       : particle z velocities
    l_olds    : old particle left cell indices
    J         : 3D current array
    n_threads : number of parallel threads
    indices   : particle index start/stop for each thread
    q         : particle charge
    xidx      : grid positions

    No returns neccessary as arrays are modified in-place.
    """
    for k in numba.prange(n_threads):
        for j in range( indices[k], indices[k+1] ):   
            
            x = xs[j]
            x_old = x_olds[j]
            w = ws[j]*q

            i = l_olds[j] # l for x_old
            xi = xidx[i] # left grid cell position
                   
            if x != x_old: # Jx
                dx1 = xi - x
                dx0 = xi - x_old

                J[k,0, i-2] += w*( integrated_quadratic_shape_factor( dx0-2 ) - integrated_quadratic_shape_factor( dx1-2 ) )
                J[k,0, i-1] += w*( integrated_quadratic_shape_factor( dx0-1 ) - integrated_quadratic_shape_factor( dx1-1 ) )
                J[k,0, i  ] += w*( integrated_quadratic_shape_factor( dx0   ) - integrated_quadratic_shape_factor( dx1   ) )
                J[k,0, i+1] += w*( integrated_quadratic_shape_factor( dx0+1 ) - integrated_quadratic_shape_factor( dx1+1 ) )
                J[k,0, i+2] += w*( integrated_quadratic_shape_factor( dx0+2 ) - integrated_quadratic_shape_factor( dx1+2 ) )
                J[k,0, i+3] += w*( integrated_quadratic_shape_factor( dx0+3 ) - integrated_quadratic_shape_factor( dx1+3 ) )
            
            vy = vys[j]
            vz = vzs[j]
            dx = xi + 0.5 - .5*(x+x_old) 
            
            # shape factors expect a positive value, dx can be slightly negative (> -1)
            # when calculating for a cell centre, so abs the central values
            if vy != 0.: # Jy
                J[k,1, i-2] += w*vy * cubic_shape_factor( 2-dx ) 
                J[k,1, i-1] += w*vy * cubic_shape_factor( 1-dx ) 
                J[k,1, i  ] += w*vy * cubic_shape_factor(abs(dx)) 
                J[k,1, i+1] += w*vy * cubic_shape_factor( 1+dx ) 
                J[k,1, i+2] += w*vy * cubic_shape_factor( 2+dx ) 
            
            if vz != 0.: # Jz
                J[k,2, i-2] += w*vz * cubic_shape_factor( 2-dx ) 
                J[k,2, i-1] += w*vz * cubic_shape_factor( 1-dx ) 
                J[k,2, i  ] += w*vz * cubic_shape_factor(abs(dx)) 
                J[k,2, i+1] += w*vz * cubic_shape_factor( 1+dx ) 
                J[k,2, i+2] += w*vz * cubic_shape_factor( 2+dx )

    return

@numba.njit("(f8[::1], f8[::1], f8[::1], f8[::1], f8[::1], i4[::1], f8[:,:,::1], i8, i4[::1], f8, f8[::1])", 
            parallel=parallel, cache=cache, fastmath=fastmath)
def J4o( xs, x_olds, ws, vys, vzs, l_olds, J,
                   n_threads, indices, q, xidx ):
    """
    Current deposition for quartic particle shapes with periodic boundaries
        
    xs        : current particle positions
    x_olds    : old particle positions
    ws        : particle weights
    vys       : particle y velocities
    vzs       : particle z velocities
    l_olds    : old particle left cell indices
    J         : 3D current array
    n_threads : number of parallel threads
    indices   : particle index start/stop for each thread
    q         : particle charge
    xidx      : grid positions

    No returns neccessary as arrays are modified in-place.
    """
    for k in numba.prange(n_threads):
        for j in range( indices[k], indices[k+1] ):   
            
            x = xs[j]
            x_old = x_olds[j]
            w = ws[j]*q

            i = l_olds[j] # l for x_old
            xi = xidx[i] # left grid cell position
   
            if x != x_old: # Jx
                dx1 = xi - x
                dx0 = xi - x_old
                
                J[k,0, i-2] += w*( integrated_cubic_shape_factor( dx0-2 ) - integrated_cubic_shape_factor( dx1-2 ) )
                J[k,0, i-1] += w*( integrated_cubic_shape_factor( dx0-1 ) - integrated_cubic_shape_factor( dx1-1 ) )
                J[k,0, i  ] += w*( integrated_cubic_shape_factor( dx0   ) - integrated_cubic_shape_factor( dx1   ) )
                J[k,0, i+1] += w*( integrated_cubic_shape_factor( dx0+1 ) - integrated_cubic_shape_factor( dx1+1 ) )
                J[k,0, i+2] += w*( integrated_cubic_shape_factor( dx0+2 ) - integrated_cubic_shape_factor( dx1+2 ) )
                J[k,0, i+3] += w*( integrated_cubic_shape_factor( dx0+3 ) - integrated_cubic_shape_factor( dx1+3 ) )
            
            vy = vys[j]
            vz = vzs[j]
            dx = xi + 0.5 - .5*(x+x_old) 
            
            # shape factors expect a positive value, dx can be slightly negative (> -1)
            # when calculating for a cell centre, so abs the central values
            if vy != 0.: # Jy
                J[k,1, i-3] += w*vy * quartic_shape_factor( 3-dx )
                J[k,1, i-2] += w*vy * quartic_shape_factor( 2-dx ) 
                J[k,1, i-1] += w*vy * quartic_shape_factor( 1-dx ) 
                J[k,1, i  ] += w*vy * quartic_shape_factor(abs(dx)) 
                J[k,1, i+1] += w*vy * quartic_shape_factor( 1+dx ) 
                J[k,1, i+2] += w*vy * quartic_shape_factor( 2+dx ) 
                J[k,1, i+3] += w*vy * quartic_shape_factor( 3+dx )
            
            if vz != 0.: # Jz
                J[k,2, i-3] += w*vz * quartic_shape_factor( 3-dx )
                J[k,2, i-2] += w*vz * quartic_shape_factor( 2-dx ) 
                J[k,2, i-1] += w*vz * quartic_shape_factor( 1-dx ) 
                J[k,2, i  ] += w*vz * quartic_shape_factor(abs(dx)) 
                J[k,2, i+1] += w*vz * quartic_shape_factor( 1+dx ) 
                J[k,2, i+2] += w*vz * quartic_shape_factor( 2+dx )
                J[k,2, i+3] += w*vz * quartic_shape_factor( 3+dx )

    return
@numba.njit("(f8[::1], f8[::1], f8[::1], f8[::1], f8[::1], i4[::1], f8[:,:,::1], i8, i4[::1], f8, f8[::1])", 
            parallel=parallel, cache=cache, fastmath=fastmath)
def J1p( xs, x_olds, ws, vys, vzs, l_olds, J,
                   n_threads, indices, q, xidx ):
    """
    Current deposition for linear particle shapes with periodic boundaries
    
    xs        : current particle positions
    x_olds    : old particle positions
    ws        : particle weights
    vys       : particle y velocities
    vzs       : particle z velocities
    l_olds    : old particle left cell indices
    J         : 3D current array
    n_threads : number of parallel threads
    indices   : particle index start/stop for each thread
    q         : particle charge
    xidx      : grid positions

    No returns neccessary as arrays are modified in-place.
    """
    Nx = len(xidx)
    
    for k in numba.prange(n_threads):
        for j in range( indices[k], indices[k+1] ):   
            
            x = xs[j]
            x_old = x_olds[j]
            w = ws[j]*q

            i = l_olds[j] # l for x_old
            xi = xidx[i] # left grid cell position
            
            ip1 = i+1
            ip2 = i+2

            # check for particles that have traversed the grid edges, 
            # and fix their displacement. then manually set the correct indices
            if i == 0:  
                if abs(x - x_old) > 1:
                    x = x_old - x_old%1 - (1-x%1)
 
            elif i == Nx-1:
                ip1 = 0
                ip2 = 1 
                if abs(x - x_old) > 1:   
                    x = x_old + (1-x_old%1) + x%1
                    
            elif i == Nx-2:
                ip2 = 0
                    
            if x != x_old: # Jx
     
                dx1 = xi - x
                dx0 = xi - x_old
                
                # integrated NGP factor implemented with intrinsic functions for speed
                J[k,0, i-1] += w*( min(max( dx0-1, -0.5), 0.5) - min(max( dx1-1, -0.5), 0.5) )
                J[k,0, i  ] += w*( min(max( dx0  , -0.5), 0.5) - min(max( dx1  , -0.5), 0.5) )
                J[k,0, ip1] += w*( min(max( dx0+1, -0.5), 0.5) - min(max( dx1+1, -0.5), 0.5) )
                J[k,0, ip2] += w*( min(max( dx0+2, -0.5), 0.5) - min(max( dx1+2, -0.5), 0.5) )
            
            vy = vys[j]
            vz = vzs[j]
            dx = xi + 0.5 - .5*(x+x_old) 
            
            # the linear shape factor is also faster with intrinsics
            if vy != 0.: # Jy
                J[k,1, i-1] += w*vy * max( 0, dx ) 
                J[k,1, i  ] += w*vy * max( 0, 1-abs(dx)   ) 
                J[k,1, ip1] += w*vy * max( 0, -dx ) 
            
            if vz != 0.: # Jz
                J[k,2, i-1] += w*vz * max( 0, dx ) 
                J[k,2, i  ] += w*vz * max( 0, 1-abs(dx)   ) 
                J[k,2, ip1] += w*vz * max( 0, -dx ) 

    return

@numba.njit("(f8[::1], f8[::1], f8[::1], f8[::1], f8[::1], i4[::1], f8[:,:,::1], i8, i4[::1], f8, f8[::1])", 
            parallel=parallel, cache=cache, fastmath=fastmath)
def J2p( xs, x_olds, ws, vys, vzs, l_olds, J,
                   n_threads, indices, q, xidx ):
    """
    Current deposition for quadratic particle shapes with periodic boundaries
    
    xs        : current particle positions
    x_olds    : old particle positions
    ws        : particle weights
    vys       : particle y velocities
    vzs       : particle z velocities
    l_olds    : old particle left cell indices
    J         : 3D current array
    n_threads : number of parallel threads
    indices   : particle index start/stop for each thread
    q         : particle charge
    xidx      : grid positions

    No returns neccessary as arrays are modified in-place.
    """
    Nx = len(xidx)

    for k in numba.prange(n_threads):
        for j in range( indices[k], indices[k+1] ):   
            
            x = xs[j]
            x_old = x_olds[j]
            w = ws[j]*q

            i = l_olds[j] # l for x_old
            xi = xidx[i] # left grid cell position
            
            ip1 = i+1
            ip2 = i+2

            # check for particles that have traversed the grid edges, 
            # and fix their displacement. then manually set the correct indices
            if i == 0:  
                if abs(x - x_old) > 1:
                    x = x_old - x_old%1 - (1-x%1)
 
            elif i == Nx-1:
                ip1 = 0
                ip2 = 1 
                if abs(x - x_old) > 1:   
                    x = x_old + (1-x_old%1) + x%1
                    
            elif i == Nx-2:
                ip2 = 0
                    
            if x != x_old: # Jx
            
                dx1 = xi - x
                dx0 = xi - x_old

                J[k,0, i-1] += w*( integrated_linear_shape_factor( dx0-1 ) - integrated_linear_shape_factor( dx1-1 ) )
                J[k,0, i  ] += w*( integrated_linear_shape_factor( dx0   ) - integrated_linear_shape_factor( dx1   ) )
                J[k,0, ip1] += w*( integrated_linear_shape_factor( dx0+1 ) - integrated_linear_shape_factor( dx1+1 ) )
                J[k,0, ip2] += w*( integrated_linear_shape_factor( dx0+2 ) - integrated_linear_shape_factor( dx1+2 ) )
            
            vy = vys[j]
            vz = vzs[j]
            dx = xi + 0.5 - .5*(x+x_old) 
            

            if vy != 0.: # Jy
                J[k,1, i-2] += w*vy * quadratic_shape_factor( 2-dx ) 
                J[k,1, i-1] += w*vy * quadratic_shape_factor( 1-dx ) 
                J[k,1, i  ] += w*vy * quadratic_shape_factor(abs(dx)) 
                J[k,1, ip1] += w*vy * quadratic_shape_factor( 1+dx ) 
                J[k,1, ip2] += w*vy * quadratic_shape_factor( 2+dx ) 
            
            if vz != 0.: # Jz
                J[k,2, i-2] += w*vz * quadratic_shape_factor( 2-dx ) 
                J[k,2, i-1] += w*vz * quadratic_shape_factor( 1-dx ) 
                J[k,2, i  ] += w*vz * quadratic_shape_factor(abs(dx)) 
                J[k,2, ip1] += w*vz * quadratic_shape_factor( 1+dx ) 
                J[k,2, ip2] += w*vz * quadratic_shape_factor( 2+dx )

    return

@numba.njit("(f8[::1], f8[::1], f8[::1], f8[::1], f8[::1], i4[::1], f8[:,:,::1], i8, i4[::1], f8, f8[::1])", 
            parallel=parallel, cache=cache, fastmath=fastmath)
def J3p( xs, x_olds, ws, vys, vzs, l_olds, J,
                   n_threads, indices, q, xidx ):
    """
    Current deposition for cubic particle shapes with periodic boundaries
        
    xs        : current particle positions
    x_olds    : old particle positions
    ws        : particle weights
    vys       : particle y velocities
    vzs       : particle z velocities
    l_olds    : old particle left cell indices
    J         : 3D current array
    n_threads : number of parallel threads
    indices   : particle index start/stop for each thread
    q         : particle charge
    xidx      : grid positions

    No returns neccessary as arrays are modified in-place.
    """
    Nx = len(xidx)

    for k in numba.prange(n_threads):
        for j in range( indices[k], indices[k+1] ):   
            
            x = xs[j]
            x_old = x_olds[j]
            w = ws[j]*q

            i = l_olds[j] # l for x_old
            xi = xidx[i] # left grid cell position
            
            ip1 = i+1
            ip2 = i+2
            ip3 = i+3
            
            # check for particles that have traversed the grid edges, 
            # and fix their displacement. then manually set the correct indices
            if i == 0 and abs(x - x_old) > 1:
                x = x_old - x_old%1 - (1-x%1)
            
            elif i == Nx-1:
                ip1 = 0
                ip2 = 1 
                ip3 = 2
                if abs(x - x_old) > 1:   
                    x = x_old + (1-x_old%1) + x%1
            
            elif i == Nx-2:
                ip2 = 0
                ip3 = 1
                
            elif i == Nx-3:
                ip3 = 0
                    
            if x != x_old: # Jx
                dx1 = xi - x
                dx0 = xi - x_old

                J[k,0, i-2] += w*( integrated_quadratic_shape_factor( dx0-2 ) - integrated_quadratic_shape_factor( dx1-2 ) )
                J[k,0, i-1] += w*( integrated_quadratic_shape_factor( dx0-1 ) - integrated_quadratic_shape_factor( dx1-1 ) )
                J[k,0, i  ] += w*( integrated_quadratic_shape_factor( dx0   ) - integrated_quadratic_shape_factor( dx1   ) )
                J[k,0, ip1] += w*( integrated_quadratic_shape_factor( dx0+1 ) - integrated_quadratic_shape_factor( dx1+1 ) )
                J[k,0, ip2] += w*( integrated_quadratic_shape_factor( dx0+2 ) - integrated_quadratic_shape_factor( dx1+2 ) )
                J[k,0, ip3] += w*( integrated_quadratic_shape_factor( dx0+3 ) - integrated_quadratic_shape_factor( dx1+3 ) )
            
            vy = vys[j]
            vz = vzs[j]
            dx = xi + 0.5 - .5*(x+x_old) 
            
            # shape factors expect a positive value, dx can be slightly negative (> -1)
            # when calculating for a cell centre, so abs the central values
            if vy != 0.: # Jy
                J[k,1, i-2] += w*vy * cubic_shape_factor( 2-dx ) 
                J[k,1, i-1] += w*vy * cubic_shape_factor( 1-dx ) 
                J[k,1, i  ] += w*vy * cubic_shape_factor(abs(dx)) 
                J[k,1, ip1] += w*vy * cubic_shape_factor( 1+dx ) 
                J[k,1, ip2] += w*vy * cubic_shape_factor( 2+dx ) 
            
            if vz != 0.: # Jz
                J[k,2, i-2] += w*vz * cubic_shape_factor( 2-dx ) 
                J[k,2, i-1] += w*vz * cubic_shape_factor( 1-dx ) 
                J[k,2, i  ] += w*vz * cubic_shape_factor(abs(dx)) 
                J[k,2, ip1] += w*vz * cubic_shape_factor( 1+dx ) 
                J[k,2, ip2] += w*vz * cubic_shape_factor( 2+dx )

    return

@numba.njit("(f8[::1], f8[::1], f8[::1], f8[::1], f8[::1], i4[::1], f8[:,:,::1], i8, i4[::1], f8, f8[::1])", 
            parallel=parallel, cache=cache, fastmath=fastmath)
def J4p( xs, x_olds, ws, vys, vzs, l_olds, J,
                   n_threads, indices, q, xidx ):
    """
    Current deposition for quartic particle shapes with periodic boundaries
        
    xs        : current particle positions
    x_olds    : old particle positions
    ws        : particle weights
    vys       : particle y velocities
    vzs       : particle z velocities
    l_olds    : old particle left cell indices
    J         : 3D current array
    n_threads : number of parallel threads
    indices   : particle index start/stop for each thread
    q         : particle charge
    xidx      : grid positions

    No returns neccessary as arrays are modified in-place.
    """
    Nx = len(xidx)  
    
    for k in numba.prange(n_threads):
        for j in range( indices[k], indices[k+1] ):   
            
            x = xs[j]
            x_old = x_olds[j]
            w = ws[j]*q

            i = l_olds[j] # l for x_old
            xi = xidx[i] # left grid cell position
            
            ip1 = i+1
            ip2 = i+2
            ip3 = i+3
            
            # check for particles that have traversed the grid edges, 
            # and fix their displacement. then manually set the correct indices
            if i == 0 and abs(x - x_old) > 1:
                x = x_old - x_old%1 - (1-x%1)
            
            elif i == Nx-1:
                ip1 = 0
                ip2 = 1 
                ip3 = 2
                if abs(x - x_old) > 1:   
                    x = x_old + (1-x_old%1) + x%1
            
            elif i == Nx-2:
                ip2 = 0
                ip3 = 1
                
            elif i == Nx-3:
                ip3 = 0
                    
            if x != x_old: # Jx
                dx1 = xi - x
                dx0 = xi - x_old
                
                J[k,0, i-2] += w*( integrated_cubic_shape_factor( dx0-2 ) - integrated_cubic_shape_factor( dx1-2 ) )
                J[k,0, i-1] += w*( integrated_cubic_shape_factor( dx0-1 ) - integrated_cubic_shape_factor( dx1-1 ) )
                J[k,0, i  ] += w*( integrated_cubic_shape_factor( dx0   ) - integrated_cubic_shape_factor( dx1   ) )
                J[k,0, ip1] += w*( integrated_cubic_shape_factor( dx0+1 ) - integrated_cubic_shape_factor( dx1+1 ) )
                J[k,0, ip2] += w*( integrated_cubic_shape_factor( dx0+2 ) - integrated_cubic_shape_factor( dx1+2 ) )
                J[k,0, ip3] += w*( integrated_cubic_shape_factor( dx0+3 ) - integrated_cubic_shape_factor( dx1+3 ) )
            
            vy = vys[j]
            vz = vzs[j]
            dx = xi + 0.5 - .5*(x+x_old) 
            
            # shape factors expect a positive value, dx can be slightly negative (> -1)
            # when calculating for a cell centre, so abs the central values
            if vy != 0.: # Jy
                J[k,1, i-3] += w*vy * quartic_shape_factor( 3-dx )
                J[k,1, i-2] += w*vy * quartic_shape_factor( 2-dx ) 
                J[k,1, i-1] += w*vy * quartic_shape_factor( 1-dx ) 
                J[k,1, i  ] += w*vy * quartic_shape_factor(abs(dx)) 
                J[k,1, ip1] += w*vy * quartic_shape_factor( 1+dx ) 
                J[k,1, ip2] += w*vy * quartic_shape_factor( 2+dx ) 
                J[k,1, ip3] += w*vy * quartic_shape_factor( 3+dx )
            
            if vz != 0.: # Jz
                J[k,2, i-3] += w*vz * quartic_shape_factor( 3-dx )
                J[k,2, i-2] += w*vz * quartic_shape_factor( 2-dx ) 
                J[k,2, i-1] += w*vz * quartic_shape_factor( 1-dx ) 
                J[k,2, i  ] += w*vz * quartic_shape_factor(abs(dx)) 
                J[k,2, ip1] += w*vz * quartic_shape_factor( 1+dx ) 
                J[k,2, ip2] += w*vz * quartic_shape_factor( 2+dx )
                J[k,2, ip3] += w*vz * quartic_shape_factor( 3+dx )

    return