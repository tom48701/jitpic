import numba
import numpy as np

# toggles for numba options
parallel = True
fastmath = True
cache = True

# define the iecewise functions for the various particle shapes
# assume positive argument values only
# (0th and 1st order shapes are simple enough that they are faster to implement using intrinsic functions)
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

@numba.njit("(f8[::1], f8[::1], f8[::1], f8[::1], f8[::1], i4[::1], f8[:,:,::1], i8, i4[::1], f8, f8[::1])", 
            parallel=parallel, cache=cache, fastmath=fastmath)
def deposit_J_linear_numba( xs, x_olds, ws, vys, vzs, l_olds, J,
                   n_threads, indices, q, xidx ):
    """
    Current deposition for linear particle shapes
    
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
            
            if x == x_old: # no displacement? no current; continue
                continue
             
            w = ws[j]*q
            vy = vys[j]
            vz = vzs[j]
            
            i = int(l_olds[j]) # l for x_old
            xi = xidx[i] # left grid cell position

            dx1 = xi - x
            dx0 = xi - x_old
            
            # integrated NGP factor implemented with intrinsic functions for speed
            J[k,0, i-1] += w*( min(max( dx0-1, -0.5), 0.5) - min(max( dx1-1, -0.5), 0.5) )
            J[k,0, i  ] += w*( min(max( dx0  , -0.5), 0.5) - min(max( dx1  , -0.5), 0.5) )
            J[k,0, i+1] += w*( min(max( dx0+1, -0.5), 0.5) - min(max( dx1+1, -0.5), 0.5) )
            J[k,0, i+2] += w*( min(max( dx0+2, -0.5), 0.5) - min(max( dx1+2, -0.5), 0.5) )
            
            dx = xi + 0.5 - .5*(x+x_old) 
            
            # linear shape factors are faster to implement with intrinsic functions
            J[k,1, i-1] += w*vy * max( 0, dx ) 
            J[k,1, i  ] += w*vy * max( 0, 1-abs(dx)   ) 
            J[k,1, i+1] += w*vy * max( 0, -dx ) 

            J[k,2, i-1] += w*vz * max( 0, dx ) 
            J[k,2, i  ] += w*vz * max( 0, 1-abs(dx)   ) 
            J[k,2, i+1] += w*vz * max( 0, -dx ) 

    return

@numba.njit("(f8[::1], f8[::1], f8[::1], f8[::1], f8[::1], i4[::1], f8[:,:,::1], i8, i4[::1], f8, f8[::1])", 
            parallel=parallel, cache=cache, fastmath=fastmath)
def deposit_J_quadratic_numba( xs, x_olds, ws, vys, vzs, l_olds, J,
                   n_threads, indices, q, xidx ):
    """
    Current deposition for quadratic particle shapes
    
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
    shape_factor = integrated_linear_shape_factor  
    shape_factor_unint = quadratic_shape_factor
    
    for k in numba.prange(n_threads):
        
        for j in range( indices[k], indices[k+1] ):   
            
            x = xs[j]
            x_old = x_olds[j]
            
            if x == x_old: # no displacement? no current; continue
                continue
             
            w = ws[j]*q
            vy = vys[j]
            vz = vzs[j]
            
            i = int(l_olds[j]) # l for x_old
            xi = xidx[i] # left grid cell position

            dx1 = xi - x
            dx0 = xi - x_old

            J[k,0, i-1] += w*( shape_factor( dx0-1 ) - shape_factor( dx1-1 ) )
            J[k,0, i  ] += w*( shape_factor( dx0   ) - shape_factor( dx1   ) )
            J[k,0, i+1] += w*( shape_factor( dx0+1 ) - shape_factor( dx1+1 ) )
            J[k,0, i+2] += w*( shape_factor( dx0+2 ) - shape_factor( dx1+2 ) )
            
            dx = xi + 0.5 - .5*(x+x_old) 
            
            # shape factors expect a positive value, dx can be slightly negative (> -1)
            # when calculating for a cell centre, so abs the central values
            J[k,1, i-2] += w*vy * shape_factor_unint( 2-dx ) 
            J[k,1, i-1] += w*vy * shape_factor_unint( 1-dx ) 
            J[k,1, i  ] += w*vy * shape_factor_unint(abs(dx)) 
            J[k,1, i+1] += w*vy * shape_factor_unint( 1+dx ) 
            J[k,1, i+2] += w*vy * shape_factor_unint( 2+dx ) 

            J[k,2, i-2] += w*vz * shape_factor_unint( 2-dx ) 
            J[k,2, i-1] += w*vz * shape_factor_unint( 1-dx ) 
            J[k,2, i  ] += w*vz * shape_factor_unint(abs(dx)) 
            J[k,2, i+1] += w*vz * shape_factor_unint( 1+dx ) 
            J[k,2, i+2] += w*vz * shape_factor_unint( 2+dx )
    return

@numba.njit("(f8[::1], f8[::1], f8[::1], f8[::1], f8[::1], i4[::1], f8[:,:,::1], i8, i4[::1], f8, f8[::1])", 
            parallel=parallel, cache=cache, fastmath=fastmath)
def deposit_J_cubic_numba( xs, x_olds, ws, vys, vzs, l_olds, J,
                   n_threads, indices, q, xidx ):
    """
    Current deposition for cubic particle shapes
        
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
    shape_factor = integrated_quadratic_shape_factor  
    shape_factor_unint = cubic_shape_factor
    
    for k in numba.prange(n_threads):
        
        for j in range( indices[k], indices[k+1] ):   
            
            x = xs[j]
            x_old = x_olds[j]
            
            if x == x_old: # no displacement? no current; continue
                continue
             
            w = ws[j]*q
            vy = vys[j]
            vz = vzs[j]
            
            i = int(l_olds[j]) # l for x_old
            xi = xidx[i] # left grid cell position

            dx1 = xi - x
            dx0 = xi - x_old
            
            J[k,0, i-2] += w*( shape_factor( dx0-2 ) - shape_factor( dx1-2 ) )
            J[k,0, i-1] += w*( shape_factor( dx0-1 ) - shape_factor( dx1-1 ) )
            J[k,0, i  ] += w*( shape_factor( dx0   ) - shape_factor( dx1   ) )
            J[k,0, i+1] += w*( shape_factor( dx0+1 ) - shape_factor( dx1+1 ) )
            J[k,0, i+2] += w*( shape_factor( dx0+2 ) - shape_factor( dx1+2 ) )
            J[k,0, i+3] += w*( shape_factor( dx0+3 ) - shape_factor( dx1+3 ) )
            
            dx = xi + 0.5 - .5*(x+x_old) 
            
            # shape factors expect a positive value, dx can be slightly negative (> -1)
            # when calculating for a cell centre, so abs the central values
            J[k,1, i-2] += w*vy * shape_factor_unint( 2-dx ) 
            J[k,1, i-1] += w*vy * shape_factor_unint( 1-dx ) 
            J[k,1, i  ] += w*vy * shape_factor_unint(abs(dx)) 
            J[k,1, i+1] += w*vy * shape_factor_unint( 1+dx ) 
            J[k,1, i+2] += w*vy * shape_factor_unint( 2+dx ) 

            J[k,2, i-2] += w*vz * shape_factor_unint( 2-dx ) 
            J[k,2, i-1] += w*vz * shape_factor_unint( 1-dx ) 
            J[k,2, i  ] += w*vz * shape_factor_unint(abs(dx)) 
            J[k,2, i+1] += w*vz * shape_factor_unint( 1+dx ) 
            J[k,2, i+2] += w*vz * shape_factor_unint( 2+dx )

    return

@numba.njit("(f8[::1], f8[::1], f8[::1], f8[::1], f8[::1], i4[::1], f8[:,:,::1], i8, i4[::1], f8, f8[::1])", 
            parallel=parallel, cache=cache, fastmath=fastmath)
def deposit_J_quartic_numba( xs, x_olds, ws, vys, vzs, l_olds, J,
                   n_threads, indices, q, xidx ):
    """
    Current deposition for quartic particle shapes
        
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
    
    shape_factor = integrated_cubic_shape_factor
    shape_factor_unint = quartic_shape_factor
    
    for k in numba.prange(n_threads):
        
        for j in range( indices[k], indices[k+1] ):   
            
            x = xs[j]
            x_old = x_olds[j]

            if x == x_old: # no displacement? no current; continue
                continue
               
            w = ws[j]*q
            vy = vys[j]
            vz = vzs[j]
            
            i = l_olds[j] # l for x_old
            xi = xidx[i] # left grid cell position

            dx1 = xi - x
            dx0 = xi - x_old
            
            J[k,0, i-2] += w*( shape_factor( dx0-2 ) - shape_factor( dx1-2 ) )
            J[k,0, i-1] += w*( shape_factor( dx0-1 ) - shape_factor( dx1-1 ) )
            J[k,0, i  ] += w*( shape_factor( dx0   ) - shape_factor( dx1   ) )
            J[k,0, i+1] += w*( shape_factor( dx0+1 ) - shape_factor( dx1+1 ) )
            J[k,0, i+2] += w*( shape_factor( dx0+2 ) - shape_factor( dx1+2 ) )
            J[k,0, i+3] += w*( shape_factor( dx0+3 ) - shape_factor( dx1+3 ) )
            
            dx = xi + 0.5 - .5*(x+x_old) 
            
            # shape factors expect a positive value, dx can be slightly negative (> -1)
            # when calculating for a cell centre, so abs the central values
            
            J[k,1, i-3] += w*vy * shape_factor_unint( 3-dx )
            J[k,1, i-2] += w*vy * shape_factor_unint( 2-dx ) 
            J[k,1, i-1] += w*vy * shape_factor_unint( 1-dx ) 
            J[k,1, i  ] += w*vy * shape_factor_unint(abs(dx)) 
            J[k,1, i+1] += w*vy * shape_factor_unint( 1+dx ) 
            J[k,1, i+2] += w*vy * shape_factor_unint( 2+dx ) 
            J[k,1, i+3] += w*vy * shape_factor_unint( 3+dx )

            J[k,2, i-3] += w*vz * shape_factor_unint( 3-dx )
            J[k,2, i-2] += w*vz * shape_factor_unint( 2-dx ) 
            J[k,2, i-1] += w*vz * shape_factor_unint( 1-dx ) 
            J[k,2, i  ] += w*vz * shape_factor_unint(abs(dx)) 
            J[k,2, i+1] += w*vz * shape_factor_unint( 1+dx ) 
            J[k,2, i+2] += w*vz * shape_factor_unint( 2+dx )
            J[k,2, i+3] += w*vz * shape_factor_unint( 3+dx )

    return
    
@numba.njit("(i8, i8, f8[::1], f8, f8[::1], f8[:,::1], u4[::1], u4[::1], i4[::1], f8[::1], i4)", 
            parallel=parallel, cache=cache, fastmath=fastmath)
def deposit_rho_linear_numba(N, n_threads, x, idx, qw, rho, l, r, indices, xg, Nx ):
    """
    Charge density deposition for linear particle shapes
    
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
def deposit_rho_quadratic_numba(N, n_threads, x, idx, qw, rho, l, r, indices, xg, Nx ):
    """
    Charge density deposition for quadratic particle shapes

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
def deposit_rho_cubic_numba(N, n_threads, x, idx, qw, rho, l, r, indices, xg, Nx ):
    """
    Charge density deposition for cubic particle shapes

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
def deposit_rho_quartic_numba(N, n_threads, x, idx, qw, rho, l, r, indices, xg, Nx ):
    """
    Charge density deposition for quartic particle shapes

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

@numba.njit("(f8[:,::1], f8[:,::1], f8, f8[:,::1], f8[:,::1], f8[::1], f8[::1], f8[::1], f8, f8, i8, b1)", 
            parallel=parallel, cache=cache, fastmath=fastmath)
def cohen_numba( E, B, qmdt, p, v, x, x_old, rg, m, dt, N, backstep=False):
    """
    Cohen particle push
    
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
    backstep : wether or not to push x as well as p,v (for initial setup)
    
    No returns neccessary as arrays are modified in-place.
    """

    for i in numba.prange(N):

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
        
        if not backstep:
            # update x
            x[i] = x[i] + v[0,i] * dt
         
    return

@numba.njit("(f8[:,::1], f8[:,::1], f8, f8[:,::1], f8[:,::1], f8[::1], f8[::1], f8[::1], f8, f8, i8, b1)", 
            parallel=parallel, cache=cache, fastmath=fastmath)
def boris_numba( E, B, qmdt2, p, v, x, x_old, rg, m, dt, N, backstep=False):
    """
    Boris particle push
    
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
    backstep : wether or not to push x as well as p,v (for initial setup)
    
    No returns neccessary as arrays are modified in-place.
    """
    
    for i in numba.prange(N):

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
        
        if not backstep:
            # update x
            x[i] = x[i] + v[0,i] * dt
         
    return

@numba.njit("(i8, f8[::1], b1[::1], u4[::1], u4[::1], f8, f8, f8, i8)", 
            parallel=parallel, cache=cache, fastmath=fastmath)
def reseat_numba(N,x,state,l,r,x0,x1,idx,Nx ):
    """
    Update the left and right indices for particles
    after a push, then look for particles that have 
    been pushed off the grid and disable them.
    
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
    
    l[:] = np.floor((x-x0)*idx) # left cell
    r[:] = l+1 # right cell   
        
    for i in numba.prange(N):
        if x[i] >= x1 or x[i] < x0:
            #print( 'particle out of range')
            state[i] = False

            l[i] = 0
            r[i] = 0 # particles must have valid indices even if they're dead
 
    return

@numba.njit("(f8[:,::1], f8[:,::1], f8[:,::1], f8[:,::1], u4[::1], u4[::1], f8[::1], i4)", 
            parallel=parallel, cache=cache, fastmath=fastmath)
def interpolate_linear_numba(Eg,Bg, Es,Bs, l,r, x, N):
    """
    Interpolate fields from the grid onto particles
    
    Eg : grid E field
    Eb : grid B field
    Es : particle E field
    Bs : particle B field
    l  : left cell indices
    r  : right cell indices
    x  : normalised particle positions
    N  : number of particles

    No return neccessary as arrays are modified in-place.
    """
    for i in numba.prange(N):

        xi = x[i] - l[i]

        Es[:,i] = (1-xi)*Eg[:,l[i]] + xi*Eg[:,r[i]]
        Bs[:,i] = (1-xi)*Bg[:,l[i]] + xi*Bg[:,r[i]]
        
    return

@numba.njit("(f8[:,::1], f8[:,::1], f8[:,::1], f8[:,::1], u4[::1], u4[::1], f8[::1], i4)", 
            parallel=parallel, cache=cache, fastmath=fastmath)
def interpolate_quadratic_numba(Eg,Bg, Es,Bs, l,r, x, N):
    """
    Interpolate fields from the grid onto particles
    
    Eg : grid E field
    Eb : grid B field
    Es : particle E field
    Bs : particle B field
    l  : left cell indices
    r  : right cell indices
    x  : normalised particle positions
    N  : number of particles

    No return neccessary as arrays are modified in-place.
    """
    
    for i in numba.prange(N):

        xi = x[i] - l[i]

        Es[:,i] = Eg[:,l[i]-1]*quadratic_shape_factor(   xi+1 ) + \
                  Eg[:,l[i]  ]*quadratic_shape_factor(   xi   ) + \
                  Eg[:,r[i]  ]*quadratic_shape_factor( 1-xi   ) + \
                  Eg[:,r[i]+1]*quadratic_shape_factor( 2-xi   ) 

                  
        Bs[:,i] = Bg[:,l[i]-1]*quadratic_shape_factor(   xi+1 ) + \
                  Bg[:,l[i]  ]*quadratic_shape_factor(   xi   ) + \
                  Bg[:,r[i]  ]*quadratic_shape_factor( 1-xi   ) + \
                  Bg[:,r[i]+1]*quadratic_shape_factor( 2-xi   ) 

    return

@numba.njit("(f8[:,::1], f8[:,::1], f8[:,::1], f8[:,::1], u4[::1], u4[::1], f8[::1], i4)", 
            parallel=parallel, cache=cache, fastmath=fastmath)
def interpolate_cubic_numba(Eg,Bg, Es,Bs, l,r, x, N):
    """
    Interpolate fields from the grid onto particles
    
    Eg : grid E field
    Eb : grid B field
    Es : particle E field
    Bs : particle B field
    l  : left cell indices
    r  : right cell indices
    x  : normalised particle positions
    N  : number of particles

    No return neccessary as arrays are modified in-place.
    """
    
    for i in numba.prange(N):

        xi = x[i] - l[i]

        Es[:,i] = Eg[:,l[i]-1]*cubic_shape_factor(   xi+1 ) + \
                  Eg[:,l[i]  ]*cubic_shape_factor(   xi   ) + \
                  Eg[:,r[i]  ]*cubic_shape_factor( 1-xi   ) + \
                  Eg[:,r[i]+1]*cubic_shape_factor( 2-xi   ) 

                  
        Bs[:,i] = Bg[:,l[i]-1]*cubic_shape_factor(   xi+1 ) + \
                  Bg[:,l[i]  ]*cubic_shape_factor(   xi   ) + \
                  Bg[:,r[i]  ]*cubic_shape_factor( 1-xi   ) + \
                  Bg[:,r[i]+1]*cubic_shape_factor( 2-xi   ) 


    return

@numba.njit("(f8[:,::1], f8[:,::1], f8[:,::1], f8[:,::1], u4[::1], u4[::1], f8[::1], i4)", 
            parallel=parallel, cache=cache, fastmath=fastmath)
def interpolate_quartic_numba(Eg,Bg, Es,Bs, l,r, x, N):
    """
    Interpolate fields from the grid onto particles
    
    Eg : grid E field
    Eb : grid B field
    Es : particle E field
    Bs : particle B field
    l  : left cell indices
    r  : right cell indices
    x  : normalised particle positions
    N  : number of particles

    No return neccessary as arrays are modified in-place.
    """
    
    for i in numba.prange(N):

        xi = x[i] - l[i]
        
        Es[:,i] = Eg[:,l[i]-2]*quartic_shape_factor(   xi+2 ) + \
                  Eg[:,l[i]-1]*quartic_shape_factor(   xi+1 ) + \
                  Eg[:,l[i]  ]*quartic_shape_factor(   xi   ) + \
                  Eg[:,r[i]  ]*quartic_shape_factor( 1-xi   ) + \
                  Eg[:,r[i]+1]*quartic_shape_factor( 2-xi   ) + \
                  Eg[:,r[i]+2]*quartic_shape_factor( 3-xi   ) 

                  
        Bs[:,i] = Bg[:,l[i]-2]*quartic_shape_factor(   xi+2 ) + \
                  Bg[:,l[i]-1]*quartic_shape_factor(   xi+1 ) + \
                  Bg[:,l[i]  ]*quartic_shape_factor(   xi   ) + \
                  Bg[:,r[i]  ]*quartic_shape_factor( 1-xi   ) + \
                  Bg[:,r[i]+1]*quartic_shape_factor( 2-xi   ) + \
                  Bg[:,r[i]+2]*quartic_shape_factor( 3-xi   ) 

    return
   