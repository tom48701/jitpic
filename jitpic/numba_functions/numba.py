import numba
import numpy as np

@numba.njit()
def ngp_shape_factor(x):
    if x < -0.5:
        return 0
    if x < 0.5:
        return 1
    else:
        return 0
       
@numba.njit()
def linear_shape_factor(x):
    if x < -1:
        return 0
    if x < 0:
        return x+1
    elif x < 1:
        return 1-x
    else:
        return 0
    
@numba.njit()
def quadratic_shape_factor(x):
    if x < -1.5:
        return 0
    elif x < -0.5:
        return (x**2+3*x+9/4.)/2.
    elif x < 0.5:
        return 0.75-x**2
    elif x < 1.5:
        return (x**2-3*x+9/4.)/2.
    else:
        return 0

@numba.njit() 
def cubic_shape_factor(x):
    if x < -2:
        return 0
    elif x < -1:
        return (x**3+6*x**2+12*x+8)/6.
    elif x < 0:
        return (-3*x**3-6*x**2+4)/6.
    elif x < 1:
        return (3*x**3-6*x**2+4)/6.
    elif x < 2:
        return (-x**3+6*x**2-12*x+8)/6.
    else: 
        return 0

@numba.njit() 
def quartic_shape_factor(x):
    if x < -2.5:
        return 0
    elif x < -1.5:
        return (2*x+5)**4/384.
    elif x < -0.5:
        return -(16*x**4+80*x**3+120*x**2+20*x-55)/96.
    elif x < 0.5:
        return (48*x**4-120*x**2+115)/192.
    elif x < 1.5:
        return (-16*x**4+80*x**3-120*x**2+20*x+55)/96.
    elif x < 2.5:
        return (2*x-5)**4/384.
    else: 
        return 0
    
@numba.njit()
def integrated_ngp_shape_factor(x):    
    if x < -0.5:
        return -0.5
    elif x < 0.5:
        return x
    else:# x>1:
        return 0.5 
    
@numba.njit()
def integrated_linear_shape_factor(x):    
    if x < -1:
        return -0.5
    elif x < 0:
        return .5*x**2+x
    elif x < 1:
        return x-.5*x**2
    else:# x>1:
        return 0.5  
           
@numba.njit()
def integrated_quadratic_shape_factor(x):    
    if x < -1.5:
        return -0.5
    elif x < -0.5:
        return (8*x**3+36*x**2+54*x+3)/48.
    elif x < 0.5:
        return 0.75*x-x**3/3.
    elif x < 1.5:
        return (8*x**3-36*x**2+54*x-3)/48.
    else: # x>1.5
        return 0.5
    
@numba.njit()
def integrated_cubic_shape_factor(x):    
    if x < -2:
        return -0.5
    elif x < -1:
        return (x**4+8*x**3+24*x**2+32*x+4)/24.
    elif x < 0:
        return -x*(3*x**3+8*x**2-16)/24.
    elif x < 1:
        return x*(3*x**3-8*x**2+16)/24.
    elif x < 2:
        return (-x**4+8*x**3-24*x**2+32*x-4)/24.
    else: # x > 2
        return 0.5

@numba.njit(parallel=True)
def deposit_J_linear_numba( xs, x_olds, ws, vys, vzs, l_olds, J,
                   n_threads, indices, q, xidx ):

    shape_factor = integrated_ngp_shape_factor  
    shape_factor_unint = linear_shape_factor
    
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

            J[k,0, i-1] -= w*( shape_factor( xi-x-1 ) - shape_factor( xi-x_old-1 ) )
            J[k,0, i  ] -= w*( shape_factor( xi-x   ) - shape_factor( xi-x_old   ) )
            J[k,0, i+1] -= w*( shape_factor( xi-x+1 ) - shape_factor( xi-x_old+1 ) )
            J[k,0, i+2] -= w*( shape_factor( xi-x+2 ) - shape_factor( xi-x_old+2 ) )

            mid = .5*(x+x_old)
            xi2 = xi + 0.5
            
            
            J[k,1, i-3] += w*vy * shape_factor_unint( xi2-mid-3 )   
            J[k,1, i-2] += w*vy * shape_factor_unint( xi2-mid-2 )
            J[k,1, i-1] += w*vy * shape_factor_unint( xi2-mid-1 ) 
            J[k,1, i  ] += w*vy * shape_factor_unint( xi2-mid   ) 
            J[k,1, i+1] += w*vy * shape_factor_unint( xi2-mid+1 ) 
            J[k,1, i+2] += w*vy * shape_factor_unint( xi2-mid+2 )
            J[k,1, i+3] += w*vy * shape_factor_unint( xi2-mid+3 )
            
            J[k,2, i-3] += w*vz * shape_factor_unint( xi2-mid-3 ) 
            J[k,2, i-2] += w*vz * shape_factor_unint( xi2-mid-2 ) 
            J[k,2, i-1] += w*vz * shape_factor_unint( xi2-mid-1 ) 
            J[k,2, i  ] += w*vz * shape_factor_unint( xi2-mid   ) 
            J[k,2, i+1] += w*vz * shape_factor_unint( xi2-mid+1 ) 
            J[k,2, i+2] += w*vz * shape_factor_unint( xi2-mid+2 )
            J[k,2, i+3] += w*vz * shape_factor_unint( xi2-mid+3 )
    return

@numba.njit(parallel=True)
def deposit_J_quadratic_numba( xs, x_olds, ws, vys, vzs, l_olds, J,
                   n_threads, indices, q, xidx ):

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

            J[k,0, i-1] -= w*( shape_factor( xi-x-1 ) - shape_factor( xi-x_old-1 ) )
            J[k,0, i  ] -= w*( shape_factor( xi-x   ) - shape_factor( xi-x_old   ) )
            J[k,0, i+1] -= w*( shape_factor( xi-x+1 ) - shape_factor( xi-x_old+1 ) )
            J[k,0, i+2] -= w*( shape_factor( xi-x+2 ) - shape_factor( xi-x_old+2 ) )

            mid = .5*(x+x_old)
            xi2 = xi + 0.5
            
            J[k,1, i-4] += w*vy * shape_factor_unint( xi2-mid-4 )
            J[k,1, i-3] += w*vy * shape_factor_unint( xi2-mid-3 )
            J[k,1, i-2] += w*vy * shape_factor_unint( xi2-mid-2 )  
            J[k,1, i-1] += w*vy * shape_factor_unint( xi2-mid-1 ) 
            J[k,1, i  ] += w*vy * shape_factor_unint( xi2-mid   ) 
            J[k,1, i+1] += w*vy * shape_factor_unint( xi2-mid+1 ) 
            J[k,1, i+2] += w*vy * shape_factor_unint( xi2-mid+2 )
            J[k,1, i+3] += w*vy * shape_factor_unint( xi2-mid+3 )
            J[k,1, i+4] += w*vy * shape_factor_unint( xi2-mid+4 )
            
            J[k,2, i-4] += w*vz * shape_factor_unint( xi2-mid-4 )
            J[k,2, i-3] += w*vz * shape_factor_unint( xi2-mid-3 )
            J[k,2, i-2] += w*vz * shape_factor_unint( xi2-mid-2 ) 
            J[k,2, i-1] += w*vz * shape_factor_unint( xi2-mid-1 ) 
            J[k,2, i  ] += w*vz * shape_factor_unint( xi2-mid   ) 
            J[k,2, i+1] += w*vz * shape_factor_unint( xi2-mid+1 ) 
            J[k,2, i+2] += w*vz * shape_factor_unint( xi2-mid+2 )
            J[k,2, i+3] += w*vz * shape_factor_unint( xi2-mid+3 )
            J[k,2, i+4] += w*vz * shape_factor_unint( xi2-mid+4 )

    return

@numba.njit(parallel=True)
def deposit_J_cubic_numba( xs, x_olds, ws, vys, vzs, l_olds, J,
                   n_threads, indices, q, xidx ):

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

            J[k,0, i-2] -= w*( shape_factor( xi-x-2 ) - shape_factor( xi-x_old-2 ) )
            J[k,0, i-1] -= w*( shape_factor( xi-x-1 ) - shape_factor( xi-x_old-1 ) )
            J[k,0, i  ] -= w*( shape_factor( xi-x   ) - shape_factor( xi-x_old   ) )
            J[k,0, i+1] -= w*( shape_factor( xi-x+1 ) - shape_factor( xi-x_old+1 ) )
            J[k,0, i+2] -= w*( shape_factor( xi-x+2 ) - shape_factor( xi-x_old+2 ) )
            J[k,0, i+3] -= w*( shape_factor( xi-x+3 ) - shape_factor( xi-x_old+3 ) )

            mid = .5*(x+x_old)
            xi2 = xi + 0.5
            
            J[k,1, i-5] += w*vy * shape_factor_unint( xi2-mid-5 )
            J[k,1, i-4] += w*vy * shape_factor_unint( xi2-mid-4 )
            J[k,1, i-3] += w*vy * shape_factor_unint( xi2-mid-3 )
            J[k,1, i-2] += w*vy * shape_factor_unint( xi2-mid-2 ) 
            J[k,1, i-1] += w*vy * shape_factor_unint( xi2-mid-1 ) 
            J[k,1, i  ] += w*vy * shape_factor_unint( xi2-mid   ) 
            J[k,1, i+1] += w*vy * shape_factor_unint( xi2-mid+1 ) 
            J[k,1, i+2] += w*vy * shape_factor_unint( xi2-mid+2 )
            J[k,1, i+3] += w*vy * shape_factor_unint( xi2-mid+3 )
            J[k,1, i+4] += w*vy * shape_factor_unint( xi2-mid+4 )
            J[k,1, i+5] += w*vy * shape_factor_unint( xi2-mid+5 )
            
            J[k,2, i-5] += w*vz * shape_factor_unint( xi2-mid-5 )
            J[k,2, i-4] += w*vz * shape_factor_unint( xi2-mid-4 )
            J[k,2, i-3] += w*vz * shape_factor_unint( xi2-mid-3 )
            J[k,2, i-2] += w*vz * shape_factor_unint( xi2-mid-2 ) 
            J[k,2, i-1] += w*vz * shape_factor_unint( xi2-mid-1 ) 
            J[k,2, i  ] += w*vz * shape_factor_unint( xi2-mid   ) 
            J[k,2, i+1] += w*vz * shape_factor_unint( xi2-mid+1 ) 
            J[k,2, i+2] += w*vz * shape_factor_unint( xi2-mid+2 )
            J[k,2, i+3] += w*vz * shape_factor_unint( xi2-mid+3 )
            J[k,2, i+4] += w*vz * shape_factor_unint( xi2-mid+4 )
            J[k,2, i+5] += w*vz * shape_factor_unint( xi2-mid+5 )

    return

@numba.njit(parallel=True)
def deposit_J_quartic_numba( xs, x_olds, ws, vys, vzs, l_olds, J,
                   n_threads, indices, q, xidx ):

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
            
            i = int(l_olds[j]) # l for x_old
            xi = xidx[i] # left grid cell position

            J[k,0, i-3] -= w*( shape_factor( xi-x-3 ) - shape_factor( xi-x_old-3 ) )
            J[k,0, i-2] -= w*( shape_factor( xi-x-2 ) - shape_factor( xi-x_old-2 ) )
            J[k,0, i-1] -= w*( shape_factor( xi-x-1 ) - shape_factor( xi-x_old-1 ) )
            J[k,0, i  ] -= w*( shape_factor( xi-x   ) - shape_factor( xi-x_old   ) )
            J[k,0, i+1] -= w*( shape_factor( xi-x+1 ) - shape_factor( xi-x_old+1 ) )
            J[k,0, i+2] -= w*( shape_factor( xi-x+2 ) - shape_factor( xi-x_old+2 ) )
            J[k,0, i+3] -= w*( shape_factor( xi-x+3 ) - shape_factor( xi-x_old+3 ) )
            J[k,0, i+4] -= w*( shape_factor( xi-x+4 ) - shape_factor( xi-x_old+4 ) )

            mid = .5*(x+x_old)
            xi2 = xi + 0.5
            
            J[k,1, i-6] += w*vy * shape_factor_unint( xi2-mid-6 )
            J[k,1, i-5] += w*vy * shape_factor_unint( xi2-mid-5 )
            J[k,1, i-4] += w*vy * shape_factor_unint( xi2-mid-4 )
            J[k,1, i-3] += w*vy * shape_factor_unint( xi2-mid-3 )
            J[k,1, i-2] += w*vy * shape_factor_unint( xi2-mid-2 ) 
            J[k,1, i-1] += w*vy * shape_factor_unint( xi2-mid-1 ) 
            J[k,1, i  ] += w*vy * shape_factor_unint( xi2-mid   ) 
            J[k,1, i+1] += w*vy * shape_factor_unint( xi2-mid+1 ) 
            J[k,1, i+2] += w*vy * shape_factor_unint( xi2-mid+2 )
            J[k,1, i+3] += w*vy * shape_factor_unint( xi2-mid+3 )
            J[k,1, i+4] += w*vy * shape_factor_unint( xi2-mid+4 )
            J[k,1, i+5] += w*vy * shape_factor_unint( xi2-mid+5 )
            J[k,1, i+6] += w*vy * shape_factor_unint( xi2-mid+6 )
            
            J[k,2, i-6] += w*vz * shape_factor_unint( xi2-mid-6 )            
            J[k,2, i-5] += w*vz * shape_factor_unint( xi2-mid-5 )
            J[k,2, i-4] += w*vz * shape_factor_unint( xi2-mid-4 )
            J[k,2, i-3] += w*vz * shape_factor_unint( xi2-mid-3 )
            J[k,2, i-2] += w*vz * shape_factor_unint( xi2-mid-2 ) 
            J[k,2, i-1] += w*vz * shape_factor_unint( xi2-mid-1 ) 
            J[k,2, i  ] += w*vz * shape_factor_unint( xi2-mid   ) 
            J[k,2, i+1] += w*vz * shape_factor_unint( xi2-mid+1 ) 
            J[k,2, i+2] += w*vz * shape_factor_unint( xi2-mid+2 )
            J[k,2, i+3] += w*vz * shape_factor_unint( xi2-mid+3 )
            J[k,2, i+4] += w*vz * shape_factor_unint( xi2-mid+4 )
            J[k,2, i+5] += w*vz * shape_factor_unint( xi2-mid+5 )
            J[k,2, i+6] += w*vz * shape_factor_unint( xi2-mid+6 )
    return
    
@numba.njit(parallel=True)
def deposit_rho_linear_numba(N, n_threads, x, dx, qw, rho, l, r, indices, xg, Nx ):

    for j in numba.prange(n_threads):
        for i in range( indices[j], indices[j+1] ):   
            
            delta = (x[i] - xg[l[i]])/dx
                              
            rho[j,l[i]] += qw[i] * (1.-delta)
            rho[j,r[i]] += qw[i] * delta  
            
    return

@numba.njit(parallel=True)
def deposit_rho_quadratic_numba(N, n_threads, x, dx, qw, rho, l, r, indices, xg, Nx ):
    
    for j in numba.prange(n_threads):
        for i in range( indices[j], indices[j+1] ):   
            
            delta = (x[i] - xg[l[i]])/dx
            rp1 = (r[i]+1)%Nx
            
            rho[j,l[i]-1] += qw[i] * quadratic_shape_factor(1.+delta)
            rho[j,l[i]]   += qw[i] * quadratic_shape_factor(   delta)
            rho[j,r[i]]   += qw[i] * quadratic_shape_factor(1.-delta)  
            rho[j,rp1]    += qw[i] * quadratic_shape_factor(2.-delta)

    return

@numba.njit(parallel=True)
def deposit_rho_cubic_numba(N, n_threads, x, dx, qw, rho, l, r, indices, xg, Nx ):
    
    for j in numba.prange(n_threads):
        for i in range( indices[j], indices[j+1] ):   
            
            delta = (x[i] - xg[l[i]])/dx
            
            rp1 = (r[i]+1)%Nx
            rp2 = (r[i]+2)%Nx  
            
            rho[j,l[i]-2] += qw[i] * cubic_shape_factor(2.+delta)
            rho[j,l[i]-1] += qw[i] * cubic_shape_factor(1.+delta)
            rho[j,l[i]]   += qw[i] * cubic_shape_factor(   delta)
            rho[j,r[i]]   += qw[i] * cubic_shape_factor(1.-delta)  
            rho[j,rp1]    += qw[i] * cubic_shape_factor(2.-delta)
            rho[j,rp2]    += qw[i] * cubic_shape_factor(3.-delta)  
    return

@numba.njit(parallel=True)
def deposit_rho_quartic_numba(N, n_threads, x, dx, qw, rho, l, r, indices, xg, Nx ):
    
    for j in numba.prange(n_threads):
        for i in range( indices[j], indices[j+1] ):   
            
            delta = (x[i] - xg[l[i]])/dx
            
            rp1 = (r[i]+1)%Nx
            rp2 = (r[i]+2)%Nx 
            rp3 = (r[i]+3)%Nx 
            
            rho[j,l[i]-3] += qw[i] * quartic_shape_factor(3.+delta)
            rho[j,l[i]-2] += qw[i] * quartic_shape_factor(2.+delta)
            rho[j,l[i]-1] += qw[i] * quartic_shape_factor(1.+delta)
            rho[j,l[i]]   += qw[i] * quartic_shape_factor(   delta)
            rho[j,r[i]]   += qw[i] * quartic_shape_factor(1.-delta)  
            rho[j,rp1]    += qw[i] * quartic_shape_factor(2.-delta)
            rho[j,rp2]    += qw[i] * quartic_shape_factor(3.-delta)  
            rho[j,rp3]    += qw[i] * quartic_shape_factor(4.-delta) 
    return

@numba.njit(parallel=True)
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

@numba.njit(parallel=True)
def reseat_numba(N,x,state,l,r,x0,x1,dx,Nx ):
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
    dx    : grid dx
    Nx    : grid Nx
     
    No return neccessary as arrays are modified in-place.
    """
    
    l[:] = np.floor((x-x0)/dx) # left cell
    r[:] = l+1 # right cell   
        
    for i in numba.prange(N):
        if x[i] >= x1 or x[i] < x0:
            #print( 'particle out of range')
            state[i] = False

            l[i] = 0
            r[i] = 0 # particles must have valid indices even if they're dead
 
    return

@numba.njit(parallel=True)
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

@numba.njit(parallel=True)
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
    shape_factor = quadratic_shape_factor
    
    for i in numba.prange(N):

        xi = x[i] - l[i]

        Es[:,i] = Eg[:,l[i]-1]*shape_factor(-xi-1) + \
                  Eg[:,l[i]  ]*shape_factor(-xi  ) + \
                  Eg[:,r[i]  ]*shape_factor(-xi+1) + \
                  Eg[:,r[i]+1]*shape_factor(-xi+2) 
                  
        Bs[:,i] = Bg[:,l[i]-1]*shape_factor(-xi-1) + \
                  Bg[:,l[i]  ]*shape_factor(-xi  ) + \
                  Bg[:,r[i]  ]*shape_factor(-xi+1) + \
                  Bg[:,r[i]+1]*shape_factor(-xi+2) 

    return

@numba.njit(parallel=True)
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
    shape_factor = cubic_shape_factor
    
    for i in numba.prange(N):

        xi = x[i] - l[i]

        Es[:,i] = Eg[:,l[i]-2]*shape_factor(-xi-2) + \
                  Eg[:,l[i]-1]*shape_factor(-xi-1) + \
                  Eg[:,l[i]  ]*shape_factor(-xi  ) + \
                  Eg[:,r[i]  ]*shape_factor(-xi+1) + \
                  Eg[:,r[i]+1]*shape_factor(-xi+2) + \
                  Eg[:,r[i]+2]*shape_factor(-xi+3) 
                  
        Bs[:,i] = Bg[:,l[i]-2]*shape_factor(-xi-2) + \
                  Bg[:,l[i]-1]*shape_factor(-xi-1) + \
                  Bg[:,l[i]  ]*shape_factor(-xi  ) + \
                  Bg[:,r[i]  ]*shape_factor(-xi+1) + \
                  Bg[:,r[i]+1]*shape_factor(-xi+2) + \
                  Bg[:,r[i]+2]*shape_factor(-xi+3) 

    return
 
@numba.njit(parallel=True)
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
    shape_factor = quartic_shape_factor
    
    for i in numba.prange(N):

        xi = x[i] - l[i]

        Es[:,i] = Eg[:,l[i]-3]*shape_factor(-xi-3) + \
                  Eg[:,l[i]-2]*shape_factor(-xi-2) + \
                  Eg[:,l[i]-1]*shape_factor(-xi-1) + \
                  Eg[:,l[i]  ]*shape_factor(-xi  ) + \
                  Eg[:,r[i]  ]*shape_factor(-xi+1) + \
                  Eg[:,r[i]+1]*shape_factor(-xi+2) + \
                  Eg[:,r[i]+2]*shape_factor(-xi+3) + \
                  Eg[:,r[i]+3]*shape_factor(-xi+4)
                  
        Bs[:,i] = Bg[:,l[i]-3]*shape_factor(-xi-3) + \
                  Bg[:,l[i]-2]*shape_factor(-xi-2) + \
                  Bg[:,l[i]-1]*shape_factor(-xi-1) + \
                  Bg[:,l[i]  ]*shape_factor(-xi  ) + \
                  Bg[:,r[i]  ]*shape_factor(-xi+1) + \
                  Bg[:,r[i]+1]*shape_factor(-xi+2) + \
                  Bg[:,r[i]+2]*shape_factor(-xi+3) + \
                  Bg[:,r[i]+3]*shape_factor(-xi+4) 
    return
   