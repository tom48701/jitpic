import numba
import numpy as np

@numba.njit(parallel=True)
def deposit_numba( xs, x_olds, ws, vys, vzs, l_olds, J,
                   n_threads, indices, q, xidx, x2 ):
    """
    current deposition algorithm, set up for numba parallelisation,
    algorithm sourced from JPIC and parralelisation technique 
    adapted from FBPIC.
    
    xs        : pre-masked particle positions
    x_olds    : pre-masked old particle positions
    ws        : pre-masked particle weights
    vys       : pre-masked particle y velocities
    vzs       : pre-masked particle z velocities
    l_olds    : pre masked old left indices
    J         : 3D current density array
    n_threads : number of numba threads
    indices   : list of start/end indices for chunking
    q         : particle charge
    xidx      : normalised grid x positions
    x2        : normalised grid mid-cell x positions
    
    No returns neccessary as arrays are modified in-place.
    """

    for k in numba.prange(n_threads):
        
        for j in range( indices[k], indices[k+1] ):   
            
            x = xs[j]
            x_old = x_olds[j]
            
            if abs(x - x_old) == 0.: # no displacement? no current; continue
                continue
            
            w = ws[j]*q
            vy = vys[j]
            vz = vzs[j]
            i = int(l_olds[j]) # l for x_old

            # depending on boundary conditions, these can change
            im2 = i-2
            im1 = i-1
            ip0 = i
            ip1 = i+1
            ip2 = i+2
            
            xi = xidx[i] 
            
            Jl = 0. 
            Jr = 0.
            Jc = 0.
            
            if x_old >= xi and x_old < x2[i]: # left half of the cell
                
                if (x >= x2[i-1]) and (x < x2[i]): #small move left
                    
                    # x
                    J[k,0,ip0] += w*(x - x_old)
                    
                    # y
                    Jr = 0.5*w*vy*(1.+ x + x_old - 2.*xi)
                    Jl = w*vy - Jr
                    
                    J[k,1,ip0] += Jr
                    J[k,1,im1] += Jl
    
                    # z
                    Jr = 0.5*w*vz*(1.+ x + x_old - 2.*xi)
                    Jl = w*vz - Jr
                    
                    J[k,2,ip0] += Jr
                    J[k,2,im1] += Jl
                    
                        
                    
                elif (x < x2[i-1]): # large move left
                    ee = (x_old - xi + 0.5) / (x_old - x) ###
                    
                    # x
                    Jr = w*(-0.5 - (x_old - xi))
                    Jl = w*(x - x_old) - Jr                     
                    
                    J[k,0,ip0] += Jr
                    J[k,0,im1] += Jl
                    
                    # y
                    Jr = 0.5*ee*w*vy*(0.5 + x_old - xi)
                    Jl = 0.5*(1. - ee)*w*vy*(-0.5 - (x - xi))
                    Jc = w*vy + Jr - Jl
                    
                    J[k,1,ip0] += Jr
                    J[k,1,im1] += Jc
                    J[k,1,im2] += Jl
                    
                    # z
                    Jr = 0.5*ee*w*vz*(0.5 + x_old - xi)
                    Jl = 0.5*(1. - ee)*w*vz*(-0.5 - (x - xi))
                    Jc = w*vz + Jr - Jl
                    
                    J[k,2,ip0] += Jr
                    J[k,2,im1] += Jc
                    J[k,2,im2] += Jl
                    
                    
                elif (x >= x2[i]): # small/large move right
                    ee =  (x_old - xi - 0.5) / (x_old - x) ###
    
                    # x
                    Jl = w*(0.5 - (x_old - xi))
                    Jr = w*(x - x_old) - Jl                 
                    
                    J[k,0,ip0] += Jl
                    J[k,0,ip1] += Jr
                    
                    # y
                    Jl = 0.5*ee*w*vy*(0.5-(x_old-xi))
                    Jr = 0.5*(1.-ee)*w*vy*(x-xi-0.5)
                    Jc = w*vy - Jl - Jr
                    
                    J[k,1,im1] += Jl
                    J[k,1,ip0] += Jc
                    J[k,1,ip1] += Jr
                    
                    # z
                    Jl = 0.5*ee*w*vz*(0.5-(x_old-xi))
                    Jr = 0.5*(1.-ee)*w*vz*(x-xi-0.5)
                    Jc = w*vz - Jl - Jr
                    
                    J[k,2,im1] += Jl
                    J[k,2,ip0] += Jc
                    J[k,2,ip1] += Jr                            
                    
                    
                else:
                    print('particle moving but no current deposited?')
                        
            elif x_old >= x2[i] and x_old < xi+1.: #right half of the cell
    
                if (x >= x2[i]) and (x < x2[i] + 1.): #small move right
                    
                    # x
                    J[k,0,ip1] += w*(x - x_old)
       
                    # y
                    Jl = 0.5*w*vy*(3. + 2*xi - x_old - x)
                    Jr = w*vy - Jl
                    
                    J[k,1,ip0] += Jl
                    J[k,1,ip1] += Jr
                    
                    # z
                    Jl = 0.5*w*vz*(3. + 2*xi - x_old - x)
                    Jr = w*vz - Jl
                    
                    J[k,2,ip0] += Jl
                    J[k,2,ip1] += Jr
                    
                    
                elif (x >= x2[i] + 1.): # large move right
                    ee =  (x_old - xi - 1.5) / (x_old - x) 
                    
                    # x
                    Jl = w*(1.5 - (x_old - xi))
                    Jr = w*(x - x_old) - Jl                
                       
                    J[k,0,ip1] += Jl
                    J[k,0,ip2] += Jr
                    
                    # y
                    Jl = 0.5*ee*w*vy*(1.5 - (x_old - xi))
                    Jr = 0.5*(1.-ee)*w*vy*(x - xi - 1.5)
                    Jc = w*vy - Jl - Jr
                    
                    J[k,1,ip0] += Jl
                    J[k,1,ip1] += Jc
                    J[k,1,ip2] += Jr
                    
                    Jl = 0.5*ee*w*vz*(1.5 - (x_old - xi))
                    Jr = 0.5*(1.-ee)*w*vz*(x - xi - 1.5)
                    Jc = w*vz - Jl - Jr
                    
                    J[k,2,ip0] += Jl
                    J[k,2,ip1] += Jc
                    J[k,2,ip2] += Jr
                    
                        
                elif (x < x2[i]): # small/large move left
                    ee =  (x_old - xi - 0.5) / (x_old - x) 
    
                    # x
                    Jr = w*(0.5 - (x_old - xi))
                    Jl = w*(x - x_old) - Jr           
                    
                    J[k,0,ip1] += Jr
                    J[k,0,ip0] += Jl
                    
                    #y
                    Jr = 0.5*ee*w*vy*(x_old - xi - 0.5)
                    Jl = 0.5*(1. - ee)*w*vy*(0.5 - (x - xi))
                    Jc = w*vy - Jr - Jl
                    
                    J[k,1,ip1] += Jr
                    J[k,1,im1] += Jl
                    J[k,1,ip0] += Jc
                    
                    # z
                    Jr = 0.5*ee*w*vz*(x_old - xi - 0.5)
                    Jl = 0.5*(1. - ee)*w*vz*(0.5 - (x - xi))
                    Jc = w*vz - Jr - Jl
                    
                    J[k,2,ip1] += Jr
                    J[k,2,im1] += Jl
                    J[k,2,ip0] += Jc
                    
                
                else: # for debug, can probably remove
                    print('particle moving but no current deposited?')   
    
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
def interpolate_numba(Eg,Bg, Es,Bs, l,r,x, x0, dx, N):
    """
    Interpolate fields from the grid onto particles
    
    Eg : grid E field
    Eb : grid B field
    Es : particle E field
    Bs : particle B field
    l  : left cell indices
    r  : right cell indices
    x  : particle positions
    x0 : grid x0
    dx : grid dx
    N  : number of particles

    No return neccessary as arrays are modified in-place.
    """
    for i in numba.prange(N):
        
        xi = (x[i]-x0)/dx  
        alpha = l[i] + 1. - xi
        beta = 1. - alpha

        Es[:,i] = alpha*Eg[:,l[i]] + beta*Eg[:,r[i]]
        Bs[:,i] = alpha*Bg[:,l[i]] + beta*Bg[:,r[i]]
        
    return
    