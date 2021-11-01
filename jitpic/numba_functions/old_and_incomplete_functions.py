import numba
import numpy as np

@numba.njit("f8(f8)")
def ngp_shape_factor(x):
    if abs(x) < 0.5:
        return 1
    else:
        return 0
    
@numba.njit("f8(f8)")
def linear_shape_factor(x):
    if x < 1:
        return 1-x
    else:
        return 0
    
@numba.njit("f8(f8)")
def linear_shape_factor_old(x):
    if x < -1:
        return 0
    if x < 0:
        return x+1
    elif x < 1:
        return 1-x
    else:
        return 0
    
@numba.njit("f8(f8)")
def quadratic_shape_factor_old(x):
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
    
@numba.njit("f8(f8)") 
def cubic_shape_factor_old(x):
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
    
@numba.njit("f8(f8)") 
def quartic_shape_factor_old(x):
    if x < -2.5:
        return 0
    elif x < -1.5:
        return (2*x+5)**4 /384.
    elif x < -0.5:
        return -(16*x**4+80*x**3+120*x**2+20*x-55) /96.
    elif x < 0.5:
        return (48*x**4-120*x**2+115)/192.
    elif x < 1.5:
        return (-16*x**4+80*x**3-120*x**2+20*x+55) /96.
    elif x < 2.5:
        return (2*x-5)**4 /384.
    else: 
        return 0

@numba.njit("f8(f8)")
def integrated_ngp_shape_factor(x):    
    if x < -0.5:
        return -0.5
    elif x < 0.5:
        return x
    else:# x>1:
        return 0.5 
    
@numba.njit("f8(f8)")
def integrated_linear_shape_factor_old(x):    
    if x < -1:
        return -0.5
    elif x < 0:
        return .5*x**2+x
    elif x < 1:
        return x-.5*x**2
    else:# x>1:
        return 0.5  

@numba.njit("f8(f8)")
def integrated_quadratic_shape_factor_old(x):    
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

@numba.njit("f8(f8)")
def integrated_cubic_shape_factor_old(x):    
    if x < -2:
        return -0.5
    elif x < -1:
        return (x**4+8*x**3+24*x**2+32*x+4) /24.
    elif x < 0:
        return -x*(3*x**3+8*x**2-16) /24.
    elif x < 1:
        return x*(3*x**3-8*x**2+16) /24.
    elif x < 2:
        return (-x**4+8*x**3-24*x**2+32*x-4) /24.
    else: # x > 2
        return 0.5
    
@numba.njit(parallel=True)
def deposit_current_old_version( xs, x_olds, ws, vys, vzs, l_olds, J,
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

            

            
            if x == x_old: # no displacement? no current; continue
                continue
            
            i = int(l_olds[j]) # l for x_old
            xi = xidx[i] # left grid cell position

            #print( 'x1',x-xi, 'x0',x_old-xi, 'xg',(x-xi)-(x_old-xi))     
            
            w = ws[j]*q
            vy = vys[j]
            vz = vzs[j]

            # depending on boundary conditions, these can change
            im2 = i-2
            im1 = i-1
            ip0 = i
            ip1 = i+1
            ip2 = i+2
            
            Jl = 0. 
            Jr = 0.
            Jc = 0.
            
            
            if x_old >= xi and x_old < x2[i]: # left half of the cell
                    
                if (x >= x2[i]-1.) and (x < x2[i]): #small move left
    
                    # x
                    J[k,0,ip0] += w*(x - x_old)
                    
                    #print( '(0)', w*(x-x_old) )
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
                    
                elif (x >= x2[i]): # small/large move right
                        
                    ee =  (x_old - xi - 0.5) / (x_old - x) ###
    
                    # x
                    Jl = w*(0.5 - (x_old - xi))
                    Jr = w*(x - x_old) - Jl                 
                        
                    J[k,0,ip0] += Jl
                    J[k,0,ip1] += Jr
                    
                    #print( '(0)', Jl, '(1)', Jr )
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

                        
                else:# (x < x2[i-1]): # large move left
                        
                    ee = (x_old - xi + 0.5) / (x_old - x) ###
                    
                    # x
                    Jr = w*(-0.5 - (x_old - xi))
                    Jl = w*(x - x_old) - Jr                     
                    
                    J[k,0,ip0] += Jr
                    J[k,0,im1] += Jl
                    #print( '(-1)', Jl, '(0)', Jr )
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
                    
                # else:
                #     print('particle moving but no current deposited?')
                        
            else: #if x_old >= x2[i] and x_old < xi+1.: #right half of the cell
    
                if (x >= x2[i]) and (x < x2[i] + 1.): #small move right
                    
                    # x
                    J[k,0,ip1] += w*(x - x_old)
                    #print( '(1)', w*(x-x_old) )
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
                    
                elif (x < x2[i]): # small/large move left
                    ee =  (x_old - xi - 0.5) / (x_old - x) 
    
                    # x
                    Jr = w*(0.5 - (x_old - xi))
                    Jl = w*(x - x_old) - Jr           
                    
                    J[k,0,ip1] += Jr
                    J[k,0,ip0] += Jl
                    
                    #print( '(0)', Jl, '(1)', Jr)
                    
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
                    
                else:# (x >= x2[i] + 1.): # large move right
                    ee =  (x_old - xi - 1.5) / (x_old - x) 
                    
                    # x
                    Jl = w*(1.5 - (x_old - xi))
                    Jr = w*(x - x_old) - Jl                
                       
                    J[k,0,ip1] += Jl
                    J[k,0,ip2] += Jr
                    
                    #print( '(1)', Jl, '(2)', Jr)
                    
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

                # else: # for debug, can probably remove
                #     print('particle moving but no current deposited?')   
                
            # else:
            # print('particle not seated?')
            
    return

@numba.njit(parallel=True)
def deposit_J_numba( xs, x_olds, ws, vys, vzs, l_olds, J,
                   n_threads, indices, q, xidx,
                   shape, l_factor, t_factor):
    """
    Current deposition for arbitrary particle shapes
    """
    for k in numba.prange(n_threads):
        for j in range( indices[k], indices[k+1] ):   
            
            x = xs[j]
            x_old = x_olds[j]
            
            if x == x_old: # no displacement? no current; continue
                continue
            
            i = int(l_olds[j]) # l for x_old
            xi = xidx[i] # left grid cell position

            dx1 = xi - x
            dx0 = xi - x_old
            
            # longitudinal start/end indices
            l0 = -shape//2
            l1 = 3 + (shape-1)//2
            
            for l in range(l0, l1, 1):
                J[k,0, i+l ] += ws[j]*q*( l_factor( dx0+l ) - l_factor( dx1+l ) )

            dx = xi + 0.5 - .5*(x+x_old) 
            
            # transverse start/end indices
            t0 = -(shape-1)//2 - 1
            t1 = 2+2*(shape//2) + (1-shape)//2
            for t in range(t0, t1, 1):
                J[k,1, i+t ] += vys[j]*ws[j]*q * t_factor( dx+t ) 
                J[k,2, i+t ] += vzs[j]*ws[j]*q * t_factor( dx+t ) 

    return

@numba.njit(parallel=True)
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
        
        #a = p0 + q*E + q/2*p/gamma x B
        a[0] = p[0,i] + qmdt*E[0,i] +  0.5*qmdt*rg[i]*(p[1,i]*B[2] - p[2,i]*B[1])
        a[1] = p[1,i] + qmdt*E[1,i] +  0.5*qmdt*rg[i]*(p[2,i]*B[0] - p[0,i]*B[2])
        a[2] = p[2,i] + qmdt*E[2,i] +  0.5*qmdt*rg[i]*(p[0,i]*B[1] - p[1,i]*B[0])
        
        b = 0.5*qmdt*B[i,:]
        
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
