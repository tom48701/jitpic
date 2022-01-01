import math
from numba import cuda

@cuda.jit("f8(f8)", device=True) 
def quartic_shape_factor(x):
    if x < 0.5:
        return (48*x**4-120*x**2+115) * 0.005208333333333333
    elif x < 1.5:
        return (-16*x**4+80*x**3-120*x**2+20*x+55)* 0.010416666666666666
    elif x < 2.5:
        return (2*x-5)**4 * 0.0026041666666666665
    else: 
        return 0

@cuda.jit("f8(f8)", device=True)
def integrated_quartic_shape_factor(x):    
    sgn = int(x/abs(x))
    x = abs(x)
    if x < 0.5:
        return sgn * (48*x**5-200*x**3+575*x) * 0.0010416666666666667
    elif x < 1.5:
        return sgn * (-16*x**5+100*x**4-200*x**3+50*x**2+275*x+1.25) * 0.0020833333333333333
    elif x < 2.5:
        return sgn * ((2*x-5)**5 * 0.00026041666666666666 + 0.5)
    else: # x > 2.5
        return sgn * 0.5

@cuda.jit("(f8[::1], f8[::1], f8[::1], f8[:,::1], f8[:,::1], f8, f8[::1], f8, f8, b1[::1])")
def J4o( xs, x_olds, ws, vs, J, q, xidx, idx, x0, state ):
    
    j = cuda.grid(1)
    
    if j < xs.size:
        
        if state[j]:
            x = xs[j] * idx
            x_old = x_olds[j] * idx
            w = ws[j]*q
    
            i = int(math.floor((x_old-x0*idx))) #l_olds[j] # l for x_old
            xi = xidx[i] * idx # left grid cell position
       
            if x != x_old: # Jx
                dx1 = xi - x
                dx0 = xi - x_old
    
                cuda.atomic.add(J, (0, i-3), w*( integrated_quartic_shape_factor( dx0-3 ) - integrated_quartic_shape_factor( dx1-3 ) ))
                cuda.atomic.add(J, (0, i-2), w*( integrated_quartic_shape_factor( dx0-2 ) - integrated_quartic_shape_factor( dx1-2 ) ))
                cuda.atomic.add(J, (0, i-1), w*( integrated_quartic_shape_factor( dx0-1 ) - integrated_quartic_shape_factor( dx1-1 ) ))
                cuda.atomic.add(J, (0, i  ), w*( integrated_quartic_shape_factor( dx0   ) - integrated_quartic_shape_factor( dx1   ) ))
                cuda.atomic.add(J, (0, i+1), w*( integrated_quartic_shape_factor( dx0+1 ) - integrated_quartic_shape_factor( dx1+1 ) ))
                cuda.atomic.add(J, (0, i+2), w*( integrated_quartic_shape_factor( dx0+2 ) - integrated_quartic_shape_factor( dx1+2 ) ))
                cuda.atomic.add(J, (0, i+3), w*( integrated_quartic_shape_factor( dx0+3 ) - integrated_quartic_shape_factor( dx1+3 ) ))
                cuda.atomic.add(J, (0, i+4), w*( integrated_quartic_shape_factor( dx0+4 ) - integrated_quartic_shape_factor( dx1+4 ) ))
                
            vy = vs[1,j]
            vz = vs[2,j]
            dx = xi + 0.5 - .5*(x+x_old) 
            
            # shape factors expect a positive value, dx can be slightly negative (> -1)
            # when calculating for a cell centre, so abs the central values
            if vy != 0.: # Jy
                cuda.atomic.add(J, (1, i-3), w*vy * quartic_shape_factor( 3-dx ) )
                cuda.atomic.add(J, (1, i-2), w*vy * quartic_shape_factor( 2-dx ) ) 
                cuda.atomic.add(J, (1, i-1), w*vy * quartic_shape_factor( 1-dx ) )
                cuda.atomic.add(J, (1, i  ), w*vy * quartic_shape_factor(abs(dx))) 
                cuda.atomic.add(J, (1, i+1), w*vy * quartic_shape_factor( 1+dx ) ) 
                cuda.atomic.add(J, (1, i+2), w*vy * quartic_shape_factor( 2+dx ) )
                cuda.atomic.add(J, (1, i+3), w*vy * quartic_shape_factor( 3+dx ) )
            
            if vz != 0.: # Jz
                cuda.atomic.add(J, (2, i-3), w*vz * quartic_shape_factor( 3-dx ) )
                cuda.atomic.add(J, (2, i-2), w*vz * quartic_shape_factor( 2-dx ) ) 
                cuda.atomic.add(J, (2, i-1), w*vz * quartic_shape_factor( 1-dx ) ) 
                cuda.atomic.add(J, (2, i  ), w*vz * quartic_shape_factor(abs(dx))) 
                cuda.atomic.add(J, (2, i+1), w*vz * quartic_shape_factor( 1+dx ) ) 
                cuda.atomic.add(J, (2, i+2), w*vz * quartic_shape_factor( 2+dx ) )
                cuda.atomic.add(J, (2, i+3), w*vz * quartic_shape_factor( 3+dx ) )

    return

@cuda.jit("(f8[:,::1], f8[:,::1], f8[:,::1], f8[:,::1], u4[::1], u4[::1], f8[::1], i4, f8, f8)")
def I4o(Eg,Bg, Es,Bs, l,r, x, N, x0, idx):
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
    i = cuda.grid(1)

    if i < N:

        xi = (x[i]-x0)*idx - l[i]
        
        Lm2 = quartic_shape_factor(   xi+2 )
        Lm1 = quartic_shape_factor(   xi+1 )
        L   = quartic_shape_factor(   xi   ) 
        R   = quartic_shape_factor( 1-xi   )
        Rp1 = quartic_shape_factor( 2-xi   ) 
        Rp2 = quartic_shape_factor( 3-xi   )
        
        Es[0,i] = Eg[0,l[i]-2]*Lm2 + \
                  Eg[0,l[i]-1]*Lm1 + \
                  Eg[0,l[i]  ]*L + \
                  Eg[0,r[i]  ]*R + \
                  Eg[0,r[i]+1]*Rp1 + \
                  Eg[0,r[i]+2]*Rp2 

        Es[1,i] = Eg[1,l[i]-2]*Lm2 + \
                  Eg[1,l[i]-1]*Lm1 + \
                  Eg[1,l[i]  ]*L + \
                  Eg[1,r[i]  ]*R + \
                  Eg[1,r[i]+1]*Rp1 + \
                  Eg[1,r[i]+2]*Rp2 
                  
        Es[2,i] = Eg[2,l[i]-2]*Lm2 + \
                  Eg[2,l[i]-1]*Lm1 + \
                  Eg[2,l[i]  ]*L + \
                  Eg[2,r[i]  ]*R + \
                  Eg[2,r[i]+1]*Rp1 + \
                  Eg[2,r[i]+2]*Rp2 
                  
        Bs[0,i] = Bg[0,l[i]-2]*Lm2 + \
                  Bg[0,l[i]-1]*Lm1 + \
                  Bg[0,l[i]  ]*L + \
                  Bg[0,r[i]  ]*R + \
                  Bg[0,r[i]+1]*Rp1 + \
                  Bg[0,r[i]+2]*Rp2 

        Bs[1,i] = Bg[1,l[i]-2]*Lm2 + \
                  Bg[1,l[i]-1]*Lm1 + \
                  Bg[1,l[i]  ]*L + \
                  Bg[1,r[i]  ]*R + \
                  Bg[1,r[i]+1]*Rp1 + \
                  Bg[1,r[i]+2]*Rp2 
                  
        Bs[2,i] = Bg[2,l[i]-2]*Lm2 + \
                  Bg[2,l[i]-1]*Lm1 + \
                  Bg[2,l[i]  ]*L + \
                  Bg[2,r[i]  ]*R + \
                  Bg[2,r[i]+1]*Rp1 + \
                  Bg[2,r[i]+2]*Rp2 
    return

@cuda.jit("(f8[:,::1], f8[:,::1], f8, f8[:,::1], f8[:,::1], f8[::1], f8[::1], f8[::1], f8, f8, i8)")
def cohen_push( E, B, qmdt, p, v, x, x_old, rg, m, dt, N):
    """
    Cohen particle push
    http://dx.doi.org/10.1016/j.nima.2009.03.083
    
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
    i = cuda.grid(1)
    if i < N:

        #a = np.empty(3) # must be assigned within the loop to be private
        
        B0 = 0.5*qmdt*B[0,i]
        B1 = 0.5*qmdt*B[1,i]
        B2 = 0.5*qmdt*B[2,i]
        
        #a = p0 + q*E + q/2*p/gamma x B
        A0 = p[0,i] + qmdt*E[0,i] +  rg[i]*(p[1,i]*B2 - p[2,i]*B1)
        A1 = p[1,i] + qmdt*E[1,i] +  rg[i]*(p[2,i]*B0 - p[0,i]*B2)
        A2 = p[2,i] + qmdt*E[2,i] +  rg[i]*(p[0,i]*B1 - p[1,i]*B0)
        
        a2 = A0**2 + A1**2 + A2**2
        b2 = B0**2 + B1**2 + B2**2
        
        a2b2 = 0.5*(1. + a2 - b2)
        adotb = A0*B0 + A1*B1 + A2*B2 
        
        gamma2 = a2b2 + math.sqrt( a2b2**2 + b2 + adotb**2 ) 
        gamma = math.sqrt(gamma2)
        
        p[0,i] = ( gamma2*A0 + gamma*(A1*B2 - A2*B1) + B0*adotb ) / (gamma2 + b2)
        p[1,i] = ( gamma2*A1 + gamma*(A2*B0 - A0*B2) + B1*adotb ) / (gamma2 + b2)
        p[2,i] = ( gamma2*A2 + gamma*(A0*B1 - A1*B0) + B2*adotb ) / (gamma2 + b2)
        
        rg[i] = 1./gamma
        
        # make a note of the old positions
        x_old[i] = x[i]
        
        # update v
        v[0,i] = p[0,i] * rg[i] / m
        v[1,i] = p[1,i] * rg[i] / m
        v[2,i] = p[2,i] * rg[i] / m
        
        # update x
        x[i] = x[i] + v[0,i] * dt
         
    return

@cuda.jit("(i8, f8[::1], f8, f8, f8[::1], f8[::1], u4[::1], u4[::1], f8[::1], i4, b1[::1])")
def R4o(N, x, idx, q, w, rho, l, r, xg, Nx, state ):
    """
    Charge density deposition for quartic particle shapes and open boundaries

    N         : number of particles
    x         : particle positions
    idx       : inverse grid dx
    q         : particle q
    w         : particle w
    rho       : rho array
    l         : particle left cell indices
    r         : particle right cell indices
    indices   : particle index start/stop for each thread
    xg        : grid points
    Nx        : grid Nx
    state     : particle states
    
    No returns neccessary as arrays are modified in-place.
    """ 
    i = cuda.grid(1)
    
    if i < x.size:
        if state[i]:
            
            xi = (x[i] - xg[l[i]])*idx
            qw = q*w[i]
            
            if l[i] > 1:
                cuda.atomic.add( rho, l[i]-2, qw*quartic_shape_factor(2+xi) )
    
            if l[i] > 0:
                cuda.atomic.add( rho, l[i]-1, qw*quartic_shape_factor(1+xi) )
                
            cuda.atomic.add( rho, l[i], qw*quartic_shape_factor(xi  ) )
            cuda.atomic.add( rho, r[i], qw*quartic_shape_factor(1-xi) )  
            
            if r[i] < Nx-2:
                cuda.atomic.add( rho, r[i]+1, qw*quartic_shape_factor(2-xi) )
            if r[i] < Nx-3:
                cuda.atomic.add( rho, r[i]+2, qw*quartic_shape_factor(3-xi) )
                
    return

@cuda.jit("(i8, f8[::1], b1[::1], u4[::1], u4[::1], f8, f8, f8, f8, i8, i4[::1])")
def reseat_open(N,x,state,l,r,x0,x1,dx,idx,Nx,N_alive ):
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
    N_alive
    
    No return neccessary as arrays are modified in-place.
    """
    i = cuda.grid(1)
    if i < x.size:
        l[i] = math.floor((x[i]-x0)*idx) # left cell
        r[i] = l[i]+1 # right cell   
        
        cuda.atomic.add(N_alive, 0, 1)
        
        if x[i] >= x1 or x[i] < x0:

            state[i] = False

            l[i] = 0
            r[i] = 0 # particles must have valid indices even if they're dead
 
    return

@cuda.jit("(f8, f8, f8[:,::1], f8[:,::1], f8[:,::1], f8[::1], f8[::1], f8[::1], f8[::1], f8[::1], f8[::1], f8[::1], f8[::1], i4[::1])")
def NDFX_solver_1(pidt, Nx, E, B, J, PR, PL, SR, SL, PRs, PLs, SRs, SLs, f_shift):
    """
    CUDA implementation of the field solver
    
    first compute intermediate quantities, block-level synchronisation does
    not resolve all race conditions, we need grid-level sync.
    
    """
    
    i = cuda.grid(1)
    
    if i < Nx:
        PR[i] = (E[1,i] + B[2,i]) * .5
        PL[i] = (E[1,i] - B[2,i]) * .5
        SR[i] = (E[2,i] - B[1,i]) * .5
        SL[i] = (E[2,i] + B[1,i]) * .5
        
        #PRs[i] = PR[shift]
        #PLs[i] = PL[shift]
        #SRs[i] = SR[shift]
        #SLs[i] = SL[shift]
        
        PRs[i] = 0.
        PLs[i] = 0.
        SRs[i] = 0.
        SLs[i] = 0.
        return

@cuda.jit("(f8, f8, f8[:,::1], f8[:,::1], f8[:,::1], f8[::1], f8[::1], f8[::1], f8[::1], f8[::1], f8[::1], f8[::1], f8[::1], i4[::1])")
def NDFX_solver_2(pidt, Nx, E, B, J, PR, PL, SR, SL, PRs, PLs, SRs, SLs, f_shift):
    """
    CUDA implementation of the field solver
    
    pidt : timestep constant
    
    """
    
    i = cuda.grid(1)
    
    if i < Nx:
        
        shift = f_shift[i]
        
        E[0,i] -= 2. * pidt * J[0,i]
             
        # PR[i] = (E[1,i] + B[2,i]) * .5
        # PL[i] = (E[1,i] - B[2,i]) * .5
        # SR[i] = (E[2,i] - B[1,i]) * .5
        # SL[i] = (E[2,i] + B[1,i]) * .5

        # PRs[i] = 0.
        # PLs[i] = 0.
        # SRs[i] = 0.
        # SLs[i] = 0.
        
        #cuda.syncthreads()
        
        #PR[shift] = PR[i] - pidt * J[1,i]
        #PL[i] = PL[shift] - pidt * J[1,i]
        #SR[shift] = SR[i] - pidt * J[2,i]
        #SL[i] = SL[shift] - pidt * J[2,i]
        
        cuda.atomic.add( PRs, shift, PR[i]-pidt*J[1,i] )
        cuda.atomic.add( PLs, i, PL[shift]-pidt*J[1,i] ) 
        cuda.atomic.add( SRs, shift, SR[i]-pidt*J[2,i] )
        cuda.atomic.add( SLs, i, SL[shift]-pidt*J[2,i] ) 
        
        #cuda.syncthreads()
        
        # E[1,i] = PRs[i] + PLs[i]
        # B[2,i] = PRs[i] - PLs[i]
        # E[2,i] = SLs[i] + SRs[i]
        # B[1,i] = SLs[i] - SRs[i]


    return
    
@cuda.jit("(f8, f8, f8[:,::1], f8[:,::1], f8[:,::1], f8[::1], f8[::1], f8[::1], f8[::1], f8[::1], f8[::1], f8[::1], f8[::1], i4[::1])")
def NDFX_solver_3(pidt, Nx, E, B, J, PR, PL, SR, SL, PRs, PLs, SRs, SLs, f_shift):
    """
    CUDA implementation of the field solver
    
    finally reduce back to E/B from P/L fields
    
    """ 
    i = cuda.grid(1)
    
    if i < Nx:
        E[1,i] = PRs[i] + PLs[i]
        B[2,i] = PRs[i] - PLs[i]
        E[2,i] = SLs[i] + SRs[i]
        B[1,i] = SLs[i] - SRs[i]
        return 