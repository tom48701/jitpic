import numpy as np

def default_profile(x):
    return np.ones_like(x)
        
class Species:
    """ Particle Species """
    
    def __init__(self, name, ppc, n, p0, p1, m=1., q=-1., eV=0., dens=None,
                 p_x=0., p_y=0., p_z=0.):
        """
        name  : str    : species name for diagnostic purposes
        ppc   : int    : number of particles per cell
        p0    : float  : species start position
        p1    : float  : species end position
        m     : float  : particle mass in electron masses 
        q     : float  : particle charge in elementary charges 
        eV    : flaot  : species temperature in electron volts (
        dens  : func   : function describing the density profile. The function
                         takes a single argument; the grid x-positions with the
                         sane shape as grid.x, and similarly should return an 
                         array with the same shape as grid.x densities 
                         specified by functions are normalised to the
                         reference density species.n
        p_(i) : float  : flow momenta
        """

        self.name = name 
        self.dfunc = dens 
        self.m = m 
        self.q = q 
        self.qm = q/m # charge/mass ratio
        self.n = n 
        self.ppc = ppc 
        self.p0 = p0 
        self.p1 = p1 
        self.ddx = None # inter-particle half-spacing (set later)
        
        self.eV = eV
        self.Ek = self.eV / 5.11e5 # / electron rest mass energy
        self.p_th = np.sqrt(self.Ek**2 + 2.*self.Ek*self.m) / np.sqrt(3) # thermal momentum
        self.p_th_reduced = self.p_th/np.sqrt(2.) # reduced thermal momentum
        self.p_flow = np.array([p_x, p_y, p_z]) # flow momentum
        
        # use the default (flat) density profile if none is specified
        if dens is None:
            self.dfunc = default_profile
        else:
            self.dfunc = dens
    
        return

    def initialise_particles(self, grid):
        """
        Generate and seat the initial set of particles onto the grid.
        
        grid : simgrid : grid object onto which to seat particles
        """
        dx = grid.dx
        x = grid.x
        
        self.ddx = dx/(2*self.ppc) # inter-particle half-spacing

        if grid.boundaries == 'open':
            min_cell = max( 0,         int(self.p0/dx - grid.x0/dx)     )
            max_cell = min( grid.Nx-1, int(self.p1/dx - grid.x0/dx + 1) )
            
            N = self.ppc*(max_cell-min_cell)
            x = np.linspace( x[min_cell]+self.ddx, x[max_cell]-self.ddx, N )
            
        elif grid.boundaries == 'periodic':
            N = self.ppc*grid.Nx
            x = np.linspace( x[0] + self.ddx, x[-1]+dx - self.ddx, N)
            
            
        w = np.full(N, self.n/self.ppc) * self.dfunc(x) 

        self.w = w[w != 0]
        self.x = x[w != 0]
        self.x_old = np.zeros_like(self.x)
        
        self.N = len(self.x)
        self.N0 = self.N

        self.E = np.zeros((3,self.N))
        self.B = np.zeros((3,self.N))     
        
        self.state = np.full(self.N, True, dtype=bool)
        self.N_alive = self.N # all particles should be alive at first
 
        p = np.zeros((3,self.N))       

        # set initial flow and thermal momentum
        p[0,:] = self.p_flow[0] * np.ones(self.N) + np.random.normal(0., self.p_th_reduced, self.N) 
        p[1,:] = self.p_flow[1] * np.ones(self.N) + np.random.normal(0., self.p_th_reduced, self.N) 
        p[2,:] = self.p_flow[2] * np.ones(self.N) + np.random.normal(0., self.p_th_reduced, self.N) 
        self.p = p
        
        self.rg = 1./np.sqrt(1. + (self.p**2).sum(axis=0)/self.m**2)  # reciprocal gamma factor      
        
        self.v = np.zeros_like(self.p)
        self.v[:,:] = self.p[:,:] * self.rg / self.m
        
        # setup array to contain the left and right cell indices
        self.l = np.zeros(self.N, dtype=np.dtype('u4'))
        self.r = np.zeros(self.N, dtype=np.dtype('u4'))
        self.l[:] = np.floor((self.x-grid.x0)/grid.dx) # left cells
        
        self.r[:] = np.mod(self.l+1, grid.Nx) # mod should only affect periodic mode

        return
    
    def revert_x(self):
        """
        Overwrite current particle positions with their previous positions.
        Used only for the initial p,v,J offset.
        """
        self.x[:] = self.x_old[:]
        return
        
    def inject_particles(self, grid):
        """
        Generate and append one cell's-worth of particles to the grid
        """
        x1 = grid.x1
        ppc = self.ppc
        
        if x1 >= self.p0 and x1 <= self.p1:

            # expand particle positions & set values
            self.x = np.pad( self.x, (0,ppc), 'empty' )
            self.x_old = np.pad( self.x_old, (0,ppc), 'empty' )
            x = np.linspace( x1-grid.dx+self.ddx, x1-self.ddx, ppc)
            self.x[-ppc:] = x
            self.x_old[-ppc:] = x
            
            # expand weights, & set values
            self.w = np.pad( self.w, (0,ppc), 'empty' )
            self.w[-ppc:] = self.dfunc(x)*self.n/ppc
            # (skip the zero-weight check for speed)
            
            # expand index arrays & add values
            self.l = np.pad( self.l, (0,ppc), constant_values=(0,grid.Nx-2) )
            self.r = np.pad( self.r, (0,ppc), constant_values=(0,grid.Nx-1) )

            # calculate thermal motion
            if self.eV != 0.:
 
                self.p  = np.pad( self.p, ((0,0),(0,ppc)), 'empty')
                self.v  = np.pad( self.v, ((0,0),(0,ppc)), 'empty' )
                self.rg = np.pad( self.rg, (0,ppc), 'empty' )
                
                p = np.random.normal(0., self.p_th_reduced, ((3,ppc)) )
                rg = 1./np.sqrt(1. + (p**2).sum(axis=0)/self.m**2) 
                v = p*rg / self.m 
                
                self.p[:,-ppc:]  = p
                self.v[:,-ppc:]  = v
                self.rg[-ppc:]   = rg
                
            else:
                self.p  = np.pad( self.p, ((0,0),(0,ppc)), constant_values=0. )
                self.v  = np.pad( self.v, ((0,0),(0,ppc)), constant_values=0. )
                self.rg = np.pad( self.rg, (0,ppc), constant_values=1. )

            # expand field arrays
            self.E = np.pad( self.E, ((0,0),(0,ppc)), constant_values=0. )
            self.B = np.pad( self.B, ((0,0),(0,ppc)), constant_values=0. )

            # update state array and total particle count
            self.state = np.pad( self.state, (0,ppc), constant_values=True )
            self.N_alive += ppc
            self.N += ppc
            
            return

    def sort_particles(self, key='x'):
        """
        Sort the particle arrays using a specified quantity, 
        return the sorting indices also
        
        Still in development!
        """        
        
        sort = np.argsort( getattr( self, key) )
        
        self.state[:] = self.state[sort]
        self.x[:] = self.x[sort]
        self.x_old[:] = self.x_old[sort]
        self.w[:] = self.w[sort]
        self.rg[:] = self.rg[sort]
        self.l[:] = self.l[sort]
        self.r[:] = self.r[sort]
        
        # broadcast indexing a 2D-array causes the order to change to F for some reason
        self.v[:,:] = np.ascontiguousarray( self.v[:,sort] )
        self.p[:,:] = np.ascontiguousarray( self.p[:,sort] )
        self.E[:,:] = np.ascontiguousarray( self.E[:,sort] )
        self.B[:,:] = np.ascontiguousarray( self.B[:,sort] )

        return sort

    
    def compact_2D_array(self, arr):
        """
        Compute a compacted 2D array
        """
        A = np.empty((3,self.N_alive))
        
        A[0] = arr[0,self.state]
        A[1] = arr[1,self.state]
        A[2] = arr[2,self.state]
        
        return A
    
    def compact_particle_arrays(self, report_stats=False):
        """
        Compact the particle arrays by removing dead particles
        """
        
        if self.N_alive == self.N:
            return
        
        if report_stats:
            print('%.1f%% particles dead'%(100-100*self.N_alive/self.N))
            
        state = self.state

        self.x = self.x[state]
        self.x_old = self.x_old[state]
        self.w = self.w[state]
        self.rg = self.rg[state]
        self.l = self.l[state]
        self.r = self.r[state]
        
        # 2D arrays are slightly more involved
        self.v = self.compact_2D_array( self.v )
        self.p = self.compact_2D_array( self.p )
        self.E = self.compact_2D_array( self.E )
        self.B = self.compact_2D_array( self.B )
        
        self.N = self.N_alive
        self.state = np.full(self.N, True, dtype=bool)

        return
    
    def get_x(self):
        """Return only living particle positions"""
        return self.x[self.state]
    
    def get_p(self):
        """Return only living particle momenta"""
        return self.p[:,self.state]
    
    def get_gamma(self):
        """Return only living particle gamma factors"""
        return 1./self.rg[self.state]
    
    def get_w(self):
        """Return only living particle weights"""
        return self.w[self.state]
    
    def get_v(self):
        """Return only living particle velocities """
        return self.v[:,self.state]

    def get_KE(self):
        """Return only living particle KEs"""
        return (1./self.rg[self.state]  - 1.)*self.w[self.state] 