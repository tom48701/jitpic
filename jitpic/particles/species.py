import numpy as np

class species:
    """ Particle Species """
    
    def __init__(self, name, ppc, n, p0, p1, m=1., q=-1., eV=0., dens=None ):
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
        """
        
        def default_profile(x):
            return np.ones_like(x)
        
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
        
        # use the default (flat) density profile if none is specified
        if dens is None:
            self.dfunc = default_profile
        else:
            self.dfunc = dens
            
        return

    def initialise_particles(self, grid, n_threads):
        """
        Seat the particles onto the grid
        
        grid : simgrid : grid object onto which to seat particles
        """
        
        def position(x, dx, n):
            """
            Compute the first or last particle location given a limit.
            x (float) : lower/upper limit
            n (int)   : denotes if we are computing a first or last particle,
                n = 1 for first
                n = 3 for last 
            
            p0/p1 denote the plasma start/end positions, NOT neccesarily the 
            positions of the first/last particles. Interparticle spacing is
            held constant. Therefore, particles are distributed by working out 
            the first and last particle positions, and hence the total number 
            of particles. N will be at most (ppc*Nx).
            """

            i = np.floor(x/dx) # nearest left cell
            dxi = np.floor((x - i*dx) / self.ddx) # closest integer ddx interval to x
            pi = 2*np.floor((dxi + 3) / 2) - n # closest legal particle location

            return i*grid.dx + pi*self.ddx # final real position
        
        self.ddx = grid.dx/(2*self.ppc) # inter-particle half-spacing
        
        first_particle = position(self.p0, grid.dx, 1)
        last_particle = position(self.p1, grid.dx, 3)
        
        assert (first_particle > grid.x0) and (last_particle < grid.x1), 'Particles initialised outside the grid!'

        self.N = int((self.p1-self.p0) / (2*self.ddx))

        self.E = np.zeros((3,self.N))
        self.B = np.zeros((3,self.N))     
        
        self.state = np.full(self.N, True, dtype=bool)
        self.N_alive = self.N # all particle should be alive at first
        
        self.x = np.zeros(self.N)
        self.x[:] = np.linspace(first_particle, last_particle, self.N) 
        
        self.x_old = np.zeros_like(self.x)
        
        self.w = np.full(self.N, self.n/self.ppc) * self.dfunc(self.x) # weight

        Ek = self.eV / 5.11e5 # / electron rest mass energy
        pi = np.sqrt(Ek**2 + 2.*Ek*self.m) / np.sqrt(3) # initial momentum
    
        self.p = np.zeros((3,self.N))       
        if Ek > 0.:
            self.p = np.random.normal(0., pi/np.sqrt(2.), self.p.shape) # initial thermal momentum
            
        self.rg = 1./np.sqrt(1. + (self.p**2).sum(axis=0)/self.m**2)  # reciprocal gamma factor      
        
        self.v = self.velocity()
        
        # setup array to contain the left and right cell indices
        self.l = np.zeros(self.N, dtype=np.dtype('u4'))
        self.r = np.zeros(self.N, dtype=np.dtype('u4'))
        self.l[:] = np.floor((self.x-grid.x0)/grid.dx) # left cells
        self.r[:] = self.l+1 # grid fields index - should be safe
        
        return
    
    def velocity(self):
        """
        Return the particle velocities 
        """
        return self.p[:,self.state] * self.rg[self.state] / self.m 

    def KE(self):
        """
        Return the particle KEs
        """
        return (1./self.rg[self.state]  - 1.)*self.m*self.w[self.state] 