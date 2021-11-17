import numpy as np

class Laser:
    """
    Class containing laser information and methods
    """
    def __init__(self, a0, lambda_0=1., p=0, x0=0., ctau=1., d=1, theta_pol=0., cep=0.,
                 clip=None, method='box', x_antenna=0., 
                 t_stop=np.finfo(np.double).max, t0=0):
        """
        Initialise a laser object
        
        a0         : float      : normalised amplitude
        x0         : float      : laser centroid
        ctau       : float      : pulse duration
        lambda_0   : float      : normalised wavelength 
        p (1,0,-1) : int        : polarisation type 
        d (1,-1)   : int        : propagation direction 
        theta_pol  : float      : polarisation angle (0 - pi)
        cep        : float      : carrier envelope phase offset (0 - 2pi)
        clip       : float/None : Set the distance from x0 at which to 
                                  set the laser fields to 0 (optional)
        method     : str        : initialisation method, 'box' or 'antenna' 
        x_antenna  : float      : antenna position, ignored if method == 'box'
        t_stop     : float      : simulation time after which to turn off the 
                                  antenna (default never)
        t0         : float      : time offset for envelope positioning
        """
        
        self.a0 = a0
        self.lambda_0 = lambda_0
        self.p = p
        self.x0 = x0
        self.ctau = ctau
        self.d = d
        self.theta_pol = theta_pol
        self.cep = cep
        self.clip = clip
        # register the current simulation time to use as an offset
        # for lasers initialised after t=0
        self.t0 = t0
        
        self.method = method
        if method == 'antenna':
            self.is_antenna = True
            self.x_antenna = x_antenna
            self.t_stop = t_stop
        else:
            self.is_antenna = False
        
        return
    
    def phase(self, x, t):
        """
        Calculate the (real) phase of the laser at a given x and t
        """
        k_0 = 2.*np.pi/self.lambda_0
        return k_0*(x - (t-self.t0)*self.d) - self.cep 
    
    def envelope(self, x, t):
        """
        Calculate the envelope of the laser at a given x and t
        """
        return np.exp(-((x-self.x0-(t-self.t0)*self.d)/self.ctau)**2) 
        
    def box_fields( self, x ):
        """
        Calculate the laser E and B fields for an array of positions
        
        x : array : positions to calculate fields for

        returns:
            
        E : Electric fields, shape: (3,len(x)) 
        B : Magnetic fields, shape: (3,len(x))
        """
        
        #x = grid.x
        
        #phase and envelope
        phi = self.phase( x, 0 )
        psi = self.envelope( x, 0 )
        
        # polarisation vectors
        ex = np.cos(self.theta_pol) + 1j*self.p*np.sin(self.theta_pol)
        ey = np.sin(self.theta_pol) + 1j*self.p*np.cos(self.theta_pol)
        
        Ex = (self.a0 * ex * psi * np.exp(1j*phi))
        Ey = (self.a0 * ey * psi * np.exp(1j*phi))
        
        # clip the fields if specified
        if self.clip is not None: 
            Ex = np.where( x < (self.x0 - self.clip), 0., Ex)
            Ex = np.where( x > (self.x0 + self.clip), 0., Ex)
            Ey = np.where( x < (self.x0 - self.clip), 0., Ey)
            Ey = np.where( x > (self.x0 + self.clip), 0., Ey)
        
        # E = np.zeros_like(grid.E[:,:grid.NEB])
        # B = np.zeros_like(grid.B[:,:grid.NEB])
        E = np.zeros( (3,len(x)) )
        B = np.zeros( (3,len(x)) )
        
        E[1,:] =  Ex.real
        E[2,:] =  Ey.real
        
        # B = (1/k) x (E)
        B[1,:] = -self.d*Ey.real
        B[2,:] =  self.d*Ex.real
    
        return E, B
    
    def configure_antenna( self, sim ):
        """
        Configure the laser antenna. 
        
        Sets the cell index for field injection via floor divide
        i.e. grid.x[antenna_index] <= x_antenna
        """
        # check bounds
        assert self.x_antenna >= sim.grid.x0 and self.x_antenna <= sim.grid.x1, "antenna position must be inside the simulation box"
        
        # antennas must be buffered by at least one cell from the start of the box
        self.antenna_index =  max(1, int((self.x_antenna - sim.grid.x0) // sim.grid.dx) )
        
        return
    
    def antenna_fields( self, x, t):
        """
        Calculate the laser E and B fields at a single time and position
        
        x : float : position to calculate fields for
        t : float : instant to calculate fields for
        
        returns:
            
        E : Electric fields, shape: (3) 
        B : Magnetic fields, shape: (3)
        """

        phi = self.phase( x, t )
        psi = self.envelope( x, t )
        
        # polarisation vectors
        ex = np.cos(self.theta_pol) + 1j*self.p*np.sin(self.theta_pol)
        ey = np.sin(self.theta_pol) + 1j*self.p*np.cos(self.theta_pol)
        
        Ex = (self.a0 * ex * psi * np.exp(1j*phi))
        Ey = (self.a0 * ey * psi * np.exp(1j*phi))
        
        E = np.zeros(3)
        B = np.zeros(3)
        
        E[1] =  Ex.real
        E[2] =  Ey.real
        
        # B = (1/k) x (E)
        B[1] = -self.d*Ey.real
        B[2] =  self.d*Ex.real

        return E, B


        
        