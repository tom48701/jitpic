import numpy as np

class Laser:
    """
    Class containing laser information and methods
    """
    def __init__(self, a0, x0=0., ctau=1., lambda_0=1., p=0, d=1, theta_pol=0., 
                 cep=0., clip=None, method='box', x_antenna=0., 
                 t_stop=np.inf, t0=0.):
        """
        Initialise a laser object.
        
        Parameters
        ----------
        a0: float
            Normalised amplitude.
            
        x0: float
            Laser centroid position.
            
        ctau: float
            Pulse duration.

        lambda_0: float, optional
            Normalised wavelength (Default: 1).
            
        p: int (1, 0 or -1), optional
            Polarisation state, 0 corresponds to linear polarisation, 1 to
            left-circular polarisation and -1 to right-circular polarisation
            (Default: 0).
            
        d: int (1,-1), optional
            Propagation direction. A value of 1 corresponds to left-to-right
            propatation, and -1 to right-to-left propagation (Default: 1).
            
        theta_pol: float (in radians), optional
            Polarisation angle, (default: 0).
            
        cep: float (in radians), optional
            Carrier envelope phase offset (Default: 0).
            
        clip: float or None, optional
            The distance from x0 at which to set the laser fields to 0. 
            Fields are not clipped if None (Default: None).
            
        method: str (`box` or `antenna`), optional
            Laser injection method.
            - `box` initialises the laser on the whole grid instantaneously.
            - `antenna` initialises the laser progressively from the cell
               closest to `x_antenna`.
            (Default: `box`)
            
        x_antenna: Float or None, optional
            Antenna position. If None, uses the grid start position
            (Default: None).
            
        t_stop: float, optional
            Simulation time after which to turn off the antenna (Default: inf).
            
        t0: float, optional
            Simulation time at initialisation. Required as an offset for 
            antennas, igonred if method=`box` (Default: 0).
        """
        # register quantities
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
        # set quantities depending on the initialisation method
        self.method = method
        if method == 'antenna':
            self.is_antenna = True
            self.x_antenna = x_antenna
            self.t_stop = t_stop
        elif method == 'box':
            self.is_antenna = False
        return
    
    def phase(self, x, t):
        """ Calculate the phase angle of the laser at a given x and t. """
        k_0 = 2.*np.pi/self.lambda_0
        return k_0*(x - (t-self.t0)*self.d) - self.cep 
    
    def envelope(self, x, t):
        """ Calculate the envelope of the laser at a given x and t. """
        return np.exp(-((x-self.x0-(t-self.t0)*self.d)/self.ctau)**2) 
        
    def box_fields( self, x ):
        """
        Calculate the laser E and B fields for an array of positions
        
        Parameters
        ----------
        x: array
            Positions to calculate fields for

        Returns
        -------    
        E: array
            Electric fields. 
        B: array
            Magnetic fields.
        """
        #phase and envelope
        phi = self.phase( x, 0 )
        psi = self.envelope( x, 0 )
        # polarisation vectors
        ex = np.cos(self.theta_pol) + 1j*self.p*np.sin(self.theta_pol)
        ey = np.sin(self.theta_pol) + 1j*self.p*np.cos(self.theta_pol)
        # complex E-fields
        Ex = (self.a0 * ex * psi * np.exp(1j*phi))
        Ey = (self.a0 * ey * psi * np.exp(1j*phi))
        # clip the fields if specified
        if self.clip is not None: 
            Ex = np.where( x < (self.x0 - self.clip), 0., Ex)
            Ex = np.where( x > (self.x0 + self.clip), 0., Ex)
            Ey = np.where( x < (self.x0 - self.clip), 0., Ey)
            Ey = np.where( x > (self.x0 + self.clip), 0., Ey)
        # full E and B field arrays
        E = np.zeros( (3,len(x)) )
        B = np.zeros( (3,len(x)) )
        # real E fields
        E[1,:] =  Ex.real
        E[2,:] =  Ey.real
        # B = k/|k| cross E
        B[1,:] = -self.d*Ey.real
        B[2,:] =  self.d*Ex.real
        return E, B
    
    def configure_antenna( self, sim ):
        """
        Configure the laser antenna. 
        
        This method sets the cell index for field injection via floor divide
        of the antenna position against the cell vertices such that 
        grid.x[antenna_index] <= x_antenna
        """
        # check bounds
        assert self.x_antenna >= sim.grid.x0 and self.x_antenna <= sim.grid.x1, "antenna position must be inside the simulation box"
        # antennas must be buffered by at least one cell from the start of the box
        self.antenna_index =  max(1, int((self.x_antenna - sim.grid.x0) // sim.grid.dx) )
        return
    
    def antenna_fields( self, x, t):
        """
        Calculate the laser E and B fields at a single time and position
        
        Parameters
        ----------
        x: float
            Position to calculate fields for.
        t: float
            Time to calculate fields for.
        
        Returns
        -------   
        E: array
            Electric fields .
        B: array
            Magnetic fields.
        """
        # phase and envelope
        phi = self.phase( x, t )
        psi = self.envelope( x, t )
        # polarisation vectors
        ex = np.cos(self.theta_pol) + 1j*self.p*np.sin(self.theta_pol)
        ey = np.sin(self.theta_pol) + 1j*self.p*np.cos(self.theta_pol)
        # complex E-fields
        Ex = (self.a0 * ex * psi * np.exp(1j*phi))
        Ey = (self.a0 * ey * psi * np.exp(1j*phi))
        # E and B fields
        E = np.zeros(3)
        B = np.zeros(3)
        # real E field
        E[1] =  Ex.real
        E[2] =  Ey.real
        # B = (1/k) cross E
        B[1] = -self.d*Ey.real
        B[2] =  self.d*Ex.real
        return E, B