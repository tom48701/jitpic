import numpy as np

class laser:
    """
    Class containing laser information and methods
    """
    def __init__(self, a0, lambda_0=1., p=0, x0=0., tau=1., d=1, theta_pol=0., cep=0.,
                 clip=None):
        """
        Initialise a laser object
        
        a0         : float      : normalised amplitude
        x0         : float      : laser centroid
        tau        : float      : pulse duration
        lambda_0   : float      : normalised wavelength 
        p (1,0,-1) : int        : polarisation type 
        d (1,-1)   : int        : propagation direction 
        theta_pol  : float      : polarisation angle (0 - pi)
        cep        : float      : carrier envelope phase offset (0 - 2pi)
        clip       : float/None : Set the distance from x0 at which to 
                                  set the laser fields to 0 (optional)
        """
        
        self.a0 = a0
        self.lambda_0 = lambda_0
        self.p = p
        self.x0 = x0
        self.tau = tau
        self.d = d
        self.theta_pol = theta_pol
        self.cep = cep
        self.clip = clip
        
        return
    
    def fields( self, grid ):
        """
        Calculate the laser E and B fields 
        
        grid : simgrid : parent grid object
        """
        def a(x, x0, tau):
            return np.exp(-((x-x0)/tau)**2)
        
        def phase(x):
            k_0 = 2.*np.pi/self.lambda_0
            return k_0*x - self.cep   
               
        x = grid.x
        
        #phase and envelope
        phi = phase( x )
        psi = a( x, self.x0, self.tau )
        
        # polarisation vectors
        ex = np.cos(self.theta_pol) + 1j*self.p*np.sin(self.theta_pol)
        ey = np.sin(self.theta_pol) + 1j*self.p*np.cos(self.theta_pol)
        
        Ex = (self.a0 * ex * psi * np.exp(1j*phi))
        Ey = (self.a0 * ey * psi * np.exp(1j*phi))
        
        if self.clip is not None: 
            Ex = np.where( x < (self.x0 - self.clip), 0., Ex)
            Ex = np.where( x > (self.x0 + self.clip), 0., Ex)
            Ey = np.where( x < (self.x0 - self.clip), 0., Ey)
            Ey = np.where( x > (self.x0 + self.clip), 0., Ey)
        
        E = np.zeros_like(grid.E[:,:-grid.NEB])
        B = np.zeros_like(grid.B[:,:-grid.NEB])
        
        E[1,:] =  Ex.real
        E[2,:] =  Ey.real
        
        # B = (1/k) x (E)
        B[1,:] = -self.d*Ey.real
        B[2,:] =  self.d*Ex.real
    
        return E, B