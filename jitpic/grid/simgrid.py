import numpy as np

class simgrid:
    """1D simulation grid"""

    def __init__(self, x0, x1, Nx, n_threads, particle_shape=1):
        """
        Initialise the simulation grid.
        
        The boundary conditions require 1 extra cell on the end of the 
        E and B field arrays, and 4 extra cells on the J array. 
        These should not be included in any diagnostics, hopefully I've caught
        all the instances of them potentially appearing. 
        
        x0        : str    : grid start position
        x1        : func   : grid end position
        Nx        : int    : number of cells
        n_threads : int    : number of CPU threads to be used
        """
        self.x0 = float(x0)
        self.x1 = float(x1)
        self.Nx = Nx
        
        self.x = np.linspace(x0, x1, Nx, endpoint=True)
        self.dx = self.x[1] - self.x[0]
        
        self.x2 = (self.x + self.dx/2.)
        
        self.rho = np.zeros(self.Nx)
        
        # E and B require at least one extra cell on the end for field solving
        # and probably more for interpolation
        self.NEB = max(1, 2*particle_shape)
        self.E = np.zeros((3,self.Nx+self.NEB))
        self.B = np.zeros((3,self.Nx+self.NEB))
        
        # J also requires extra cells for interpolation
        self.NJ = 2*(particle_shape+1)
        self.J = np.zeros((3,self.Nx+self.NJ))
        
        # we also need a higher dimensional arrays to avoid a race conditions in the
        # deposition methods
        self.rho_2D = np.zeros((n_threads,self.Nx))
        self.J_3D = np.zeros((n_threads,3,self.Nx+self.NJ))
        
        # forward index shift for field solving
        self.f_shift = np.arange(1, Nx+1, dtype=int) 
        return

    def move_grid(self):
        """
        Move the grid one cell forward in x
        """
        dx = self.dx
        
        # shift the grid along by one cell
        self.x0 += dx
        self.x1 += dx
        self.x += dx
        self.x2 += dx
        
        # roll the fields forward by one cell, zero the edge cells
        self.E = np.roll(self.E, -1, axis=1)
        self.E[:,self.Nx:] = 0.
        
        self.B = np.roll(self.B, -1, axis=1)
        self.B[:,self.Nx:] = 0.
        
        return
  
    def get_field(self, field):
        """
        Get the specified field without the edge cells
        
        field : str ('E','B','J') : field to extract 
        """
        
        if field == 'J':
            f = getattr(self, field)[:,:-self.NJ]
        elif field in ('E','B'):
            f = getattr(self, field)[:,:-self.NEB]
        else:
            print('unrecognised field')
            return
 
        return f
        
        