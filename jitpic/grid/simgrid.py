import numpy as np

class simgrid:
    """1D simulation grid"""

    def __init__(self, x0, x1, Nx, n_threads):
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
        
        # E and B have one extra cell on the end for field solving purposes
        self.E = np.zeros((3,self.Nx+1))
        self.B = np.zeros((3,self.Nx+1))
        
        # J has four extra cells to deal with the boundary condition
        self.J = np.zeros((3,self.Nx+4))
        # also we need a 3D array to avoid a race condition in the
        # current deposition method
        self.J_3D = np.zeros((n_threads,3,self.Nx+4))
        
        # forward index shift for field solving
        self.f_shift = np.arange(1, Nx+1, dtype=int) 
        return

    def get_field(self, field):
        """
        Get the specified field without the buffer cells
        
        field : str ('E','B','J') : field to extract 
        """
        
        if field == 'J':
            f = getattr(self, field)[:,:-4]
        elif field in ('E','B'):
            f = getattr(self, field)[:,:-1]
        else:
            print('unrecognised field')
            return
 
        return f
        
        