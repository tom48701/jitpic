import numpy as np

class Simgrid:
    """1D simulation grid"""

    def __init__(self, x0, x1, Nx, n_threads, boundaries, particle_shape=1):
        """
        Initialise the simulation grid.
        
        The boundary conditions require a variable number of extra cells on
        the grid depending on the particle shape factor. These cells are
        automatically excluded from the written diagnostics, but when 
        extracting grid quantities directly, the `get_field` methods should be 
        used rather than calling the properties directly.

        Parameters
        ----------
        x0: float
            Grid start point, must be < x1.
           
        x1: float
            Grid end point, must be > x0.
          
        Nx: int
            Number of cells.
            
        n_threads: int
            The number of CPU threads to be used. Required to avoid parallel
            race conditions in current and charge density deposition
        
        boundaries: str
            The intended grid boundary condition. Required to determine how
            many extra cells are to be added to the grid.
        
        particle_shape: int, optional
            The intended particle shape factor. Required to determine how many
            extra cells are to be added to the grid (Default: 1).
        """
        # register the start/end/number of gridpoints
        self.x0 = float(x0)
        self.x1 = float(x1)
        self.Nx = Nx
        # define the gridpoints, cell size and inverse cell size
        self.x = np.linspace(x0, x1, Nx, endpoint=True)
        self.dx = self.x[1] - self.x[0]
        self.idx = 1./self.dx
        # define the cell midpoints
        self.x2 = (self.x + self.dx/2.)
        # charge density arrays
        self.rho = np.zeros(self.Nx)
        self.rho_2D = np.zeros((n_threads,self.Nx))
        # open boundaries:
        # E and B require at least one extra cell on the end for field solving
        # and probably more for interpolation. J also requires extra cells for interpolation
        # we also need a higher dimensional arrays to avoid a race conditions in the
        # deposition methods
        self.boundaries = boundaries
        if boundaries == 'open':
            # work out how many extra cells to pad the arrays with
            iEB = max(1, 2*particle_shape)
            iJ = 2*(particle_shape+1)
            # register these quantities
            self.NEB = -iEB
            self.NJ = -iJ
            # set up field arrays
            self.E = np.zeros((3,self.Nx+iEB))
            self.B = np.zeros((3,self.Nx+iEB))
            self.J = np.zeros((3,self.Nx+iJ))
            self.J_3D = np.zeros((n_threads,3,self.Nx+iJ))
            # forward index shift for field solving
            self.f_shift = np.arange(1, Nx+1, dtype=int)   
        elif boundaries == 'periodic':
            # no extra cells required
            self.NEB = None
            self.NJ = None
            # set up field arrays
            self.E = np.zeros((3,self.Nx))
            self.B = np.zeros((3,self.Nx))
            self.J = np.zeros((3,self.Nx))
            self.J_3D = np.zeros((n_threads,3,self.Nx))
            # forward index shift for field solving
            self.f_shift = np.roll(np.arange(0, Nx, dtype=int), -1 )
        else:
            raise ValueError('Invalid boundary condition in grid setup')
        return

    def move_grid(self):
        """ Move the grid one cell forward in x. """
        dx = self.dx
        # shift the grid along by one cell
        self.x0 += dx
        self.x1 += dx
        self.x  += dx
        self.x2 += dx
        # roll the fields back by one cell, zero the edge cells
        self.E[:,:] = np.roll(self.E, -1, axis=1)
        self.E[:,self.Nx:] = 0.
        self.B[:,:] = np.roll(self.B, -1, axis=1)
        self.B[:,self.Nx:] = 0.
        return
  
    def get_field(self, field):
        """
        Get the specified field without edge cells.
        
        Parameters
        ----------
        field: str (`E`, `B`, `J` or `S`)
            Field to extract
            - `E` returns the electric field.
            - `B` returns the magnetic field.
            - `J` returns the most recent current deposition.
            - `S` returns the Poynting vector.
              
        Returns
        -------
        f: array
            The specified field
        """
        if field == 'J':
            f = getattr(self, field)[:,:self.NJ]
        elif field in ('E','B'):
            f = getattr(self, field)[:,:self.NEB]
        elif field == 'S':
            E = getattr(self, 'E')[:,:self.NEB]
            B = getattr(self, 'B')[:,:self.NEB]
            f = np.cross( E.T, B.T).T
        else:
            print('unrecognised field')
            return
        return f
