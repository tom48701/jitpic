import numba, time, h5py, gc
import numpy as np
import matplotlib.pyplot as plt 
# import JitPIC classes
from .grid import Simgrid
from .particles import Species
from .laser import Laser
from .diagnostics import Timeseries_diagnostic
from .external_fields import External_field
# import JitPIC helper functions
from .utils import make_directory, default_inline_plotting_script
# import numba function dictionary (see: numba_functions/__init__.py)
from .numba_functions import function_dict

class Simulation:
    """
    The main simulation object, containing all the PIC methods, related
    functions and diagnostic methods.
    """
    def __init__(self, x0, x1, Nx, species=[], 
                 particle_shape=1, diag_period=0, 
                 plotfunc=default_inline_plotting_script,
                 n_threads=numba.get_num_threads(), seed=0,
                 resize_period=100, boundaries='open',
                 pusher='cohen', diagdir='diags', imagedir='images'):
        """ 
        Initialise the simulation, set up the grid at the same time
        
        x0             : float     : grid start point
        x1             : float     : grid end point
        Nx             : int       : number of cells
        species        : list      : a list of `species' objects
        particle_shape : int       : particle shape factor (higher = smoother)
        diag_period    : int       : number of steps between diagnostic writes
        plotfunc       : None/func : function to instruct the simulation what
                                     figure to write alongside the diagnostics.
                                     plotfunc should take the simulation object
                                     as an argument, and return a matplotlib
                                     figure object.
        n_threads      : int       : number of CPU threads to use
        seed           : int       : set the RNG seed
        resize_period  : int       : interval between particle buffer resizing
        boundaries     : str       : boundary conditions; 'open' or 'periodic'
        pusher         : str       : particle pusher to use

        """
        # set the RNG seed for reproducability]
        self.seed = seed
        np.random.seed(seed)
        # register the number of threads, and set the numba variable accordingly
        self.n_threads = n_threads
        numba.set_num_threads(n_threads)
        # register the simulation time and iteration
        self.t = 0.
        self.iter = 0
        # initialise the simulation grid and register the timestep (dt == dx)
        self.grid = Simgrid(x0, x1, Nx, self.n_threads, boundaries, particle_shape=particle_shape)
        self.dt = self.grid.dx
        # register the particle shape factor and boundary conditions
        self.particle_shape = particle_shape
        self.boundaries = boundaries
        ## choose the correct numba functions based on simulation settings
        # set the particle reseating function
        self.reseat_func = function_dict['reseat_%s'%boundaries]
        # configure the particle pusher
        self.pusher = pusher
        self.particle_push_func = function_dict[pusher]            
        # set the correct associated constant factor for particle pushing
        if pusher == 'boris':
            self.pushconst = np.pi 
        elif pusher in ('cohen', 'vay'):
            self.pushconst = 2*np.pi 
        # set the current deposition function
        self.deposit_J_func = function_dict['J%i_%s'%(particle_shape, boundaries)] 
        # set the field interpolation function
        self.interpolate_func = function_dict['I%i_%s'%(particle_shape, boundaries)]
        # set the charge deposition function
        self.deposit_rho_func = function_dict['R%i_%s'%(particle_shape, boundaries)]  
        # register an empty list for lasers
        self.lasers = []
        # register empty list for particle species
        self.species = []
        # add any pre-defined species
        self.Nspecies = len(species) 
        if self.Nspecies > 0:  
            for i in range(self.Nspecies):
                spec = species[i]
                self.append_species(spec)
        # register the particle array resize period
        self.resize_period = resize_period
        # register the diagnostic period and associated directories
        self.diag_period = diag_period 
        self.diagdir = diagdir
        self.imagedir = imagedir
        # register the plotting function
        self.inline_plotting_script = plotfunc
        # register no external fields initially
        self.external_fields = []
        # initialise the timeseries diagnostic
        self.tsdiag = None
        #  and moving window state
        self.moving_window = False
        return

    def set_moving_window(self, state):
        """
        Activate or deactivate the moving window.
        Cannot be used with periodic boundaries!
        
        state : bool : moving window on/off
        """
        assert self.boundaries == 'open', 'Moving window must employ open boundaries'
        
        self.moving_window = state
        return
    
    def step( self, N=1, silent=False ):
        """
        advance the simulation by N cycles, automatically write diagnostics,
        timeseries data and images.
        
        N      : int  : number of steps to perform
        silent : bool : disable print diagnostic information during the run
        """
        # print summary information
        if not silent:
            print( 'Starting from t = %f with timestep dt = %f'%(self.t, self.dt) )
            print( 'Grid consists of %i cells'%self.grid.Nx )
            print( 'Simulating %i particles of shape order %i using the %s pusher'%( sum([ spec.N for spec in self.species]), self.particle_shape, self.pusher) )
            print( 'Employing %s boundary conditions'%self.boundaries )
            if self.moving_window:
                print('Moving window active')
            print( 'Using %i threads via %s'%(numba.get_num_threads(), numba.threading_layer()) )
            print( 'performing %i PIC cycles from from step %i\n'%(N, self.iter) )
        # timing information
        t0 = time.time()
        t1 = t0
        # begin the loop
        for i in range(N):
            # p,v must be offset to n=-1/2 before the first step
            if self.iter == 0:
                self.apply_initial_offset_to_pv()
            ## periodic diagnostic operations
            if self.diag_period > 0 and self.iter%self.diag_period == 0:
                if not silent:
                    print('Writing diagnostics at step %i (t = %.1f)'%(self.iter, self.t))
                    print('%.1f (%.1f) seconds elapsed (since last write)\n'%(time.time()-t0, time.time()-t1) )
                # register split time and write diagnostics
                t1 = time.time()   
                self.write_diagnostics()
                # generate figure
                if self.inline_plotting_script is not None:
                    self.plot_result(imagedir=self.imagedir)  
                # append to the timeseries diagnostics
                if self.tsdiag is not None and self.iter > self.tsdiag.istart:
                    self.tsdiag.write_data(self)
            ## every-step operations
            # gather timeseries data
            if self.tsdiag is not None:
                    self.tsdiag.gather_data(self)
            # introduce lasers from any antennas
            for laser in self.lasers:
                if laser.is_antenna:
                    self.inject_antenna_fields(laser)     
            ## main PIC cycle
            #print(self.iter)
            # E,B,x at n
            # J,p,v at n-1/2
            # interpolate fields onto particles
            self.apply_fields_to_particles()
            # push p,v to n+1/2, x to n+1
            self.push_particles( self.dt ) 
            # push J to n+1/2 
            self.deposit_current()  
            #push E,B to n+1
            self.push_fields() 
            ## post-cycle operations
            # advance the moving window
            if self.moving_window:
                self.grid.move_grid()
                # inject new particles
                for spec in self.species:
                    spec.inject_particles(self.grid)
                # periodically clean up dead particles
                if self.resize_period > 0 and self.iter%self.resize_period == 0:
                    for spec in self.species:
                        spec.compact_particle_arrays()    
            # reseat and mask particles    
            self.reseat_particles()
            # advance simulation time and iteration
            self.t += self.dt
            self.iter += 1
        print( 'Finished in %.3f s'%(time.time()-t0))
        print( 'Now at t = %.3e\n'%self.t)
        return  
    
    def add_new_species(self, name, ppc, n, p0, p1, m=1., q=-1., eV=0., dens=None):
        """ 
        Wrapper function for initialising and adding a new particle species
        to the simulation.

        name  : str    : species name for diagnostic purposes
        ppc   : int    : number of particles per cell
        p0    : float  : species start position
        p1    : float  : species end position
        m     : float  : particle mass in electron masses (optional)
        q     : float  : particle charge in elementary charges (optional)
        eV    : flaot  : species temperature in electron volts (optional)
        dens  : func   : function describing the density profile, the function
                         takes a single argument which is the grid x-positions
        """
        
        new_species = Species( name, ppc, n, p0, p1, dens=dens, m=m, q=q, eV=eV )
        
        self.append_species( new_species )
        
        return
    
    def append_species(self, spec):
        """
        Initialise and append a species object to the list of species
        """
    
        spec.initialise_particles(self.grid)
        
        self.species.append(spec)
        self.Nspecies = len(self.species)
        
        return

    def deposit_rho(self):
        """ 
        Deposit the total charge density on the grid.
        For diagnostics only!
        """
        grid = self.grid
        grid.rho_2D[:,:] = 0.
        
        for spec in self.species:

            # calculate the index splits for current deposition - assume masked arrays
            indices = [ i*spec.N//self.n_threads for i in range(self.n_threads)]+[spec.N-1] 
            indices = np.array(indices)
            
            self.deposit_rho_func(self.n_threads, spec.x, grid.idx, 
                                  spec.q*spec.w, grid.rho_2D, 
                                  spec.l, spec.r, spec.state,
                                  indices, grid.x, grid.Nx)
            
        grid.rho[:] = grid.rho_2D.sum(axis=0)
        
        return grid.rho

    def deposit_single_species(self, spec):
        """ 
        Deposity the charge density for a single species on the grid.
        For diagnostics only!
        
        spec  : species : the species to deposit
        """    
        grid = self.grid
        grid.rho_2D[:,:] = 0.
        
        # calculate the index splits for charge deposition - assume masked arrays
        indices = [ i*spec.N//self.n_threads for i in range(self.n_threads)]+[spec.N-1] 
        indices = np.array(indices)
        
        self.deposit_rho_func(self.n_threads, spec.x, grid.idx, 
                              spec.q*spec.w, grid.rho_2D, 
                              spec.l, spec.r, spec.state,
                              indices, grid.x, grid.Nx)
        
        grid.rho[:] = grid.rho_2D.sum(axis=0)
        
        return grid.rho
    
    def apply_initial_offset_to_pv(self):
        """
        Apply the initial half-step backwards to particle momentum and velocity
        """
        self.apply_fields_to_particles()
        self.push_particles( -self.dt*0.5 )
        # x has also been pushed to -1/2, but
        # x stays on integer timesteps, so revert the position
        for spec in self.species:
            spec.revert_x()
        return
       
    def deposit_current(self):
        """
        deposit current onto the grid.
        Race conditions can occur in Numba's array-reductions!
        This method determines the start and end indices for  n_thread chunks, 
        then performs the current deposition for each thread individually. 
        Finally, the contributions from each thread are reduced in serial to
        avoid the race condition.
        """
        grid = self.grid
        # zero the current
        grid.J_3D[:,:,:] = 0.
        
        for spec in self.species:
            # calculate the index splits for current deposition 
            indices = [ i*spec.N//self.n_threads for i in range(self.n_threads)]+[spec.N-1] 
            indices = np.array(indices)
    
            self.deposit_J_func( spec.x, spec.x_old, spec.w, spec.v, grid.J_3D,
                          self.n_threads, indices, spec.q, grid.x,
                          spec.state, grid.idx, grid.x0*grid.idx)

        # reduce the current to 2D
        grid.J[:,:] = grid.J_3D.sum(axis=0)
        
        return

    def add_external_field(self, field, component, magnitude, function=None):
        """
        Add an constant external field to the simulation
        
        field ('E','B')   : str           : the type of field to add
        component (0,1,2) : int           : field component index 
        magnitude         : float         : magnitude of the field
        function          : func or None  : field profile, uniform if None
        """
        
        self.external_fields.append( External_field(field, component, magnitude,
                                                    function=function) )
        return
    
    def apply_fields_to_particles(self):
        """
        interpolate grid fields onto particles.
        """
              
        for spec in self.species:
            
            self.interpolate_func(self.grid.E,self.grid.B,
                              spec.E, spec.B,
                              spec.l, spec.r, 
                              (spec.x - self.grid.x0)*self.grid.idx,
                              spec.N )
        return 
    
    def push_fields(self):
        """
        advance the fields in time using a numerical-dispersion free along x 
        (NDFX) field solver. This requires that dx == dt.
        """
        
        grid = self.grid
        
        pidt = np.pi*self.dt
        
        grid.E[0,:grid.NEB] -= 2. * pidt * grid.J[0,:grid.NJ]
             
        PR = (grid.E[1] + grid.B[2]) * .5
        PL = (grid.E[1] - grid.B[2]) * .5
        SR = (grid.E[2] - grid.B[1]) * .5
        SL = (grid.E[2] + grid.B[1]) * .5
   
        PR[grid.f_shift] = PR[:grid.NEB] - pidt * grid.J[1,:grid.NJ]
        PL[:grid.NEB] = PL[grid.f_shift] - pidt * grid.J[1,:grid.NJ]
        SR[grid.f_shift] = SR[:grid.NEB] - pidt * grid.J[2,:grid.NJ]
        SL[:grid.NEB] = SL[grid.f_shift] - pidt * grid.J[2,:grid.NJ]
        
        grid.E[1,:grid.NEB] = PR[:grid.NEB] + PL[:grid.NEB]
        grid.B[2,:grid.NEB] = PR[:grid.NEB] - PL[:grid.NEB]
        grid.E[2,:grid.NEB] = SL[:grid.NEB] + SR[:grid.NEB]
        grid.B[1,:grid.NEB] = SL[:grid.NEB] - SR[:grid.NEB]
        
        return  

    def push_particles(self, dt):
        """
        Advance the particle momenta and positions in time using the chosen 
        particle pusher.
        
        All particles are pushed, dead particles included.
        
        backstep : bool : perform a half-step backwards push, used during 
                          initial setup to offset p and v to -1/2. x must be
                          manually reverted to keep it on integer step
        """ 
        for spec in self.species:

            x = spec.x
            E = spec.E
            B = spec.B
                
            # apply external fields
            if len(self.external_fields) > 0:
                for ext_field in self.external_fields:
                    if ext_field.field == 'E':   
                        E = ext_field.add_field(x, self.t, E)
                    else:
                        B = ext_field.add_field(x, self.t, B)

            self.particle_push_func( E, B, dt*self.pushconst*spec.qm, spec.p, spec.v, x, spec.x_old, 
                                    spec.rg, spec.m, dt, spec.N) 
                      
        return

    def reseat_particles(self):
        """
        Reseat particles in new cells, look for and disable any particles 
        that got pushed off the grid.
        """ 
        for spec in self.species:

            self.reseat_func(spec.N, spec.x, spec.state, spec.l, spec.r,
                         self.grid.x0, self.grid.x1, self.grid.dx, self.grid.idx, self.grid.Nx)
            
            spec.N_alive = spec.state.sum()
        return
        
    def add_laser(self, a0, x0, ctau, lambda_0=1., p=0, d=1, theta_pol=0., 
                  cep=0., clip=None, method='box', x_antenna=None, 
                  t_stop=np.finfo(np.double).max):
        """
        Register and introduce a laser to the simulation object.
        
        a0         : float      : normalised amplitude
        x0         : float      : laser centroid
        ctau       : float      : pulse duration 
        lambda_0   : float      : normalised wavelength 
        p (1,0,-1) : int        : polarisation type (LCP, LP, RCP)
        d (1,-1)   : int        : propagation direction forwards/backwards
        theta_pol  : float      : polarisation angle (0 - pi)
        cep        : float      : carrier envelope phase offset (0 - 2pi)
        clip       : float/None : Set the distance from x0 at which to 
                                  set the laser fields to 0
        method     : str        : laser injection method, 'box' or 'antenna'
        x_antenna  : None/float : antenna position#
        t_stop     : float      : simulation time after which to turn off the 
                                  antenna (default never)
        """
        if x_antenna is None:
            x_antenna = self.grid.x0
            
        new_laser = Laser(a0, lambda_0=lambda_0, p=p, x0=x0, ctau=ctau, d=d, 
                          theta_pol=theta_pol, clip=clip, cep=cep,
                          method=method, x_antenna=x_antenna,
                          t_stop=t_stop, t0=self.t)
        
        self.lasers.append(new_laser)
        
        if new_laser.method == 'box':
            E_laser, B_laser = new_laser.box_fields( self.grid.x )
            
            self.grid.E[:,:self.grid.NEB] += E_laser
            self.grid.B[:,:self.grid.NEB] += B_laser
            
        elif new_laser.method == 'antenna':
            self.has_antennas = True
            new_laser.configure_antenna( self )
   
        return
    
    def inject_antenna_fields(self, laser):
        """
        Inject laser fields into one grid cell
        """
        if laser.is_antenna and self.t < laser.t_stop:
            E_laser, B_laser = laser.antenna_fields( self.grid.x[laser.antenna_index], self.t )
                        
            self.grid.E[:,laser.antenna_index] += E_laser
            self.grid.B[:,laser.antenna_index] += B_laser

        return
    
    def write_diagnostics(self):
        """ 
        Write all grid quantities to file 
        
        diagdir : str : folder to write diagnostics to
        """

        make_directory(self.diagdir)
            
        fname = '%s/diags-%.8i.h5'%(self.diagdir, self.iter)
        
        with h5py.File(fname, 'w') as f:
            
            f.attrs['iter'] = self.iter
            f.attrs['time'] = self.t
            f.attrs['dt'] = self.dt
            
            E = self.grid.get_field('E')
            B = self.grid.get_field('B')
            J = self.grid.get_field('J')
            
            f.create_dataset('x', data=self.grid.x)
            
            f.create_dataset('Ex', data=E[0] )
            f.create_dataset('Ey', data=E[1] )
            f.create_dataset('Ez', data=E[2] )
            f.create_dataset('Bx', data=B[0] )
            f.create_dataset('By', data=B[1] )
            f.create_dataset('Bz', data=B[2] )
            f.create_dataset('Jx', data=J[0] )
            f.create_dataset('Jy', data=J[1] )
            f.create_dataset('Jz', data=J[2] )    
            
            f.create_dataset('rho', data=self.deposit_rho())
            for spec in self.species:
                f.create_dataset(spec.name, data=self.deposit_single_species(spec) )
   
        return
        
    def plot_result(self, index=None, dpi=220, fontsize=8, lambda0=8e-7, imagedir='images'):
        """
        Plot and save summary images.

        index : int      : specify a numerical index for the image, otherwise
                           the current simulation iteration is used
        dpi      : int   : image DPI, to quickly scale the images up or down in size
        fontsize : int   : reference fontsize to be used in the images
        lambda0  : float : real wavelength to be used for any conversions
        imagedir : str   : folder in which to save the images
        """
        
        # make the image directory if it doesn't already exist
        make_directory(imagedir)
        
        # generate a figure
        fig = self.inline_plotting_script( self, fontsize=fontsize )
   
        # set the file name and save
        if index is not None:
            fig.savefig('%s/%.8i.png'%(imagedir, index), dpi=dpi)            
        else:
            fig.savefig('%s/%.8i.png'%(imagedir, self.iter), dpi=dpi)
        
        # clean up
        plt.close(fig) 
        gc.collect()
        
        return
    
    def add_timeseries_diagnostic( self, fields, cells, fname='timeseries_data'):
        """
        Add a timeseries diagnostic to the simulation
        
        This will automatically collect the specified field information from
        the specified cells at every timestep, and continually append the
        timeseries file throughout the simulation.

        fields  : list of str : list of fields to track ('E','B','J')
        cells   : list of int : list of cells to track [0 : Nx-1]
        fname   : str         : output file name
        """
        
        self.tsdiag = Timeseries_diagnostic(self, fields, cells,
                                            buffer_size=self.diag_period,
                                            diagdir=self.diagdir, fname=fname)      
        
        