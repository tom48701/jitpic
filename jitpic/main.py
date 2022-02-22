import numba, time, h5py, gc, os
import numpy as np
import matplotlib.pyplot as plt 
# import meta information
from . import __version__ 
# import JitPIC classes
from .grid import Simgrid
from .particles import Species
from .laser import Laser
from .diagnostics import Timeseries_diagnostic, Particle_tracking_diagnostic
from .external_fields import External_field
# import JitPIC helper functions
from .utils import make_directory, summary_fig_func, check_for_file
# import numba function dictionary (see: numba_functions/__init__.py)
from .numba_functions import function_dict

class Simulation:
    """
    The main simulation class, containing all the PIC methods, related
    functions and diagnostic methods.
    """
    def __init__(self, x0, x1, Nx, species=[], 
                 particle_shape=1, diag_period=0, 
                 plotfunc=summary_fig_func,
                 n_threads=numba.get_num_threads(), seed=0,
                 resize_period=100, boundaries='open',
                 pusher='cohen', diagdir='diags', imagedir='images',
                 sort_period=0, moving_window=False):
        """ 
        Initialise a simulation.
        
        Initialise a simulation and construct the grid at the same time. By
        default, the simulation is initialised with no particles, lasers, or 
        diagnostics.
        
        Parameters
        ----------
        x0: float
            Grid start point, must be < x1.
           
        x1: float
            Grid end point, must be > x0.
          
        Nx: int
            Number of cells.
            
        species: list, optional
            A list of `Species` objects to initialise with the simulation
            (Default: empty).
            
        particle_shape: int, optional
            The particle shape factor for current/charge dposition and field
            gathering. Shapes up to order 4 (quartic) are implemented
            (Default: 1).
              
        diag_period: int, optional
            The number of steps between diagnostic writes. All diagnostics
            are governed by this write period. A value of zero turns off all
            diagnostics (Default: 0).
            
        plotfunc: func or None, optional
            The function called during writes to generate a figure. The
            function must take the simulation object as an argument, and return
            a matplotlib figure object (see notes). by default this argument
            is set to the exemplar function `summary_fig_func` contained in
            the `utils` module. Passing `None' to this argument  will turn off
            figures completely.
                                     
        n_threads: int, optional
            The number of CPU threads to use. By default jitpic will use the
            numba thread count, which is typically the maximum number of
            threads available. For intel CPUs with HT, this value can sometimes
            benefit from being set to half of the maximum, as numba cannot 
            distinguish between logical and physical cores.
            
        seed: int, optional
            Set the RNG seed (ensures reproducability between runs). If a
            varying random seed is desired, this value can be set dynamically
            at runtime. The system clock is a good source for varying seeds
            (Default: 0).
        
        resize_period: int, optional
            The interval between particle buffer resizing. This is only
            relevant for moving-window simulations, and can usually be left
            as-is. If the resize period is too short or too long, performance
            suffers. Resizing is turned off when this option is set to 0
            (Default: 100).
            
        sort_period: int, optional
            Set the interval between particle array sorting. This is a 
            debugging feature which gives no physics benefits nor performance
            gains, and should generally be left alone (Default: 0).
            
        boundaries: str, optional
            Set the boundary conditions; `open` or `periodic` (Default: `open`)
                                                               
        pusher: str, optional
            Set the particle pusher to use. Currently implemented are: `boris`,
            `vay`, 'cohen`. (Default: `cohen`)
        moving_window: bool, optional
            Set the moving window state. Cannot be used with periodic 
            boundaries (Default: False).
        
        diagdir: str, optional
            Set the directory in which to save diagnostics. If it does not
            exist, it will be created upon the first diagnostic write
            (Default: `diags`).
             
        imagedir: str, optional
            Set the directory in which to save images. If it does not exist, it
            will be created as the first image is saved
            (Default: `images`).
        
        Notes
        -----
        The plotting function use signature must be as follows:
            
            ```fig = plotting_function(sim)```
            
            Parameters
            ----------
            sim: Simulation
                Parent simulation object
            
            Returns
            -------
            fig: Figure
                Matplotlib Figure object
        
        Beyond this, the contents are entirely down to the user.
        """
        # set the RNG seed for reproducability
        self.seed = seed
        np.random.seed(seed)
        # register the number of threads, and set the numba variable accordingly
        self.n_threads = n_threads
        numba.set_num_threads(n_threads)
        # register the simulation time and iteration
        self.t = 0.
        self.iter = 0
        # register the particle shape factor and boundary conditions
        self.particle_shape = particle_shape
        self.boundaries = boundaries
        # initialise the simulation grid and register the timestep (dt == dx)
        self.grid = Simgrid(x0, x1, Nx, self.n_threads, self.boundaries, particle_shape=self.particle_shape)
        self.dt = self.grid.dx
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
        self.Nspecies = 0
        for spec in species:
            self.append_species(spec)
        # register the particle resize and sorting periods
        self.resize_period = resize_period
        self.sort_period = sort_period
        # register the diagnostic period and associated directories
        self.diag_period = diag_period 
        self.diagdir = diagdir
        self.imagedir = imagedir
        # register the plotting function
        self.inline_plotting_script = plotfunc
        # register no external fields initially
        self.external_fields = []
        # register lists for additional diagnostics
        self.tsdiag = []
        self.ptdiag = []
        #  and moving window state
        if moving_window:
            assert self.boundaries == 'open', 'Moving window must employ open boundaries'
        self.moving_window = moving_window
        # print some information about the simulation
        print('----------\nJitPIC %s\n----------'%__version__)
        print( 'Grid consists of %i cells with timestep %f'%(self.grid.Nx, self.dt) )
        print( 'Starting with %i pre-defined particle species'%len(self.species))
        print( 'Simulating %i particles of shape order %i using the %s pusher'%( sum([ spec.N for spec in self.species]), self.particle_shape, self.pusher) )
        print( 'Employing %s boundary conditions'%self.boundaries )
        print( 'Using %i threads via %s\n'%(numba.get_num_threads(), numba.threading_layer()) )
        return
    
    def step( self, N=1, silent=False ):
        """
        Advance the simulation by N cycles.
        
        Parameters
        ----------
        N: int, optional
            Number of steps to perform (Default: 1).
            
        silent: bool, optional
            Disable printing diagnostic information during the run
            (Default: `False`).
        """
        # print summary information
        if not silent:
            print( 'Starting from t = %f'%self.t )
            print( 'performing %i PIC cycles from from step %i\n'%(N, self.iter) )
            if self.moving_window:
                print('Moving window active')
        # timing information
        t0 = time.time()
        t1 = t0
        # p,v must be offset to n=-1/2 before the first step
        if self.iter == 0:
            self.apply_initial_offset_to_pv()
        # begin the loop
        for i in range(N):
            ## periodic diagnostic operations
            t1 = self.write_diagnostics(silent, t0, t1)
            ## every-step operations
            # gather timeseries data
            for tsdiag in self.tsdiag:
                tsdiag.gather_data()
            # introduce lasers from any antennas
            for laser in self.lasers:
                if laser.is_antenna:
                    self.inject_antenna_fields(laser)     
            ## main PIC cycle
            # E,B,x at n
            # J,p,v at n-1/2
            # interpolate fields onto particles
            self.gather_fields()
            # gather tracking data
            for ptdiag in self.ptdiag:
                ptdiag.gather_data()
            # push p,v to n+1/2, x to n+1
            self.push_particles( self.dt ) 
            self.reseat_particles()
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
                # reseat particles again after grid shift
                self.reseat_particles()
                # periodically clean up dead particles
                if self.resize_period > 0 and self.iter%self.resize_period == 0:
                    for spec in self.species:
                        spec.compact_particle_arrays()    
            # periodically sort particles
            if self.sort_period > 0 and self.iter%self.sort_period == 0:
                for spec in self.species:
                    spec.sort_particles()   
            # advance simulation time and iteration
            self.t += self.dt
            self.iter += 1
        if not silent:
            print( 'Finished in %.3f s'%(time.time()-t0))
            print( 'Now at t = %.3e\n'%self.t)
        return  

    def add_new_species(self, name, ppc=1, n=1, p0=-np.inf, p1=np.inf, 
                        m=1., q=-1., T=0., dens=None, add_tags=False,
                        p_x=0., p_y=0., p_z=0.):
        """ 
        Initialise a new species and add it to the simulation
        
        Wrapper function for initialising and adding a new particle species
        to the simulation.
        
        Parameters
        ----------
        name: str
            Species name for diagnostics
            
        ppc: int, optional
            Number of particles per cell (Default: 1).
            
        n: float (in critical densities), optional
            Normalised density (Default: 1).
            
        p0: float, optional
            Species start position (Default: -inf).
            
        p1: float, optional
            Species end position (Default: inf).
            
        m: float (in electron masses), optional
            Species mass (Default: 1).
            
        q: float (in elementary charges), optional
            Species charge (Default: -1).
            
        T: float (in electron volts), optional
            Species temperature (Default: 0).
        
        p_x, p_y, p_z: float, optional
            Particle flow momentum (Defualt: 0)
            
        dens: func or None, optional
            Function describing the density profile. When none is specified, a
            flat profile is assumed by default (Default: None)
            
        add_tags: bool, optional
            Add unique identifying tags to each particle for tracking purposes
            (Default: False).
        """
        new_species = Species( name, ppc, n, p0, p1, dens=dens, m=m, q=q, T=T, add_tags=add_tags )
        self.append_species( new_species ) 
        return
    
    def append_species(self, spec):
        """ Initialise and append a species object to the list of species. """
        spec.initialise_particles(self.grid)
        self.species.append(spec)
        self.Nspecies = len(self.species)
        return

    def deposit_rho(self):
        """ Deposit the total charge density on the grid. """
        grid = self.grid
        grid.rho_2D[:,:] = 0.
        
        for spec in self.species:
            # calculate the index splits for current deposition
            indices = np.fromiter([ i*spec.N//self.n_threads for i in range(self.n_threads)] + [spec.N], int, count=self.n_threads+1 )
            
            self.deposit_rho_func(self.n_threads, spec.x, grid.idx, 
                                  spec.q*spec.w, grid.rho_2D, 
                                  spec.l, spec.r, spec.state,
                                  indices, grid.x, grid.Nx)
            
        grid.rho[:] = grid.rho_2D.sum(axis=0)
        
        return grid.rho

    def deposit_single_species(self, spec):
        """ 
        Deposity the charge density for a single species on the grid.
        
        Parameters
        ----------
        spec: Species
            Species to deposit
        """    
        grid = self.grid
        grid.rho_2D[:,:] = 0.
        
        # calculate the index splits for charge deposition
        indices = np.fromiter([ i*spec.N//self.n_threads for i in range(self.n_threads)] + [spec.N], int, count=self.n_threads+1 )
        
        self.deposit_rho_func(self.n_threads, spec.x, grid.idx, 
                              spec.q*spec.w, grid.rho_2D, 
                              spec.l, spec.r, spec.state,
                              indices, grid.x, grid.Nx)
        
        grid.rho[:] = grid.rho_2D.sum(axis=0)
        
        return grid.rho
    
    def apply_initial_offset_to_pv(self):
        """ Apply the initial half-step backwards to particle momentum and velocity """
        self.gather_fields()
        self.push_particles( -self.dt*0.5 )
        # x has also been pushed to -1/2, but
        # x stays on integer timesteps, so revert the position
        for spec in self.species:
            spec.revert_x()
        return

    def set_moving_window(self, state):
        """
        Activate or deactivate the moving window.
        
        Parameters
        ----------
        state: bool
            Moving window on/off.
        """
        assert self.boundaries == 'open', 'Moving window must employ open boundaries'
        self.moving_window = state
        return
       
    def deposit_current(self):
        """
        deposit current onto the grid.
        
        Race conditions can occur in Numba's array-reductions, therefore
        this method determines the start and end indices for `n_thread` chunks, 
        then performs the current deposition for each thread individually. 
        Finally, the contributions from each thread are reduced in serial to
        avoid the race condition.
        """
        grid = self.grid
        # zero the current
        grid.J_3D[:,:,:] = 0.
        # loop over all species
        for spec in self.species:
            # calculate the index splits for current deposition 
            indices = np.fromiter([ i*spec.N//self.n_threads for i in range(self.n_threads)] + [spec.N], int, count=self.n_threads+1 )
            # deposit current
            self.deposit_J_func( spec.x, spec.x_old, spec.w, spec.v, grid.J_3D,
                          self.n_threads, indices, spec.q, grid.x,
                          spec.state, grid.idx, grid.x0*grid.idx)
        # reduce the current
        grid.J[:,:] = grid.J_3D.sum(axis=0)
        return

    def add_external_field(self, field, component, magnitude, function=None):
        """
        Add an external field to the simulation.
        
        External fields are calclulated and applied to all particles
        before they are pushed. 
        
        Parameters
        ----------
        field: str (`E` or `B`)
            The field to add to.
            
        component: int (0, 1 or 2)
            The field component index. 
            
        magnitude: float
            Magnitude of the field.
            
        function: func or None, optional
            Field profile, uniform if none specified (Default: None).
        """
        self.external_fields.append( External_field(field, component, magnitude,
                                                    function=function) )
        return
    
    def gather_fields(self):
        """ Interpolate grid fields onto particles. """
        # loop over all species
        for spec in self.species:
            # gather fields
            self.interpolate_func(self.grid.E,self.grid.B,
                              spec.E, spec.B,
                              spec.l, spec.r, 
                              (spec.x - self.grid.x0)*self.grid.idx,
                              spec.N, spec.state )
        return 
    
    def push_fields(self):
        """ Advance the fields in time using a numerical-dispersion free along x (NDFX) field solver. """
        grid = self.grid
        pidt = np.pi*self.dt
        # advance longitudinal E field
        grid.E[0,:grid.NEB] -= 2. * pidt * grid.J[0,:grid.NJ]
        # define intermediary P and S fields
        PR = (grid.E[1] + grid.B[2]) * .5
        PL = (grid.E[1] - grid.B[2]) * .5
        SR = (grid.E[2] - grid.B[1]) * .5
        SL = (grid.E[2] + grid.B[1]) * .5
        # advance the fields
        PR[grid.f_shift] = PR[:grid.NEB] - pidt * grid.J[1,:grid.NJ]
        PL[:grid.NEB] = PL[grid.f_shift] - pidt * grid.J[1,:grid.NJ]
        SR[grid.f_shift] = SR[:grid.NEB] - pidt * grid.J[2,:grid.NJ]
        SL[:grid.NEB] = SL[grid.f_shift] - pidt * grid.J[2,:grid.NJ]
        # convert back to E and B fields
        grid.E[1,:grid.NEB] = PR[:grid.NEB] + PL[:grid.NEB]
        grid.B[2,:grid.NEB] = PR[:grid.NEB] - PL[:grid.NEB]
        grid.E[2,:grid.NEB] = SL[:grid.NEB] + SR[:grid.NEB]
        grid.B[1,:grid.NEB] = SL[:grid.NEB] - SR[:grid.NEB]
        return  

    def push_particles(self, dt):
        """
        Push the particle momentum and position in time
        
        Advance the particle momenta and positions in time using the chosen 
        particle pusher.
        
        Parameters
        ----------
        dt: float
            Timestep. This will almost always be the usual simulation dt,
            except before the first step, when a half-backstep is required to 
            offset p and v. Particle positions must then be manually reverted.
        """ 
        # loop over all species
        for spec in self.species:
            x = spec.x
            E = spec.E
            B = spec.B
            # apply any external fields
            if len(self.external_fields) > 0:
                for ext_field in self.external_fields:
                    if ext_field.field == 'E':   
                        E = ext_field.add_field(x, self.t, E)
                    else:
                        B = ext_field.add_field(x, self.t, B)
            # push particles
            self.particle_push_func( E, B, dt*self.pushconst*spec.qm, spec.p, spec.v, x, spec.x_old, 
                                    spec.rg, spec.m, dt, spec.N, spec.state) 
        return

    def reseat_particles(self):
        """ Reseat particles in new cells, disable any particles that have been pushed off the grid. """ 
        for spec in self.species:
            self.reseat_func(spec.N, spec.x, spec.state, spec.l, spec.r,
                         self.grid.x0, self.grid.x1, self.grid.dx, self.grid.idx, self.grid.Nx)
            spec.N_alive = spec.state.sum()
        return
        
    def add_laser(self, a0, x0, ctau, lambda_0=1., p=0, d=1, theta_pol=0., 
                  cep=0., clip=None, method='box', x_antenna=None, 
                  t_stop=np.inf):
        """
        Register and initialise a laser into the simulation.
        
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
        """ Inject laser fields into one grid cell. """
        if laser.is_antenna and self.t < laser.t_stop:
            E_laser, B_laser = laser.antenna_fields( self.grid.x[laser.antenna_index], self.t )
                        
            self.grid.E[:,laser.antenna_index] += E_laser
            self.grid.B[:,laser.antenna_index] += B_laser

        return
    
    def write_diagnostics(self, silent, t0, t1):
        """ Deal with and write various diagnostics. """
        if self.diag_period > 0 and self.iter%self.diag_period == 0:
            if not silent:
                print('Writing diagnostics at step %i (t = %.1f)'%(self.iter, self.t))
                print('%.1f (%.1f) seconds elapsed (since last write)\n'%(time.time()-t0, time.time()-t1) )
            # register split time and write diagnostics
            t1 = time.time()   
            self.write_grid_diagnostics()
            # generate figure
            if self.inline_plotting_script is not None:
                self.write_figure()  
            # write timeseries diagnostics
            for tsdiag in self.tsdiag:
                if self.iter > tsdiag.istart:
                    tsdiag.write_data()
            # write tracking diagnostics
            for ptdiag in self.ptdiag:
                if self.iter > ptdiag.istart:
                    ptdiag.write_data()
        return t1
    
    def write_grid_diagnostics(self):
        """ Write all grid quantities to file. """
        make_directory(self.diagdir)
        fname = '%s/diags-%.8i.h5'%(self.diagdir, self.iter)
        if os.path.exists(fname):
            check_for_file(fname)
        # write the new file
        with h5py.File(fname, 'w') as f:
            # create attributes
            f.attrs['iter'] = self.iter
            f.attrs['time'] = self.t
            f.attrs['dt'] = self.dt
            # get fields to write
            E = self.grid.get_field('E')
            B = self.grid.get_field('B')
            J = self.grid.get_field('J')
            # create the datasets
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
        
    def write_figure(self, index=None, dpi=220):
        """
        Plot and save a summary image.
        
        Parameters
        ----------
        index : int or None, optional
            Specify a numerical index for the image, otherwise the current
            simulation iteration is used (Default: None)
            
        dpi: int, optional
            Specify the image DPI (Default: 220).
        """
        # make the image directory if it doesn't already exist
        make_directory(self.imagedir)
        # set the file name, check
        if index is not None:
            fname = '%s/%.8i.png'%(self.imagedir, index)           
        else:
            fname = '%s/%.8i.png'%(self.imagedir, self.iter)
        check_for_file(fname)
        # generate a figure
        fig = self.inline_plotting_script( self )
        # save
        fig.savefig(fname, dpi=dpi)
        # clean up
        plt.close(fig) 
        gc.collect()
        return
    
    def add_timeseries_diagnostic( self, fields, pos, method='position', fname='timeseries' ):
        """
        Add a timeseries diagnostic to the simulation.
        
        Automatically collect the specified field information from either a
        list of positions or a list of cells at every timestep.
        
        Parameters
        ----------
        fields: list 
            List of fields to track from any combination of `E`,`B` and `J`.
            
        pos: list
            List of positions or cell indices to track.
            
        method: str (`position` or `cell`), optional
            The method to use.
            - `position` tracks based on position, interpolated from the grid
            - `cell` tracks the field values in the specified cells
            (Default: `position`)
 
        fname: str, optional
            Timeseries file name (Default: `timeseries`).
        """
        path = '%s/%s.h5'%(self.diagdir, fname)
        # if a file already exists, throw an error or sliently remove it
        if len(self.tsdiag) == 0:
            check_for_file(path)
        # append the diagnostic
        self.tsdiag.append( Timeseries_diagnostic(self, fields, pos,
                                            method=method,
                                            fname=fname))
        return
        
    def add_tracking_diagnostic( self, species, tags,
                                track_x=True, track_p=True,
                                track_E=False, track_B=False,
                                fname='tracking'):
        """
        Add a particle tracking diagnostic to the simulation
        
        This will automatically collect and record particle data based on the
        particle ID. This requires tags to be turned on for the relevant species.
        
        Parameters
        ----------
        species: Species
            Species object to pull from.
            
        tags: list
            List of tags to track.
            
        track_x: bool
            Track particle positions (Default: True)

        track_p: bool
            Track particle momentum (Default: True)
            
        track_E: bool
            Track particle B-field (Default: False)
            
        track_B: bool
            Track particle E-field (Default: False)            

        fname: str
            Tracking file name (Default: `tracking`)
        """
        path = '%s/%s.h5'%(self.diagdir, fname)
        # if a file already exists, throw an error or sliently remove it
        if len(self.ptdiag) == 0: 
            check_for_file(path)
        # append the diagnostic
        self.ptdiag.append( Particle_tracking_diagnostic(self, species, tags, 
                                                         track_x=track_x, track_p=track_p,
                                                         track_E=track_E, track_B=track_B,
                                                         fname=fname) )
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
        F: array
            The specified field
        """
        return self.grid.get_field(field)
            
            