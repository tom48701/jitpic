import numba, time, h5py, gc
import numpy as np

from .utils import make_directory
from .grid import simgrid
from .particles import species
from .laser import laser
from .diagnostics import timeseries_diagnostic
from .numba_functions import deposit_numba, interpolate_numba, \
                                boris_numba, reseat_numba

#import matplotlib
#matplotlib.use('Agg') # use a non-graphical backend

import matplotlib.pyplot as plt 
#plt.style.use('classic')  # revert to 1.x style

class simulation:
    """
    The main simulation object, containing all the PIC methods, related
    functions and diagnostics.
    """
    def __init__(self, x0, x1, Nx, species=[], diag_period=0, 
                 n_threads=numba.get_num_threads(), seed=0  ):
        """ 
        Initialise the simulation, set up the grid at the same time
        
        x0          : float  : grid start point
        x1          : float  : grid end point
        Nx          : int    : number of cells
        species     : list   : a list of `species' objects
        diag_period : int    : number of steps between diagnostic writes
        n_threads   : int    : number of CPU threads to use
        seed        : int    : set the RNG seed
        """
        # set the RNG seed for reproducability
        np.random.seed(seed)
        
        self.n_threads = n_threads
        numba.set_num_threads(n_threads)
        
        self.t = 0.
        self.iter = 0
        self.lasers = []

        self.grid = simgrid(x0, x1, Nx, self.n_threads)
    
        self.species = []
        self.Nspecies = len(species)
        
        if self.Nspecies > 0:  
            for i in range(self.Nspecies):
                spec = species[i]

                self.append_species(spec)    
        
        self.diag_period = diag_period 
        
        self.dt = self.grid.dx

        self.E_ext = np.zeros((3,1))
        self.B_ext = np.zeros((3,1))
        
        self.tsdiag = None
        
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
                         (optional)
        """
        
        new_species = species( name, ppc, n, p0, p1, dens=dens, m=m, q=q, eV=eV )
        
        self.append_species( new_species )
        
        return
    
    def append_species(self, spec):
        """
        Initialise and append a species object to the list of species
        """
    
        spec.initialise_particles(self.grid, self.n_threads)
        
        self.species.append(spec)
        self.Nspecies = len(self.species)
        
        return

    def deposit_rho(self):
        """ 
        Deposit the total charge density on the grid.
        For diagnostics only!
        
        The deposition is periodic for simplicity, expect small errors 
        near the box edges. Might fix later...
        """
        
        rho = np.zeros(self.grid.Nx)

        
        for spec in self.species: 
            state = spec.state
            l = spec.l[state]
            r = spec.r[state]
            w = spec.w[state]
            x = spec.x[state]
        
            for i in range(spec.N_alive):
                delta = x[i]%self.grid.dx/self.grid.dx
                rho[l[i]] += spec.q * w[i] * (1.-delta)
                rho[r[i]] += spec.q * w[i] * delta  
                
        return rho
    
    def deposit_single_species(self, spec):
        """ 
        Deposity the charge density for a single species on the grid.
        For diagnostics only!

        The deposition is periodic for simplicity, expect small errors 
        near the box edges. Might fix later...
        """
     
        rho = np.zeros(self.grid.Nx)
        state = spec.state
        l = spec.l[state]
        r = spec.r[state]
        w = spec.w[state]
        x = spec.x[state]
        
        for i in range(spec.N_alive):
            delta = x[i]%self.grid.dx/self.grid.dx
            rho[l[i]] += spec.q * w[i] * (1.-delta)
            rho[r[i]] += spec.q * w[i] * delta  
        
        return rho
       
    def deposit_current(self, backstep=False):
        """
        deposit current onto the grid.
        Race conditions can occur in Numba's array-reductions!
        This method determines the start and end indices for  n_thread chunks, 
        then performs the current deposition for each thread individually. 
        Finally, the contributions from each thread are reduced in serial to
        avoid the race condition.
        
        nthreads : int   : number of threads
        backstep : bool  : reverse the sign of the current deposited 
                           (probably will only used during initial setup 
                            to offset the current to -1/2)
        """
        
        J = self.grid.J
        J_3D = self.grid.J_3D
        
        J_3D[:,:,:] = 0.
        
        dx = self.grid.dx
        xidx = self.grid.x / dx
        x2 = self.grid.x2 / dx
            
        for spec in self.species:
            
            state = spec.state
            
            # calculate the index splits for current deposition - assume masked arrays
            indices = [ i*spec.N_alive//self.n_threads for i in range(self.n_threads)]+[spec.N_alive-1] 
            indices = np.array(indices)
            
            # mask the arrays so lengths match
            xs = spec.x[state] / dx
            x_olds = spec.x_old[state] / dx
            ws = spec.w[state]
            vys = spec.v[1][state]
            vzs = spec.v[2][state]
            l_olds = np.floor((x_olds-self.grid.x0/dx)).astype(int)
            
            deposit_numba( xs, x_olds, ws, vys, vzs, l_olds, J_3D,
                          self.n_threads, indices, spec.q, xidx, x2 )

        if backstep:
            J_3D *= -1.
        
        # reduce the current onto the grid
        J[:,:] = J_3D.sum(axis=0)
        
        return
    
    def step( self, N=1 ):
        """
        advance the simulation, automatically write diagnostics and images
        
        N : int : number of steps to perform (optional)
        """
        
        print('\nJitPIC:')
        print( 'performing %i PIC cycles from from step %i'%(N, self.iter) )
        print( 'Using %i threads via %s'%(numba.get_num_threads(), numba.threading_layer()) )
        print( 't = %f, dt = %f'%(self.t, self.dt) )
        print( 'Nx = %i, Np = %i\n'%( self.grid.Nx, sum([ spec.N for spec in self.species])) )

        t0 = time.time()
        t1 = t0
        
        for i in range(N):
            
            if self.iter == 0:
                # offset p,v,J to -1/2
                # (backstepping the particles will not advance x)
                self.apply_fields_to_particles()
                self.push_particles( backstep=True )
                self.deposit_current( backstep=True )
                
            if self.diag_period > 0 and self.iter%self.diag_period == 0:
                print('Writing diagnostics at step %i (t = %.1f)'%(self.iter, self.t))
                print('%.1f (%.1f) seconds elapsed (since last write)\n'%(time.time()-t0, time.time()-t1) )

                t1 = time.time()
                
                self.write_diagnostics()
                self.plot_result()  
                
                if self.tsdiag is not None and self.iter > 0:
                    self.tsdiag.write_data(self)

            if self.tsdiag is not None:
                    self.tsdiag.gather_data(self)
                    
            #print( 'Now at step %i'%self.iter)

            #E,B,x at i
            #J,p,v at i-1/2
            self.apply_fields_to_particles()
            self.push_particles() # push p,v to +1/2. x to +1
            self.deposit_current()  # push J to +1/2 
            self.push_fields() #push E,B to +1

            self.t += self.dt
            self.iter += 1

        print( 'Finished in %.3f s'%(time.time()-t0))
        print( 'Now at t = %.3e\n'%self.t)
  
        return  

    def add_external_field(self, fld, i, mag):
        """
        Add an constant external field to the simulation
        
        fld ('E','B') : str   : the type of field to add
        i   (0,1,2)   : int   : field component index 
        mag           : float : magnitude of the field
        """
        
        if fld == 'E':
            self.E_ext[i] = mag
        elif fld == 'B':
            self.B_ext[i] = mag
        else:
            print( 'Unrecognised field.')
        
        return

    def apply_fields_to_particles(self):
        """
        interpolate grid fields onto particles.
        """
              
        for spec in self.species:
            
            interpolate_numba(self.grid.E,self.grid.B,
                              spec.E, spec.B,
                              spec.l, spec.r, spec.x, 
                              self.grid.x0, self.grid.dx, spec.N)
    
        return 
    
    def push_fields(self):
        """
        advance the fields in time using a numerical-dispersion free along x 
        (NDFX) field solver, this requires that dx == dt.
        """
        
        pidt = np.pi*self.dt
        
        self.grid.E[0,:-1] -= 2. * pidt * self.grid.J[0,:-4]
        
        PR = (self.grid.E[1] + self.grid.B[2]) * .5
        PL = (self.grid.E[1] - self.grid.B[2]) * .5
        SR = (self.grid.E[2] - self.grid.B[1]) * .5
        SL = (self.grid.E[2] + self.grid.B[1]) * .5
        
        #shift = np.roll(np.arange(self.grid.Nx), -1)
        
        PR[self.grid.f_shift] = PR[:-1] - pidt * self.grid.J[1,:-4]
        PL[:-1] = PL[self.grid.f_shift] - pidt * self.grid.J[1,:-4]
        SR[self.grid.f_shift] = SR[:-1] - pidt * self.grid.J[2,:-4]
        SL[:-1] = SL[self.grid.f_shift] - pidt * self.grid.J[2,:-4]
        
        self.grid.E[1] = PR + PL
        self.grid.B[2] = PR - PL
        self.grid.E[2] = SL + SR
        self.grid.B[1] = SL - SR
        
        return  

    def push_particles(self, backstep=False):
        """
        Advance the particle momenta and positions in time using the Boris 
        particle pusher, then check for and disable any particles that
        have been pushed off the grid.
        
        All particles are pushed, dead particles do not matter for this part.
        
        backstep : bool : perform a half-step backwards push for the momentum 
                          and velocity (only used during initial setup 
                          to offset p and v to -1/2)
        """
        
        if backstep:
            dt = -self.dt/2.
        else:
            dt = self.dt
            
        for spec in self.species:

            qmdt2 = np.pi * spec.qm * dt
            
            # apply external fields
            E = spec.E + self.E_ext
            B = spec.B + self.B_ext
            
            boris_numba( E, B, qmdt2, spec.p, spec.v, spec.x, spec.x_old, 
                        spec.rg, spec.m, dt, spec.N, backstep=backstep)
                
        self.reseat_particles()
                
        return

    def reseat_particles(self):
        """
        Look for and disable any particles that get pushed off the grid.
        """
        
        for spec in self.species:

            reseat_numba(spec.N,spec.x,spec.state,spec.l,spec.r,
                         self.grid.x0,self.grid.x1,self.grid.dx,self.grid.Nx)
            
            spec.N_alive = spec.state.sum()
            
        return
        
    def add_laser(self, a0, x0, tau, lambda_0=1., p=0, d=1, theta_pol=0., 
                  cep=0., clip=None):
        """
        Add a laser to the simulation object.
        
        a0         : float      : normalised amplitude
        x0         : float      : laser centroid
        tau        : float      : pulse duration (of the form exp((t/R)^2))
        lambda_0   : float      : normalised wavelength 
        p (1,0,-1) : int        : polarisation type (LCP, LP, RCP)
        d (1,-1)   : int        : propagation direction forwards/backwards
        theta_pol  : float      : polarisation angle (0 - pi)
        cep        : float      : carrier envelope phase offset (0 - 2pi)
        clip       : float/None : Set the distance from x0 at which to 
                                  set the laser fields to 0
        """
        
        new_laser = laser(a0, lambda_0=lambda_0, p=p, x0=x0, tau=tau, d=d, 
                          theta_pol=theta_pol, clip=clip)
        
        self.lasers.append(new_laser)
        
        E_laser, B_laser = new_laser.fields(self.grid)
        
        self.grid.E += E_laser
        self.grid.B += B_laser
        
        return
    
    def write_diagnostics(self, diagdir='diags'):
        """ 
        Write all grid quantities to file 
        
        diagdir : str : folder to write diagnostics to
        """

        make_directory(diagdir)
            
        fname = '%s/diags-%.8i.h5'%(diagdir, self.iter)
        
        with h5py.File(fname, 'w') as f:
            
            f.attrs['iter'] = self.iter
            f.attrs['time'] = self.t
            f.attrs['dt'] = self.dt
            
            f.create_dataset('x', data=self.grid.x)
            f.create_dataset('Ex', data=self.grid.E[0,:-1] )
            f.create_dataset('Ey', data=self.grid.E[1,:-1] )
            f.create_dataset('Ez', data=self.grid.E[2,:-1] )
            f.create_dataset('Bx', data=self.grid.B[0,:-1] )
            f.create_dataset('By', data=self.grid.B[1,:-1] )
            f.create_dataset('Bz', data=self.grid.B[2,:-1] )
            f.create_dataset('Jx', data=self.grid.J[0,:-4] )
            f.create_dataset('Jy', data=self.grid.J[1,:-4] )
            f.create_dataset('Jz', data=self.grid.J[2,:-4] )    
            f.create_dataset('rho', data=self.deposit_rho())
            
            for spec in self.species:
                f.create_dataset(spec.name, data=self.deposit_single_species(spec) )
   
        return
        
    def plot_result(self, index=None, dpi=220, fontsize=8, lambda0=8e-7, imagedir='images'):
        """
        Plot and save summary images.
        
        This method can be modified AS MUCH AS NEEDED to produce 
        whatever figures are required. It has no bearing on the simulation
        itself and is here purely for your convenience.
        
        index : int      : specify a numerical index for the image, otherwise
                           the current simulation iteration is used
        dpi      : int   : image DPI, to quickly scale the images up or down in size
        fontsize : int   : reference fontsize to be used in the images
        lambda0  : float : real wavelength to be used for any conversions
        imagedir : str   : folder in which to save the images
        """
        
        # make the image directory if it doesn't already exist
        make_directory(imagedir)
        
            
        E = self.grid.E[:,:-1] # get the grid fields, without the extra cells 
        B = self.grid.B[:,:-1]
        J = self.grid.J[:,:-4]
        
        a0 = self.lasers[0].a0 # reference a0 for normalising the plots
        
        s1 =  E[1]*B[2] - B[1]*E[2] # forward Poynting vector
    
        fig, ax = plt.subplots(figsize=(6,4)) # initialise the figure
        
        if len(self.species) > 0: # get the density of the first species
            ne = abs(self.deposit_single_species(self.species[0]))

            # plot it
            ax.plot(self.grid.x, ne/self.species[0].n, 
                c='0.5', label=r'$n_e$', lw=1)   
 
        # Ey (first transverse electric field component)
        ax.plot(self.grid.x, E[1]/a0, 
            'k', label='$E_y$', lw=1)

        # Ez (second transverse electric field component)
        ax.plot(self.grid.x, 50*J[0] - 0.5, 
            'g', label='$J_x$', lw=1)
        
        # Ex (longitudinal electric field component)
        ax.plot(self.grid.x, 10*E[0] - 1., 
            'b', label='$E_x$', lw=1)

        # sqrt(s1) to retrieve the overall laser amplitude
        ax.plot(self.grid.x, np.sqrt(abs(s1))*np.sign(s1)/a0, 
            'r', label='$a$', lw=1)
        
        # label the simulation time
        ax.text(.1,.1, r'$t=%.3f \tau_0$'%self.t, transform=ax.transAxes)
        
        # calculate the total EM energy on the grid
        Eem = (( (self.grid.E**2).sum(axis=0) + (self.grid.B**2).sum(axis=0) )  ).sum() * self.grid.dx / 2.
        
        # calculate the total kinetic energy in the particles
        Ek = 0.
        for spec in self.species:
            Ek += ( spec.KE() * self.grid.dx ).sum() 
        
        # show the total energy (should not increase over the simulation) ((much)) 
        ax.text(.1,.9, r'$\mathcal{E}_{\mathrm{EM}}=%.3e $'%Eem, transform=ax.transAxes)
        ax.text(.1,.8, r'$\mathcal{E}_{\mathrm{K}}=%.3e $'%Ek,  transform=ax.transAxes)
        ax.text(.6,.9, r'$\mathcal{E}_{\mathrm{EM}}+\mathcal{E}_{\mathrm{K}}=%.3e $'%(Ek+Eem), transform=ax.transAxes)
        
        # set some figure parameters
        ax.set_xlim(self.grid.x0, self.grid.x1)
        ax.set_ylim(-2,2)
        ax.legend(loc='lower right', fontsize=fontsize)
        ax.set_xlabel('$x/\lambda_0$', fontsize=fontsize)
        fig.tight_layout()
        
        # set the file name if specified
        if index is not None:
            plt.savefig('%s/%.8i.png'%(imagedir, index), dpi=dpi)            
        else:
            plt.savefig('%s/%.8i.png'%(imagedir, self.iter), dpi=dpi)
        
        # clean up
        plt.close(fig) 
        gc.collect()
        return
    
    def add_timeseries_diagnostic( self, fields, cells, 
                                  diagdir='diags', fname='timeseries_data'):
        """
        Add a timeseries diagnostic to the simulation
        
        This will automatically collect the specified field information from
        the specified cells at every timestep, and continually append the
        timeseries file throughout the simulation.

        fields  : list of str : list of fields to track ('E','B','J')
        cells   : list of int : list of cells to track [0 : Nx-1]
        diagdir : str         : folder to write to
        fname   : str         : output file name
        """
        
        self.tsdiag = timeseries_diagnostic(self, fields, cells,
                                            buffer_size=self.diag_period,
                                            diagdir=diagdir, fname=fname)        