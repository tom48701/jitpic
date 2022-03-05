from ..config import allow_overwrite
 
import os
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def check_for_file(path):
    """ Check for an existing file, remove if allowed"""
    if os.path.exists(path) and os.path.isfile(path):
        if allow_overwrite:
            os.remove(path)
        else:
            raise OSError("The file %s already exists at the target location! "
                          "Either specify a different filename or move/remove/rename the existing file. "
                          "This error has been raised because the 'JITPIC_FORBID_OVERWRITE' environment variable is set"%path)    
            
def make_directory( dirpath, cwd=os.getcwd() ):

    cwd = Path(cwd)
    dirpath = Path( cwd/dirpath )
        
    if not dirpath.exists():
        os.mkdir(dirpath)    
    elif not dirpath.is_dir():         
        print('eponymous file in place of requested directory, cannot save')
                
    return



def fft_data(x,y):
    N = len(x)
    T = (x[-1] - x[0]) /N
    xf = np.linspace(0.0, 1.0/(2.0*T), N//2)
    
    yf = np.fft.fft(y)
    yf = 2.0/len(xf) * abs(yf[:len(xf)])
    
    return xf, yf

def summary_fig_func( sim, fontsize=8 ):
        """
        Plot a figure using information from the current simulation state
        
        sim      : simulation : the simulation object
        fontsize : int        : figure fontsize
        """
        # define shortcuts for the quantities to be used
        
        # always use the get methods for the grid fields
        grid = sim.grid
        x = grid.x
        E = sim.get_field('E') 
        J = sim.get_field('J')
        S = sim.get_field('S')
        # always use the get methods for the particle quantities
        # xp = sim.species[0].get_x()
        # gamma = sim.species[0].get_gamma()
        # etc...
        
        # normalise the fields to the laser (if there is one)
        try:
            a0 = sim.lasers[0].a0
        except IndexError:
            a0 = 1.
            
        Sx = S[0] # forward Poynting vector
    
        fig, ax = plt.subplots(figsize=(6,4)) # initialise the figure
        
        if len(sim.species) > 0: # get the density of the first species
            ne = abs(sim.deposit_single_species(sim.species[0]))

            # plot it
            ax.plot(x, ne/sim.species[0].n, 
                c='0.5', label=r'$n_e$', lw=1)   
 
        # Ey (first transverse electric field component)
        ax.plot(x, E[1]/a0, 
            'k', label='$E_y$', lw=1)

        # Jx (longitudinal current component)
        ax.plot(x, J[0] - 0.5, 
            'g', label='$J_x$', lw=1)
        
        # Ex (longitudinal electric field component)
        ax.plot(x, E[0] - 1., 
            'b', label='$E_x$', lw=1)

        # Poynting vector (light intensity and direction)
        ax.plot(x, Sx/a0**2, 
            'r', label='$\sqrt{S_x}$', lw=1, alpha=0.5)
        
        # label the simulation time
        ax.text(.1,.1, r'$t=%.3f \tau_0$'%sim.t, transform=ax.transAxes)
        
        # calculate the total EM energy density on the grid
        Eem = grid.get_field('u').sum() 
        
        # calculate the total kinetic energy density of the particles
        Ek = 0.
        for spec in sim.species:
            Ek += spec.get_u().sum() * spec.m
        
        # show the energy densities (total should not increase over the simulation) ((much)) 
        ax.text(.1,.9, r'$U_{\mathrm{EM}}=%.3e $'%Eem, transform=ax.transAxes)
        ax.text(.1,.8, r'$U_{\mathrm{K}}=%.3e $'%Ek,  transform=ax.transAxes)
        ax.text(.6,.9, r'$U_{\mathrm{EM}}+U_{\mathrm{K}}=%.3e $'%(Ek+Eem), transform=ax.transAxes)
        
        # set some figure parameters
        ax.set_xlim(grid.x0, grid.x1)
        ax.set_ylim(-2,2)
        ax.tick_params(left=False, labelleft=False)
        ax.legend(loc='lower right', fontsize=fontsize)
        ax.set_xlabel('$x/\lambda_0$', fontsize=fontsize)
        fig.tight_layout()
        
        # return the figure object to be saved
        return fig