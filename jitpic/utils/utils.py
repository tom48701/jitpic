import os
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def make_directory( dirpath, cwd=os.getcwd() ):

    cwd = Path(cwd)
    dirpath = Path( cwd/dirpath )
        
    if not dirpath.exists():
        os.mkdir(dirpath)    
    elif not dirpath.is_dir():         
        print('eponymous file in place of requested directory, cannot save')
                
    return

def default_inline_plotting_script( sim, fontsize=8 ):
        """
        Plot a figure using information from the current simulation state
        
        sim      : simulation : the simulation object
        fontsize : int        : figure fontsize
        """
        # define shortcuts for the quantities to be used
        x = sim.grid.x
        E = sim.grid.get_field('E') 
        B = sim.grid.get_field('B')
        J = sim.grid.get_field('J')
        
        try:
            a0 = sim.lasers[0].a0
        except IndexError:
            a0 = 1.
            
        Sx =  E[1]*B[2] - B[1]*E[2] # forward Poynting vector
    
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

        # sqrt(Sx) to retrieve the overall laser amplitude
        ax.plot(x, np.sqrt(abs(Sx))*np.sign(Sx)/a0, 
            'r', label='$a$', lw=1)
        
        # label the simulation time
        ax.text(.1,.1, r'$t=%.3f \tau_0$'%sim.t, transform=ax.transAxes)
        
        # calculate the total EM energy on the grid
        Eem = (( (E**2).sum(axis=0) + (B**2).sum(axis=0) )  ).sum() * sim.grid.dx / 2.
        
        # calculate the total kinetic energy in the particles
        Ek = 0.
        for spec in sim.species:
            Ek += ( spec.KE() * sim.grid.dx ).sum() 
        
        # show the total energy (should not increase over the simulation) ((much)) 
        ax.text(.1,.9, r'$\mathcal{E}_{\mathrm{EM}}=%.3e $'%Eem, transform=ax.transAxes)
        ax.text(.1,.8, r'$\mathcal{E}_{\mathrm{K}}=%.3e $'%Ek,  transform=ax.transAxes)
        ax.text(.6,.9, r'$\mathcal{E}_{\mathrm{EM}}+\mathcal{E}_{\mathrm{K}}=%.3e $'%(Ek+Eem), transform=ax.transAxes)
        
        # set some figure parameters
        ax.set_xlim(sim.grid.x0, sim.grid.x1)
        ax.set_ylim(-2,2)
        ax.legend(loc='lower right', fontsize=fontsize)
        ax.set_xlabel('$x/\lambda_0$', fontsize=fontsize)
        fig.tight_layout()
        
        return fig