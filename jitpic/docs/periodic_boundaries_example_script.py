"""
This is an example script for periodic boundaries in JitPIC,
we will try to demonstrate the two-stream instability, and define a custom
plotting script to show the phase-space vortices.
"""

import time
import numpy as np

from jitpic.main import Simulation
from jitpic.particles import Species

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

###################### Simulation Parameters #######################
    
x0 = -2  # box start position
x1 = 2  # box end position
res = 32   # resolution per laser wavelength
Nx = np.rint( (x1-x0)*res ).astype(int)   # automatically determine the total mumber of cells

n_e = .005 # plasma density 
ppc = 200   # plasma particles per cell (more = less noise)

# set the flow
p_flow = 0.1

def plotting_script( sim, fontsize=8 ):
        """
        Plot a figure using information from the current simulation state
        
        sim      : simulation : the simulation object
        fontsize : int        : figure fontsize
        """

        # get the particle positions and momenta as two lists
        x = sim.species[0].x
        p = sim.species[0].p[0]
        
        # append any other species to these lists
        for spec in sim.species[1:]:
            x = np.append( x, spec.x )
            p = np.append( p, spec.p[0] )
            
        # create the figure
        fig, ax = plt.subplots(figsize=(6,4), constrained_layout=True) 
        
        # choose the colourmap
        cmap = 'magma'

        # figure limit
        vmax = 0.5
        
        ax.hist2d(x, p, range=((sim.grid.x0,sim.grid.x1),(-vmax,vmax)),
                  bins=(sim.grid.Nx, 500), cmap=cmap)
        # labels
        ax.set_xlabel('$x$', fontsize=fontsize)
        ax.set_ylabel('$p_x$', fontsize=fontsize)
        # limits
        ax.set_ylim(-vmax, vmax)
        ax.set_xlim(sim.grid.x0, sim.grid.x1)
        
        # add some text
        ax.text( 0.05, 0.9, '$t=%.2f$'%sim.t, transform=ax.transAxes, c='w', fontsize=fontsize)
        
        # create an inset plot
        ax1 = inset_axes(ax, width='40%', height='20%', loc='upper right' )
        
        # make all of it white with a transparent background so it shows up
        ax1.set_facecolor( 'none' ) 
        ax1.spines['bottom'].set_color('w')
        ax1.spines['top'].set_color('w') 
        ax1.spines['right'].set_color('w')
        ax1.spines['left'].set_color('w')
        ax1.tick_params(axis='x', which='both', colors='w', labelsize=6)
        ax1.tick_params(axis='y', which='both', colors='w', labelsize=6)
        ax1.yaxis.label.set_color('w')
        ax1.xaxis.label.set_color('w')
        # axis labels
        ax1.set_xlabel('$x$', fontsize=6)
        ax1.set_ylabel(r'$\rho$', fontsize=6)
        
        # redefine x for the inset plot
        x = sim.grid.x
        # deposit the charge densities
        r1 = np.array(sim.deposit_single_species( sim.species[0] ))
        r2 = np.array(sim.deposit_single_species( sim.species[1] ))
        
        # plot!
        ax1.plot( x, abs(r1)/sim.species[0].n, 'r' )
        ax1.plot( x, abs(r2)/sim.species[1].n, 'b' )
        ax1.plot( x, (abs(r2)+abs(r1))/sim.species[0].n, 'w' )
        # inset figure limits
        ax1.set_xlim(sim.grid.x0, sim.grid.x1)
        ax1.set_ylim(0,5)

        return fig
    
################## Simulation Initialisation #################

# initialise the particle species 
elec1 = Species('elec+', ppc, n_e, eV=100, p_x=-p_flow )
elec2 = Species('elec-', ppc, n_e, eV=100, p_x=p_flow, )

# initialise simulation object
sim = Simulation( x0, x1, Nx, species=[elec1, elec2], diag_period=20*res,
                 boundaries='periodic',
                 plotfunc=plotting_script)

##################### Simulation Execution ######################
# make a note of the start time
t0 = time.time()

sim.step( N = 100*res + 1 ) # run the simulation
  
t1 = time.time()
print( 'All done in %.1f s'%(t1-t0))
