"""
This is an example script for using the moving window in JitPIC,
we will try to demonstrate leading-edge erosion as in: doi.org/10.1063/1.872001
"""

import time
import numpy as np

from jitpic.main import Simulation
from jitpic.particles import Species
    
###################### Simulation Parameters #######################
    
x0 = -80.  # box start position
x1 = 0.  # box end position
res = 16 # resolution per laser wavelength
Nx = np.rint( (x1-x0)*res ).astype(int)   # total mumber of cells
   
a0 = 10. # laser amplitude
centroid = -40. # laser centroid position
ctau = 12. # laser duration (see laser class for the exact definition)

n_e = 0.04 # plasma reference density 

ppc = 10 # plasma particles per cell (more = less noise)
p0 = 0. # plasma start position
p1 = 1000. # plasma end position

################## Simulation Initialisation #################

# initialise the particle species 

elec = Species('elec', ppc, n_e, p0, p1, m=1., q=-1., T=0.)

# initialise simulation object
sim = Simulation( x0, x1, Nx, species=[elec], diag_period=50*res )

# add the laser
sim.add_laser(a0, centroid, ctau )

# set the moving window
sim.set_moving_window(True)
  
##################### Simulation Execution ######################
#make a note of the start time
t0 = time.time()

sim.step( N = 500*res + 1 ) # run the simulation
  
t1 = time.time()
print( 'All done in %.3f s'%(t1-t0))


