import time, os
import numpy as np
from scipy.constants import m_e, m_p

from jitpic.main import Simulation
from jitpic.particles import Species


###################### Simulation Parameters #######################
    
x0 = -50.  # box start position
x1 =   0.  # box end position
res =  16   # resolution per laser wavelength
Nx = np.rint( (x1-x0)*res ).astype(int)   # automatically determine the total mumber of cells
   
a0 = 2.       # laser amplitude
p = 0           # laser polarisation (0=linear, 1=left circular, -1=right circular)
centroid = -15. # laser centroid position
ctau = 5.        # laser duration (see laser class for the exact definition)

n_e = 0.01 # plasma density 

def dens(x): # function describing the density profile
    L = 5.
 
    conds = [x>=p0, x>=p0+L]
    funcs = [lambda x: np.sin((x-p0)*np.pi/L/2)**2, 1.]

    return np.piecewise(x, conds, funcs)

ppc = 20   # plasma particles per cell (more = less noise)
p0 = 0.    # plasma start position

################## Simulation Initialisation #################


# initialise simulation object
sim = Simulation( x0, x1, Nx, diag_period=50*res )

# add the particles
elec = sim.add_new_species('elec', ppc=ppc, n=n_e, p0=p0, dens=dens, add_tags=True )

#add a tracking diagnostic
sim.add_tracking_diagnostic(elec, np.random.choice(1000, 10, replace=False), track_E=True, track_B=True)

# add lasers and external fields
sim.add_laser(a0, p=p, x0=centroid, ctau=ctau, clip=3*ctau )

sim.set_moving_window(True)

##################### Simulation Execution ######################
# make a note of the start time
t0 = time.time()

sim.step( N = 200*res + 1 ) # run the simulation

t1 = time.time()
print( 'All done in %.1f s'%(t1-t0))
 


