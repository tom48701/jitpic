"""
This is an example script for a laser antenna in JitPIC,
we will demonstrate reflection off an overdense plasma
"""
import time
import numpy as np
from scipy.constants import m_e, m_p

from jitpic.main import Simulation
from jitpic.particles import Species

###################### Simulation Parameters #######################
    
x0 = -15.  # box start position
x1 = 10.  # box end position
res = 64  # resolution per laser wavelength
Nx = np.rint( (x1-x0)*res ).astype(int)   # automatically determine the total mumber of cells

# plasma parameters
ppc = 10
n_e = 2.
p0 = 0.
p1 = x1

# laser parameters
a0 = 0.01       # seed laser amplitude
ctau = 5.        # seed laser duration
    
########################## Simulation Initialisation #########################

# initialise the particle species 
elec = Species('elec', ppc, n_e, p0, p1  )
ions = Species('ions', ppc, n_e, p0, p1, q=1, m=m_p/m_e )

# initialise simulation object
sim = Simulation( x0, x1, Nx, species=[elec, ions], diag_period=2*res)

# add the laser as an antenna
sim.add_laser(a0, x0=x0-3*ctau, ctau=ctau, method='antenna', x_antenna=x0, t_stop=6*ctau )

################################ Execution ##############################
# make a note of the start time
t0 = time.time()

sim.step( N = 60*res +1 ) # run the simulation

t1 = time.time()
print( 'All done in %.1f s'%(t1-t0))


