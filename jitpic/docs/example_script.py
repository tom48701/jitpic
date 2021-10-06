import time
import numpy as np
from scipy.constants import m_e, m_p

from jitpic.main import simulation
from jitpic.particles import species

###################### Simulation Parameters #######################
    
x0 = -30.  # box start position
x1 = 200.  # box end position
res = 16   # resolution per laser wavelength
Nx = np.rint( (x1-x0)*res ).astype(int)   # automatically determine the total mumber of cells
   
a0 = 1.         # laser amplitude
p = 1           # laser polarisation (0=linear, 1=left circular, -1=right circular)
centroid = -15. # laser centroid position
tau = 5.        # laser duration (see laser class for the exact definition)

n_e = 0.005 # plasma density 

def dens(x): # function describing the density profile
    L = 5.
 
    conds = [x>=p0, x>=p0+L]
    funcs = [lambda x: np.sin((x-p0)*np.pi/L/2)**2, 1.]

    return np.piecewise(x, conds, funcs)

ppc = 10   # plasma particles per cell (more = less noise)
p0 = 0.    # plasma start position
p1 = x1    # plasma end position

################## Simulation Initialisation #################

# initialise the particle species 
elec = species('elec', ppc, n_e, p0, p1, dens=dens )
#ions = species('ions', ppc, n_e, p0, p1, dens=dens, m=m_p/m_e, q=1. )

# initialise simulation object
sim = simulation( x0, x1, Nx, species=[elec], diag_period=50*res )

#add a timeseries diagnostic
sim.add_timeseries_diagnostic( cells=[ int((x-x0)*res) for x in [0, 150] ], fields=['E'] )

# add lasers and external fields
sim.add_laser(a0, p=p, x0=centroid, tau=tau, clip=3*tau )
#sim.add_external_field('B', 1, 0.02)
  
##################### Simulation Execution ######################
#make a note of the start time
t0 = time.time()

sim.step( N = 300*res + 1 ) # run the simulation
  
t1 = time.time()
print( 'All done in %.1f s'%(t1-t0))
 
####################### Post Processing #####################
# # useful quantities:
# lambda_0 = 8e-7 # typical Ti:Sapph laser
# omega_0 = 2.*np.pi*c/lambda_0
# n_c = omega_0**2*m_e*epsilon_0/e**2
# rho_c = omega_0**2*m_e*epsilon_0/e
# E0 = m_e*c*omega_0/e
# B0 = E0/c
# J0 = rho_c*c
# omega_p = np.sqrt(n_e)*omega_0


