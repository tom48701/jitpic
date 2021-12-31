from .pusher import boris_push, cohen_push, vay_push
from .reseat import reseat_open, reseat_periodic
from .current import J1o, J2o, J3o, J4o, J1p, J2p, J3p, J4p
from .charge import R1o, R2o, R3o, R4o, R1p, R2p, R3p, R4p
from .interp import I1o, I2o, I3o, I4o, I1p, I2p, I3p, I4p
                       
# create a dictionary contaning the various functions
# J: current deposition
# R: charge deposition
# I: interpolation
# 1-4: shape factor
# open/pediodic: boundary types
#
# all functions of the same type (J/R/I/reseat/push) should take the same arguments
function_dict = {
    'J1_open':J1o,                      # current deposition functions
    'J2_open':J2o,
    'J3_open':J3o,
    'J4_open':J4o,
    'J1_periodic':J1p,
    'J2_periodic':J2p,
    'J3_periodic':J3p,
    'J4_periodic':J4p,
    'R1_open':R1o,                      # charge deposition functions
    'R2_open':R2o,
    'R3_open':R3o,
    'R4_open':R4o,
    'R1_periodic':R1p,
    'R2_periodic':R2p,
    'R3_periodic':R3p,
    'R4_periodic':R4p,
    'I1_open':I1o,                      # interpolation functions
    'I2_open':I2o,
    'I3_open':I3o,
    'I4_open':I4o,  
    'I1_periodic':I1p,
    'I2_periodic':I2p,
    'I3_periodic':I3p,
    'I4_periodic':I4p,
    'reseat_open':reseat_open,          # particle reseating functions
    'reseat_periodic':reseat_periodic,
    'boris':boris_push,                 # particle pushers
    'cohen':cohen_push,
    'vay':vay_push
    }

