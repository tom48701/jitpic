from jitpic.main import simulation

sim = simulation( 0, 100, 1000, diag_period=250 )
sim.add_new_species( 'elec', 10, 0.005, 20, 100 )
sim.add_laser( 1, 10, 3 ) 

sim.step( 2501 ) 