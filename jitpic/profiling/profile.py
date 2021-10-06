import cProfile, pstats

def profile( fname, wdir, pfname='jitpic_prof', nstats=20):
    """
    Profile a given simulation, return a list of functions and the
    walltime associated with each
    
    fname  : str : file to profile
    wdir   : str : working directory to run in 
    pfname : str : file to which to write the profiling information
    nstats : int : number of functions to list
    """
    cProfile.run("runfile('%s', wdir='%s')"%(fname, wdir), pfname)
    p = pstats.Stats(pfname)
    p.strip_dirs().sort_stats('tottime').print_stats(nstats)
    return
    