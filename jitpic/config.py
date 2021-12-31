"""
JitPIC configuration file
"""
import numba

# numba option defaults
parallel = True
fastmath = True
cache = True

# test the caching
@numba.njit(cache=cache)
def cache_test():
    return None

try:
    cache_test()
except RuntimeError:
    cache = False
    

    