"""
JitPIC configuration file
"""
import os

# numba option defaults
parallel = True
fastmath = True
cache = True
# numba runtime toggles
if os.environ.get('JITPIC_DISABLE_PARALLEL', None) == '1':
    parallel = False
    
if os.environ.get('JITPIC_DISABLE_CACHING', None) == '1':
    cache = False 
    
if os.environ.get('JITPIC_DISABLE_FASTMATH', None) == '1':
    fastmath = False

# file overwriting policy
allow_overwrite = False

if os.environ.get('JITPIC_ALLOW_OVERWRITE', None) == '1':
    allow_overwrite = True
