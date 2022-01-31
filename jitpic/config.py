"""
JitPIC configuration file
"""
import os

# numba option defaults
parallel = True
fastmath = True
cache = True

# runtime toggles
if 'JITPIC_DISABLE_PARALLEL' in os.environ:
    if os.environ['JITPIC_DISABLE_PARALLEL'] == '1':
        parallel = False
        
if 'JITPIC_DISABLE_CACHING' in os.environ:
    if os.environ['JITPIC_DISABLE_CACHING'] == '1':
        cache = False
        
if 'JITPIC_DISABLE_FASTMATH' in os.environ:
    if os.environ['JITPIC_DISABLE_FASTMATH'] == '1':
        fastmath = False
