import numpy as np

def uniform_field(x, t):
    return np.ones_like(x)

class External_field:
    def __init__( self, field, component, magnitude, function=None ):
        
        self.field = field
        self.component = component
        self.magnitude = magnitude
        
        if function is None:
            self.function = uniform_field
        else:
            self.function = function
        
        return
    
    def add_field(self, x, t, F):

        F[self.component] += self.magnitude * self.function( x, t )    
        
        return F