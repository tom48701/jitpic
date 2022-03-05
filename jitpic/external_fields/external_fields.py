import numpy as np

def uniform_field(x, t):
    return np.ones_like(x)

class External_field:
    """ External field to be applied to particles at the push """
    def __init__( self, field, component, magnitude, function=None ):
        """
        Define an external field to be applied to simulated particles
        
        External fields are calclulated and applied to all particles
        before they are pushed. 
        
        Parameters
        ----------
        field: str (`E` or `B`)
            The field to add to.
            
        component: int (0, 1 or 2)
            The field component index. 
            
        magnitude: float
            Magnitude of the field.
            
        function: func or None, optional
            Field profile, uniform if none specified (Default: None).
        
        Notes
        -----
        The field function signature must be as follows:
            
            ```F = field_function(x,t)```
            
            Parameters
            ----------
            x: array of floats
                Current particle positions
            
            t: float
                Current simulation time
                
            Returns
            -------
            F: array
                Particle field to be added

        
        Beyond this, the contents are entirely down to the user.
        """
        # register quantities
        self.field = field
        self.component = component
        self.magnitude = magnitude
        # use the default (flat) profile if none is specified
        if function is None:
            self.function = uniform_field
        else:
            self.function = function
        return
    
    def add_field(self, x, t, F):
        """
        Add the external field to the existing particle field (F).
        
        Parameters
        ----------
        x: array
            Position
    
        t: float
            Time
        
        F: array
            Particle field in question
            
        Returns
        -------
        F: array
            Modified field
        """
        F[self.component] += self.magnitude * self.function( x, t )    
        return F