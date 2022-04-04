import h5py, os
import numpy as np

from ..utils import make_directory

def field_buffer(buffer_size, pos, fields, method):
    buffer = {}
    f = {}
    buffer['t'] = np.zeros(buffer_size)
    for x in pos:
        for field in fields:
            if method == 'cell':
                key = '%s_%i'%(field, x)
            elif method == 'position':
                key = '%s_%.6g'%(field, x)
            f[key] = (x, field, np.zeros((3,buffer_size)))
    buffer['f'] = f
    return buffer

class Timeseries_diagnostic:
    def __init__(self, sim, fields, pos, method='position', fname='timeseries'):
        """
        Initialise a timeseries diagnostic.
        
        Automatically collect the specified field information from either a
        list of positions or a list of cells at every timestep.
        
        Parameters
        ----------
        fields: list 
            List of fields to track from any combination of `E`,`B` and `J`.
            
        pos: list
            List of positions or cell indices to track.
            
        method: str (`position` or `cell`), optional
            The method to use.
            - `position` tracks based on position, interpolated from the grid
            - `cell` tracks the field values in the specified cells
            (Default: `position`)
 
        fname: str, optional
            Timeseries file name (Default: `timeseries`).
        """
        # nonzero diagnostic period required
        assert sim.diag_period > 0, 'diag_period must be > 0 to initialise a timeseries diagnostic'
        print('Initialising a new timeseries diagnostic at step %i'%sim.iter)
        fstr = ', '.join([f for f in fields])
        print('Tracking (%s) fields at %i points'%(fstr, len(pos)))
        print('Saving data in: %s\n'%fname)
        # register the simulation object
        self.sim = sim
        # the list of fields and cells to track
        self.fields = fields
        self.pos = pos
        # register the step at which the diag was initialised
        self.istart = sim.iter
        # register the method
        self.method = method
        # register the group name
        self.groupname = '%s_%i'%(self.method, self.istart)
        # register the buffer size, and calculate the residual for the first write
        self.repeating_buffer_size = sim.diag_period
        self.initial_buffer_size = self.repeating_buffer_size - (sim.iter % sim.diag_period)
        self.buffer_size = self.initial_buffer_size
        # register a dict to hold the buffers
        self.buffer = field_buffer(self.repeating_buffer_size, self.pos, self.fields, self.method)
        # create the file with initial empty datasets
        make_directory( sim.diagdir )
        self.filepath = '%s/%s.h5'%(sim.diagdir, fname)
        #  check if the file already exists, set access mode appropriately
        if os.path.exists(self.filepath) and os.path.isfile(self.filepath):
            mode = 'a'
        else:
            mode = 'w'
        # populate the file
        with h5py.File(self.filepath, mode) as f:
            # simulation-level information
            f.attrs['dt'] = sim.dt
            # create group for this diagnostic
            g = f.create_group(self.groupname)
            g.create_dataset( 't', shape=(0,), dtype=np.double, 
                             maxshape=(None,) )  
            g.create_dataset( 'points', data=np.array(pos) )
            # create datasets for each field
            for key in self.buffer['f'].keys():
                g.create_dataset( key, 
                                 shape=(3,0), dtype=np.double, 
                                 maxshape=(3,None) )                       
        return
    
    def gather_data(self): 
        """ Gather the required field data from the grid """
        # get the relative position within the buffer based on sim.iter
        ibuff = self.sim.iter % self.repeating_buffer_size - self.repeating_buffer_size
        self.buffer['t'][ibuff] = self.sim.t
        # get the fields for each position
        grid = self.sim.grid
        # iterate through the field buffer
        for val in self.buffer['f'].values():
            # unpack the contents of each entry
            x = val[0]
            field = val[1]
            buffer = val[2]
            # grid field
            Fg = getattr(grid, field)
            # select method
            if self.method == 'cell':
                # add to the buffer
                buffer[:,ibuff] = Fg[:,x] 
            else:
                # ignore points off the grid
                if not (grid.x0 <= x < grid.x1):
                    buffer[:,ibuff] = np.full((1,3), np.nan)
                else:
                    # mock-up the data for the interpolation function
                    Fs = np.zeros((3,1))
                    xi = np.array([(x - grid.x0)*grid.idx])
                    l = np.array([int(xi)], np.dtype('u4'))
                    r = l+1
                    # interpolate the field onto the position
                    self.sim.interpolate_func(Fg, Fg, Fs, Fs, l, r, xi, 1, np.array([True]))
                    # add to the buffer
                    buffer[:,ibuff] = Fs[:,0]
        return
    
    def write_data(self):
        """ Append the buffers to file """
        with h5py.File(self.filepath, 'a') as f:
            g = f[self.groupname]
            # expand and write all datasets
            current_size = g['t'].shape[0]
            g['t'].resize( (current_size + self.buffer_size), axis=0)                    
            g['t'][-self.buffer_size:] = self.buffer['t'][-self.buffer_size:]
            for key, val in self.buffer['f'].items():
                buffer = val[2]
                g[key].resize( (current_size + self.buffer_size), axis=1)                    
                g[key][:,-self.buffer_size:] = buffer[:,-self.buffer_size:]     
        # set the buffer size to the repeat period after the first write
        self.buffer_size = self.repeating_buffer_size
        return
    