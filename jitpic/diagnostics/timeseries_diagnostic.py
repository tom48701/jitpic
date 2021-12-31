import h5py
import numpy as np

from ..utils import make_directory

class Timeseries_diagnostic:
    def __init__(self, sim, fields, cells, buffer_size,
                 diagdir='diags', fname='timeseries_data'):
        """
        Initialise a time domain field diagnostic. Collect specified field 
        information for specified cells every step.
        
        sim         : simulation  : simulation object to attach to
        fields      : list of str : list of fields to track ('E','B','J')
        cells       : list of int : list of cells to track [0 : Nx-1]
        buffer_size : int         : size of the buffer (should be diag_period)
        diagdir     : str         : folder to write output to
        fname       : str         : output file name
        """
        
        print('Initialising a timeseries diagnostic at step %i'%sim.iter)
        fstr = ', '.join([f for f in fields])
        cstr = ', '.join([str(c) for c in cells])
        print('Tracking fields: (%s) in cells: (%s)\n'%(fstr, cstr))
        
        assert sim.diag_period > 0, 'diag_period must be > 0 to initialise a timeseries diagnostic'
            
        self.sim = sim
        self.fields = fields
        self.cells = cells
        
        self.istart = sim.iter
        
        self.repeating_buffer_size = buffer_size
        self.initial_buffer_size = buffer_size - (sim.iter % sim.diag_period)
        self.buffer_size = self.initial_buffer_size
        
        self.buffer = {}
        self.buffer['t'] = np.zeros(self.repeating_buffer_size)
        for cell in cells:
            for field in fields:
                self.buffer['%s_%i'%(field, cell)] = np.zeros( (3,self.repeating_buffer_size) )
        
        # create the file with initial empty datasets
        make_directory( diagdir )
        
        self.filepath = '%s/%s.h5'%(diagdir, fname)
        
                 
        with h5py.File(self.filepath, 'w') as f:
            
            f.attrs['dt'] = sim.dt
                            
            f.create_dataset( 't', shape=(0,), dtype=np.double, 
                             maxshape=(None,) )  
            
            f.create_dataset( 'cells', data=np.array(cells) )
            
            for field in fields:  
                for cell in cells:
                    for i in range(3):
                            
                        f.create_dataset( '%s%i_%i'%(field, i, cell), 
                                         shape=(0,), dtype=np.double, 
                                         maxshape=(None,) )                    
                        
        return
    
    def gather_data(self, sim): 
        """
        Gather the required field data from the grid
        
        sim : simulation : simulation object
        """
        
        ibuff = sim.iter % self.repeating_buffer_size - self.repeating_buffer_size
        
        self.buffer['t'][ibuff] = self.sim.t
        
        for cell in self.cells:
            for field in self.fields:
                
                for i in range(3):
                    self.buffer['%s_%i'%(field, cell)][i][ibuff] = getattr(self.sim.grid, field)[i][cell]
        return
    
    
    def write_data(self, sim):
        """
        Append the buffers to file
        
        sim : simulation : simulation object
        """
        with h5py.File(self.filepath, 'a') as f:
            
            current_size = f['t'].shape[0]
            f['t'].resize( (current_size + self.buffer_size), axis=0)                    
            f['t'][-self.buffer_size:] = self.buffer['t'][-self.buffer_size:]
            
            for cell in self.cells:
                for field in self.fields:  
                        
                    for i in range(3):
                        
                        dset = '%s%i_%i'%(field, i, cell)
                        f[dset].resize( (current_size + self.buffer_size), axis=0)                    
                        f[dset][-self.buffer_size:] = self.buffer['%s_%i'%(field,cell)][i][-self.buffer_size:]
                        
        # set the buffer size to the repeat period after the first write
        self.buffer_size = self.repeating_buffer_size
                        
        
        return
                        
                        
