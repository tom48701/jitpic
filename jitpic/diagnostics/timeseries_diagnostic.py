import h5py
import numpy as np

from ..utils import make_directory

class Timeseries_diagnostic:
    def __init__(self, sim, fields, cells, buffer_size,
                 diagdir='diags', fname='timeseries_data'):
        """
        Initialise a time domain diagnostic. Collect specified field 
        information for specified cells every step.
        
        sim         : simulation  : simulation object to attach to
        fields      : list of str : list of fields to track ('E','B','J')
        cells       : list of int : list of cells to track [0 : Nx-1]
        buffer_size : int         : size of the buffer (should be diag_period)
        diagdir     : str         : folder to write output to
        fname       : str         : output file name
        """
        
        if sim.diag_period == 0:
            raise ValueError('diag_period must be > 0 to initialise a timeseries diagnostic')
            
        self.sim = sim
        self.fields = fields
        self.cells = cells
        self.buffer_size = buffer_size
        
        self.buffer = {}
        self.buffer['t'] = np.zeros(buffer_size)
        for cell in cells:
            for field in fields:
                self.buffer['%s_%i'%(field, cell)] = np.zeros( (3,buffer_size) )
        
        # create the empty file
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
        
        ibuff = sim.iter % self.buffer_size
        
        self.buffer['t'][ibuff] = self.sim.t
        
        for cell in self.cells:
            for field in self.fields:
                
                for i in range(3):
                    self.buffer['%s_%i'%(field, cell)][i][ibuff] = getattr(self.sim.grid, field)[i][cell]
                    
    def write_data(self, sim):
        """
        Append the buffers to file
        
        sim : simulation : simulation object
        """
        with h5py.File(self.filepath, 'a') as f:
            
            current_size = f['t'].shape[0]
            
            f['t'].resize( (current_size + self.buffer_size), axis=0)                    
            f['t'][-self.buffer_size:] = self.buffer['t']
                        
            for cell in self.cells:
                for field in self.fields:  
                        
                    for i in range(3):
                        
                        dset = '%s%i_%i'%(field, i, cell)
                    
                        f[dset].resize( (current_size + self.buffer_size), axis=0)                    
                        f[dset][-self.buffer_size:] = self.buffer['%s_%i'%(field,cell)][i]
                        
                        
