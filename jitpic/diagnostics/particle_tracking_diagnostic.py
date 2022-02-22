import h5py, os
import numpy as np

from ..utils import make_directory
 
def particle_buffer(buffer_size, track_x, track_p, track_E, track_B):
    buffer = {}
    if track_x:
        buffer['x'] = np.full(buffer_size, np.nan)
    if track_p:
        buffer['p'] = np.full((3,buffer_size), np.nan)
    if track_E:
        buffer['E'] = np.full((3,buffer_size), np.nan)
    if track_B:
        buffer['B'] = np.full((3,buffer_size), np.nan)
    return buffer
         
class Particle_tracking_diagnostic:
    def __init__(self, sim, species, tags,
                 track_x=True, track_p=True, track_E=False, track_B=False,
                 fname='tracking'):
        """
        Define a particle tracking diagnostic
        
        This will automatically collect and record particle data based on the
        particle ID. This requires tags to be turned on for the relevant species.
        
        Parameters
        ----------
        sim: Simulation
            Parent simulation instance.
            
        species: Species
            Species object to pull from.
            
        tags: list
            List of tags to track. 
        
        track_x: bool
            Track particle positions (Default: True)

        track_p: bool
            Track particle momentum (Default: True)
            
        track_E: bool
            Track particle B-field (Default: False)
            
        track_B: bool
            Track particle E-field (Default: False)            

        fname: str
            Tracking file name (Default: `tracking`)
        """
        print('Initialising a new particle tracking diagnostic at step %i'%sim.iter)
        print('Tracking %i particles from species: %s\n'%(len(tags), species.name))
        # nonzero diagnostic period required
        assert sim.diag_period > 0, 'diag_period must be > 0 to initialise a tracking diagnostic'
        # register the simulation and species objects
        self.sim = sim
        self.species = species
        # register IDs to be tracked
        self.tags = tags
        self.name = species.name
        # register quantities to be tracked
        self.track1D = {'x':track_x}
        self.track2D = {'p':track_p, 'E':track_E, 'B':track_B}
        # register the step at which the diag was initialised
        self.istart = sim.iter
        # register the buffer size, and calculate the residual for the first write
        self.repeating_buffer_size = sim.diag_period
        self.initial_buffer_size = self.repeating_buffer_size - (sim.iter % sim.diag_period)
        self.buffer_size = self.initial_buffer_size
        # register a time buffer and a dict of buffer object for each particle
        self.tbuffer = np.full(self.repeating_buffer_size, np.nan)
        self.buffer = {str(ID):particle_buffer(self.repeating_buffer_size, 
                         track_x, track_p, track_E, track_B) for ID in self.tags}
        # create the file with initial empty datasets
        make_directory( sim.diagdir )
        self.filepath = '%s/%s.h5'%(sim.diagdir,fname)
        #  check if the file already exists, set access mode appropriately
        if os.path.exists(self.filepath) and os.path.isfile(self.filepath):
            mode = 'a'
        else:
            mode = 'w'
        # populate the file
        with h5py.File(self.filepath, mode) as f:
            # simulation-level information
            f.attrs['dt'] = sim.dt
            # create species group
            g = f.create_group(self.name)
            # species-level information
            g.attrs['n'] = self.species.n
            g.attrs['m'] = self.species.m
            g.attrs['q'] = self.species.q
            g.create_dataset( 'tags', data=np.array(self.tags) )
            g.create_dataset( 't', shape=(0,), dtype=np.double, maxshape=(None,) )  
            # individual particle data
            for ID in self.tags:
                s = g.create_group(str(ID))
                for key, val in self.track1D.items():
                    if val:
                        s.create_dataset( key, shape=(0,), dtype=np.double, maxshape=(None,) )  
                for key, val in self.track2D.items():
                    if val:
                        s.create_dataset( key, shape=(3,0), dtype=np.double, maxshape=(3,None) )                         
        return

    def i_from_ID(self, ID):
        """ 
        Get the particle index from its ID, 
        return None if particle is missing
        """
        ID = int(ID)
        try:
            return np.argwhere( self.species.tags == ID )[0,0]
        except IndexError:
            return None
    
    def gather_data(self): 
        """ Gather particle data to buffers """
        # get the relative position within the buffer based on sim.iter
        ibuff = self.sim.iter % self.repeating_buffer_size - self.repeating_buffer_size
        # time is common to all particles
        self.tbuffer[ibuff] = self.sim.t
        # gather particle-specific data
        for ID in self.buffer.keys():
            i = self.i_from_ID(ID)  
            # fill with NaNs if particle is dead or missing
            if (i is None) or (not self.species.state[i]):
                for key, val in self.track1D.items():
                    if val:
                        self.buffer[ID][key][ibuff] = np.nan
                for key, val in self.track2D.items():
                    if val: 
                        self.buffer[ID][key][:,ibuff] = np.full((1,3), np.nan)
            # real data if particle is alive
            else:
                for key, val in self.track1D.items():
                    if val:
                        self.buffer[ID][key][ibuff] = getattr(self.species, key)[i]
                for key, val in self.track2D.items():
                    if val:   
                        self.buffer[ID][key][:,ibuff] = getattr(self.species, key)[:,i]
        return
    
    def write_data(self):
        """ Extend file datasets and append the buffers """
        with h5py.File(self.filepath, 'a') as f:
            # species group
            g = f[self.name]
            # get current dataset size and expand 
            current_size = g['t'].shape[0]
            g['t'].resize( (current_size + self.buffer_size), axis=0)                    
            g['t'][-self.buffer_size:] = self.tbuffer[-self.buffer_size:]
            # resize and write particle data
            for ID in self.buffer.keys():
                s = g[ID]
                # keep trying to write scalar quantities in case the particle has not yet been born
                i = self.i_from_ID(ID)
                if i is not None:
                    s.attrs['w'] = self.species.w[i]
                # 1D quantities
                for key, val in self.track1D.items():
                    if val:
                        s[key].resize( (current_size + self.buffer_size), axis=0)                    
                        s[key][-self.buffer_size:] = self.buffer[ID][key][-self.buffer_size:]
                # 2D quantities
                for key, val in self.track2D.items():
                    if val:
                        s[key].resize( (current_size + self.buffer_size), axis=1)                    
                        s[key][:,-self.buffer_size:] = self.buffer[ID][key][:,-self.buffer_size:] 
        # set the buffer size to the repeat period after the first write
        self.buffer_size = self.repeating_buffer_size
        return
                        
