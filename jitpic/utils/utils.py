import os
from pathlib import Path

def make_directory( dirpath, cwd=os.getcwd() ):

    cwd = Path(cwd)
    dirpath = Path( cwd/dirpath )
        
    if not dirpath.exists():
        os.mkdir(dirpath)    
    elif not dirpath.is_dir():         
        print('eponymous file in place of requested directory, cannot save')
                
    return