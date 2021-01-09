import sys
from tqdm import tqdm

class tqdm_fix(tqdm): # fix progress bar display with some buffer flushing and carriage returns
    def __init__(self, *args, **kwargs):
        sys.stdout.flush()
        super().__init__(*args, **kwargs)
    
    def close(self, *args, **kwargs): 
        super().close(*args, **kwargs)
        sys.stdout.write('\r')
        sys.stdout.flush()

