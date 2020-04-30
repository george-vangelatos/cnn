class ProgressDisplay(): # displays % progress to console 
    lots_of_spaces = " " * 40

    def __init__(self, size, job_name=None): # initialise with size of job, optional job name
        self._size = size # total units of work
        self._last_percent = -1 # no percentage displayed yet
        self.SetJobName(job_name) # store current job name

    def SetJobName(self, job_name): self._job_name = job_name # set name of job to display with percentage

    def DisplayPercentage(self, units): # display percentage of units completed so far
        percent = round((units * 100) / self._size) # calculate percentage to display
        if percent == self._last_percent: return # quit now if already displaying this percentage
        if self._job_name is None: print(f'\r{percent}%{ProgressDisplay.lots_of_spaces}', end='\r')
        else: print(f'\r{self._job_name}: {percent}%{ProgressDisplay.lots_of_spaces}', end='\r')
        self._last_percent = percent

    def DisplayMessage(self, msg): print(f'{msg}{ProgressDisplay.lots_of_spaces}') # display a line of text