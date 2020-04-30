import time
import math

class Stopwatch(): # stopwatch for measuring elapsed time
    def __init__(self): self.Reset() # start stopwatch
    def Reset(self): self._start = time.monotonic() # record start time
    def GetInterval(self): return time.monotonic() - self._start # return elapsed seconds since start
    def FormatInterval(self, interval): # format time interval as string
        hours = math.trunc(interval / 3600) # get hours
        mins = math.trunc((interval - hours * 3600) / 60) # get minutes
        secs = math.trunc(interval - hours * 3600 - mins * 60) # get seconds
        return f"{mins}m:{secs:02}s" if hours == 0 else f"{hours}h:{mins}m:{secs:02}s"
    def FormatCurrentInterval(self): return self.FormatInterval(self.GetInterval()) # formats current interval as string


