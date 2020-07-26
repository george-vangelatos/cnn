import numpy as np
import random

class Dataset(): # abstract class for datasets
    def __init__(self, data, labels, nc): # creates dataset from supplied parameters 
        self.data = data # store numpy array holding data
        self.labels = labels # store numpy array holding labels
        self.nc = nc # store number of classes in dataset
        self.num = data.shape[0] # store number of data items
        self.item_shape = data[0].shape # store shape of each data item
        self.item_size = np.prod(self.item_shape) # store size of each data item

    @staticmethod
    def Load(path): # loads training, validation and testing datasets from location given by path; returns datasets    
        return (None, None, None)

    def Shuffle(self): # shuffles data and labels (preserves correspondence)
        shuf = list(zip(self.data, self.labels))
        random.shuffle(shuf)
        self.data, self.labels = zip(*shuf) 
        self.data = np.array(self.data) 
        self.labels = np.array(self.labels) 

    @staticmethod
    def Expand(d): return None # expands data item d in some random way; returns expanded d
    @staticmethod
    def Normalise(v): return None # normalises data value v; returns normalised v
    