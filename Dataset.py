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
        self._y_onehot = np.identity(self.nc) # used to map labels to one-hot vectors

    @staticmethod
    def Load(path): # loads training, validation and testing datasets from location given by path; returns datasets    
        return (None, None, None)

    def BuildMiniBatch(self, start, num, expand=False): # builds batch of num (normalised) inputs X (and corresponding labels y)
        X = self.data[start : start+num] # get data for specified batch
        if expand: X = np.array([self.Expand(d) for d in X[:]]) # expand data if required
        X = self.Normalise(X.reshape(num, self.item_size)) # flatten data to get 1 per row and map to floating point values in [0, 1]
        y = self.labels[start : start+num] # get labels
        return X, y

    def OneHotEncoding(self, y): # encodes vector of labels into one-hot vector rows
        return np.array([self._y_onehot[i,] for i in y]) 
 
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
    