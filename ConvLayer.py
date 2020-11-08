import numpy as np
from collections import OrderedDict
from Layer import Layer
from Weights_and_Biases import Weights, Biases
import ActFunc

class ConvLayer(Layer): # convolutional layer of neurons
    def __init__(self, f_shape, depth, af, prev):
        shape = (ConvLayer._im2col_shape(prev._shape[:2], f_shape), depth) # calculate shape of this layer
        super().__init__(shape, af, prev) # call base initialiser
        self._w = Weights((np.prod(f_shape), depth, prev._shape[2]), af.sigma(prev._size)) # initialise weights
        self._b = Biases(depth) # initialise biases

    # calculate output shape of im2col operation given shapes of input and filter
    @staticmethod
    def _im2col_shape(in_shape, f_shape): return tuple(np.add(np.subtract(in_shape, f_shape), (1, 1)))