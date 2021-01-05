import numpy as np
from skimage.util.shape import view_as_windows
from collections import OrderedDict
from Layer import Layer
from Weights_and_Biases import Weights, Biases
import ActFunc

class Conv2DLayer(Layer): # 2D convolutional layer of neurons
    def __init__(self, f_shape, depth, af, prev):
        self._f_shape = tuple(f_shape) # store (2D) filter shape 
        self._f_size = np.prod(f_shape) # store (2D) size of filter
        shape = Conv2DLayer._conv_shape(prev._shape[:-1], self._f_shape) + (depth,) # calculate shape of this layer
        super().__init__(shape, af, prev) # call base initialiser
        self._w = Weights((prev._shape[-1], self._f_size, depth), af.sigma(prev._size)) # initialise weights
        self._b = Biases(depth) # initialise biases

    def _CalcActivations(self, x, tr_flag): # calculate activations from input x
        batch_size = x.shape[0] # get number of individual inputs in mini-batch x

        # calculate activations for each input in mini-batch
        a = np.zeros((batch_size, self._size)) # pre-allocate matrix to hold activations
        for i in range(batch_size): # enumerate individual inputs in mini-batch
            xi = np.reshape(x[i], self._prev._shape) # pull out current input from batch and restore its shape
            zi = None # accumulates sum of weighted inputs for current batch input
            for c in range(self._prev._shape[-1]): # enumerate each channel in the current input
                xc = xi[:,:,c] # pull out current channel of current input
                zc = Conv2DLayer._im2col(xc, self._f_shape) @ self._w.values[c,:,:] # calculate weighted input for current input channel
                zi = zi + zc if zi is not None else zc # add weighted input to running total
            zi += self._b.values # apply bias to weighted input
            a[i] = self._af.phi(zi.ravel()) # calculate activations for current input channel and store with overall activations
        self._a = a if tr_flag else None # store activations if training
        return a

    def _CalcDerivatives(self, dz, x, no_dx=False): # performs derivative calculations for convolutional layer
        batch_size = x.shape[0] # get number of individual inputs in mini-batch x

        # initialise storage for derivative values
        dw_sum = np.zeros(np.shape(self._w.values)) # pre-allocate matrix to accumulate dw over mini-batch
        db_sum = None # stores running total of db over mini-batch
        dx = None if no_dx else np.zeros(x.shape) # pre-allocate matrix to store cost derivatives wrt x (if requested)

        # sum dw and db over mini-batch, and calculate dx for each input in mini-batch
        for i in range(batch_size): # enumerate individual inputs in mini-batch
            xi = np.reshape(x[i], self._prev._shape) # pull out current input from mini-batch and restore its shape
            dzi = np.reshape(dz[i], (-1, self._shape[-1])) # pull out dz for current input (and reshape it for later use)

            # calculate dw for current input and add it to running sum
            for c in range(self._prev._shape[-1]): # enumerate each channel in the current input
                xc = xi[:,:,c] # pull out current channel of current input
                dw_sum[c,:,:] += Conv2DLayer._im2col(xc, self._shape[:-1]) @ dzi # add to running total

            # calculate db for current input and add it to running sum
            dbi = np.reshape(np.sum(dzi, 0), self._b.values.shape) # sum dz's within filter channels to get db for current input
            db_sum = db_sum + dbi if db_sum is not None else dbi # add to running total

            # calculate dx for current input (if requested)
            if no_dx: continue # do not calculate if not needed
            for c in range(self._shape[-1]): # enumerate each channel of the current layer
                dzc = np.reshape(dzi[:,c], (self._shape[:-1])) # pull out dz for current channel and restore its 2D shape
                dzc = np.pad(dzc, np.subtract(self._f_shape, 1)) # pad dzc for full convolution
                dxc = Conv2DLayer._im2col(dzc, self._f_shape) @ np.flip(self._w.values[:,:,c], 1).T # get dx contribution for current channel
                dx[i] += np.ravel(dxc) # increase dx for current input by current channel's contribution

        return (dw_sum / batch_size, db_sum / batch_size, dx) 
    
    # calculates the shape of a convolution given shapes of input and filter (must be of same dimension)
    @staticmethod
    def _conv_shape(in_shape, f_shape): return tuple(np.subtract(in_shape, f_shape) + 1)

    # returns the windows in x got by applying filter of given shape; each window is flattened into a row
    @staticmethod
    def _im2col(x, f_shape): 
        w = view_as_windows(x, f_shape) # get the windows in x
        return w.reshape(-1, np.prod(f_shape)) # flatten windows into rows and return them

    @staticmethod
    def Deserialise(json_data, prev): # create fully connected layer from json layer data
        layer = Conv2DLayer(json_data['f_shape'], json_data['depth'], ActFunc.map_from_json[json_data['act_func']], prev) # create layer
        layer._w.Deserialise(json_data) # get weight values
        layer._b.Deserialise(json_data) # get bias values
        return layer

