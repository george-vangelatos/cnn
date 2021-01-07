import numpy as np
from skimage.util.shape import view_as_windows
from collections import OrderedDict
from Layer import Layer
from Weights_and_Biases import Weights, Biases
import ActFunc

class Conv2DLayer(Layer): # 2D convolutional layer of neurons
    def __init__(self, f_shape, depth, af, prev):
        self._f_shape = tuple(f_shape) # store (2D) filter shape 
        shape = Conv2DLayer._conv_shape(prev._shape[:-1], self._f_shape) + (depth,) # calculate shape of this layer
        super().__init__(shape, af, prev) # call base initialiser
        self._w = Weights((prev._shape[-1], np.prod(f_shape), depth), af.sigma(prev._size)) # initialise weights
        self._b = Biases(depth) # initialise biases

        # build indexes for faster forward and backward passes
        self._i2cidx_fwd = Conv2DLayer._build_i2c_idx(prev._shape[:-1], self._f_shape)
        self._i2cidx_bck = Conv2DLayer._build_i2c_idx(np.add(prev._shape[:-1], self._f_shape) - (1,), self._f_shape)

    def _CalcActivations(self, x, tr_flag): # calculate activations; tr_flag specifies whether training or not
        # transform all input images in mini-batch into columns in preparation for convolution (im2col)
        x = np.reshape(x, (x.shape[0], -1, self._prev._shape[2])) # unravel (input) channels from x
        x = np.transpose(x, (0,2,1)) # push (input) channels from last to 2nd index in x
        cols = x[:,:,self._i2cidx_fwd] # apply im2col transform to input x
        self._cols = cols if tr_flag else None # cache columns, if training

        # calculate and return activations
        a = cols @ self._w.values # calculate weighted inputs (z)
        a = np.sum(a, 1) + self._b.values # sum z across (input channels) and add biases
        a = self._af.phi(np.reshape(a, (x.shape[0], -1))) # shape to mini-batch and apply activation function
        return a 

    def _CalcDerivatives(self, dz, x, no_dx=False): # performs derivative calculations for convolutional layer
        # calculate cost derivatives wrt weights and biases
        dz = np.reshape(dz, (dz.shape[0], -1, self._shape[2])) # unravel layer channels from dz
        db = np.sum(dz, (0, 1)) / dz.shape[0] # sum dz's (within each layer channel) and average over mini-batch to get db
        dw = np.transpose(self._cols, (1, 0, 3, 2)) @ dz # calculate dw for each mini-batch input
        dw = np.sum(dw, 1) / dz.shape[0] # average dw's across inputs in mini-batch
        self._cols = None # clear this because it's big and no longer needed

        # calculate cost derivatives wrt to input, if requested
        if no_dx: dx = None
        else:
            dz = np.reshape(dz, (dz.shape[0],) + self._shape) # unravel dz completely 
            dz = np.transpose(dz, (0,3,1,2)) # push layer channels in dz forward from last to 2nd index
            dz = np.pad(dz, ((0,), (0,), (self._f_shape[0] - 1,), (self._f_shape[1] - 1,))) # pad dz for a full convolution
            dz = np.reshape(dz, (dz.shape[0], dz.shape[1], -1)) # re-ravel the end dimensions (ie the images)
            w = np.swapaxes(np.flip(self._w.values, 1), 0, 2) # rotate the weights in the filter by 180 degrees and swap the channel axes
            dx = np.reshape(np.sum(dz[:,:,self._i2cidx_bck] @ w, 1), (x.shape[0], -1)) # calculate dx and shape to mini-batch

        return (dw, db, dx) # return derivatives
   
    # calculates the shape of a convolution given shapes of input and filter (must be of same dimension)
    @staticmethod
    def _conv_shape(in_shape, f_shape): return tuple(np.subtract(in_shape, f_shape) + 1)

    # builds an im2col index which takes a flattened input image to its im2col representation
    @staticmethod
    def _build_i2c_idx(in_shape, f_shape): return Conv2DLayer._im2col(np.arange(np.prod(in_shape)).reshape(in_shape), f_shape)

    # returns the windows in x got by applying filter of given shape; each window is flattened into a row
    @staticmethod
    def _im2col(x, f_shape): 
        w = view_as_windows(x, f_shape) # get the windows in x
        return w.reshape(-1, np.prod(f_shape)) # flatten windows into rows and return them

    def Serialise(self): # convert layer to json data (ie a dict)
        d = self._b.values.shape[0] # get depth of this layer
        json_data = OrderedDict([('f_shape', self._f_shape), ('depth', d), ('act_func', ActFunc.map_to_json[self._af])])
        json_data.update(self._w.Serialise()) # serialise weights
        json_data.update(self._b.Serialise()) # serialise biases
        return json_data

    @staticmethod
    def Deserialise(json_data, prev): # create fully connected layer from json layer data
        layer = Conv2DLayer(json_data['f_shape'], json_data['depth'], ActFunc.map_from_json[json_data['act_func']], prev) # create layer
        layer._w.Deserialise(json_data) # get weight values
        layer._b.Deserialise(json_data) # get bias values
        return layer

    def ToText(self): # convert layer attributes to display text
        return 'shape={}, f_shape={}, act_func={} (params={:,})'.format(self._shape, self._f_shape, ActFunc.map_to_json[self._af], self.num_params())

