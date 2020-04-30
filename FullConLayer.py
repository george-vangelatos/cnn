from collections import OrderedDict
from Layer import Layer
from Weights_and_Biases import Weights, Biases
import ActFunc

class FullConLayer(Layer): # fully connected layer of neurons
    def __init__(self, size, af, params, prev):
        super().__init__((size,), af, prev) # call base initialiser
        self._w = Weights((prev._size, size), af.sigma(prev._size), params) # initialise weights
        self._b = Biases((size,), params) # initialise biases

    # feeds input x forward through the network; returns ultimate activations; flag specifies whether training or not; 
    # activations are stored if training
    def FeedForward(self, x, tr_flag): 
        a = self._CalcActivations(x) # calculate activations
        self._a = a if tr_flag else None # store activations if training (for subsequent back propagation)
        return self._next.FeedForward(a, tr_flag) # feed activations forward through the network

    def _CalcActivations(self, x): return self._af.phi(x @ self._w.values + self._b.values) # calculate activations from input x

    def _CalcDerivatives(self, dz, x): # performs derivative calculations for fully connected layer
        dw = (x.transpose() @ dz) / x.shape[0] # calculate cost derivatives wrt to weights (average over mini-batch)
        db = dz.sum(axis = 0) / x.shape[0] # calculate cost derivatives wrt to biases (average over mini-batch)
        dx =  dz @ self._w.values.transpose() if self._prev._prev is not None else None # calculate cost derivative wrt x if not 2nd layer
        return (dw, db, dx)

    def Serialise(self): # convert layer to json data (ie a dict)
        json_data = OrderedDict([('size', self._size), ('act_func', ActFunc.map_to_json[self._af])])
        json_data.update(self._w.Serialise()) # serialise weights
        json_data.update(self._b.Serialise()) # serialise biases
        return json_data

    @staticmethod
    def Deserialise(json_data, params, prev): # create fully connected layer from json layer data
        layer = FullConLayer(json_data['size'], ActFunc.map_from_json[json_data['act_func']], params, prev) # create layer
        layer._w.Deserialise(json_data) # get weight values
        layer._b.Deserialise(json_data) # get bias values
        return layer



