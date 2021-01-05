from collections import OrderedDict
from Layer import Layer
from Weights_and_Biases import Weights, Biases
import ActFunc

class FullConLayer(Layer): # fully connected layer of neurons
    def __init__(self, size, af, prev):
        super().__init__((size,), af, prev) # call base initialiser
        self._w = Weights((prev._size, size), af.sigma(prev._size)) # initialise weights
        self._b = Biases((1, size)) # initialise biases

    def _CalcActivations(self, x, tr_flag): # calculate activations; tr_flag specifies whether training or not
        a = self._af.phi(x @ self._w.values + self._b.values) # calculate activations from input x
        self._a = a if tr_flag else None # cache activations if training (for subsequent back propagation)
        return a

    def _CalcDerivatives(self, dz, x, no_dx=False): # performs derivative calculations for fully connected layer
        dw = (x.transpose() @ dz) / x.shape[0] # calculate cost derivatives wrt to weights (average over mini-batch)
        db = dz.sum(axis = 0) / x.shape[0] # calculate cost derivatives wrt to biases (average over mini-batch)
        dx =  None if no_dx else dz @ self._w.values.transpose() # calculate cost derivative wrt x if requested
        return (dw, db, dx)

    def Serialise(self): # convert layer to json data (ie a dict)
        json_data = OrderedDict([('size', self._size), ('act_func', ActFunc.map_to_json[self._af])])
        json_data.update(self._w.Serialise()) # serialise weights
        json_data.update(self._b.Serialise()) # serialise biases
        return json_data

    @staticmethod
    def Deserialise(json_data, prev): # create fully connected layer from json layer data
        layer = FullConLayer(json_data['size'], ActFunc.map_from_json[json_data['act_func']], prev) # create layer
        layer._w.Deserialise(json_data) # get weight values
        layer._b.Deserialise(json_data) # get bias values
        return layer

    def ToText(self): # convert layer attributes to display text
        return 'size={}, act_func={}'.format(self._size, ActFunc.map_to_json[self._af])


