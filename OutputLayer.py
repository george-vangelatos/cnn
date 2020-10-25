import numpy as np
from collections import OrderedDict
from FullConLayer import FullConLayer
import ActFunc

class OutputLayer(FullConLayer): # output layer of neurons; must be last layer network
    def __init__(self, size, af, params, prev): super().__init__(size, af, params, prev)

    def FeedForward(self, x, tr_flag): 
        a = self._CalcActivations(x) # calculate activations
        self._a = a if tr_flag else None # store activations if training (for subsequent back propagation)
        return a # return activations

class QuadOutputLayer(OutputLayer): # output layer that implements quadratic cost function: C(a) = 0.5 * (y - a)^2 
    def __init__(self, size, af, params, prev): super().__init__(size, af, params, prev)
    
    def BackProp(self, y_exp): # back-propagate given the expected output vectors for the mini-batch
        da = self._a - y_exp # calculate the cost derivatives wrt activations from the expected outputs
        super().BackProp(da) # back-propagate using these derivatives

    @staticmethod
    def Deserialise(json_data, params, prev): # create quad output layer from json layer data
        layer = QuadOutputLayer(json_data['size'], ActFunc.map_from_json[json_data['act_func']], params, prev) # create layer
        layer._w.Deserialise(json_data) # get weight values
        layer._b.Deserialise(json_data) # get bias values
        return layer
    
# nb, cross entropy cost function requires input in the range (0, 1) as (ln is undefined for non +ve values); choose activation function accordingly (eg, not tanh)
class XentOutputLayer(OutputLayer): # output layer that implements cross entropy cost function: C(a) = y * ln(a) + (1 - y) * ln(1 - a)
    def __init__(self, size, af, params, prev): super().__init__(size, af, params, prev)

    def _CalcDz(self, y_exp): # calculates cost derivatives wrt weighted inputs given expected outputs
        if self._af is ActFunc.Sigmoid: return self._a - y_exp # shortcut for sigmoid activation function
        return super()._CalcDz(np.divide(self._a - y_exp, self._af.phi_prime(self._a)))

    @staticmethod
    def Deserialise(json_data, params, prev): # create cross entropy output layer from json layer data
        layer = XentOutputLayer(json_data['size'], ActFunc.map_from_json[json_data['act_func']], params, prev) # create layer
        layer._w.Deserialise(json_data) # get weight values
        layer._b.Deserialise(json_data) # get bias values
        return layer

class SoftmaxOutputLayer(OutputLayer): # output layer that implements log-likelihood cost function with softmax activation: C(a) = -ln(a) (for a expected to be 1.0)
    def __init__(self, size, params, prev): super().__init__(size, ActFunc.Softmax, params, prev)

    def _CalcDz(self, y_exp): return self._a - y_exp # calculates cost derivatives wrt weighted inputs given expected outputs

    def Serialise(self): # convert layer to json data (ie a dict)
        json_data = OrderedDict([('size', self._size)])
        json_data.update(self._w.Serialise()) # serialise weights
        json_data.update(self._b.Serialise()) # serialise biases
        return json_data
        
    @staticmethod
    def Deserialise(json_data, params, prev): # create softmax output layer from json layer data
        layer = SoftmaxOutputLayer(json_data['size'], params, prev) # create layer
        layer._w.Deserialise(json_data) # get weight values
        layer._b.Deserialise(json_data) # get bias values
        return layer

    def ToText(self): # convert layer attributes to display text
        return 'size={}'.format(self._size)
