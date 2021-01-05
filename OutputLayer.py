import numpy as np
from collections import OrderedDict
from FullConLayer import FullConLayer
import ActFunc

class OutputLayer(FullConLayer): # output layer of neurons; must be last layer network
    def __init__(self, size, af, prev): super().__init__(size, af, prev)
    def FeedForward(self, x, tr_flag): return self._CalcActivations(x, tr_flag) # calculate and return activations

class QuadOutputLayer(OutputLayer): # output layer that implements quadratic cost function: C(a) = 0.5 * (y - a)^2 
    def __init__(self, size, af, prev): super().__init__(size, af, prev)
    
    def BackProp(self, y_exp, params): # back-propagate given the expected output vectors for the mini-batch
        da = self._a - y_exp # calculate the cost derivatives wrt activations from the expected outputs
        super().BackProp(da, params) # back-propagate using these derivatives

    @staticmethod
    def Deserialise(json_data, prev): # create quad output layer from json layer data
        layer = QuadOutputLayer(json_data['size'], ActFunc.map_from_json[json_data['act_func']], prev) # create layer
        layer._w.Deserialise(json_data) # get weight values
        layer._b.Deserialise(json_data) # get bias values
        return layer
    
# nb, cross entropy cost function requires input in the range (0, 1) as (ln is undefined for non +ve values); choose activation function accordingly (eg, not tanh)
class XentOutputLayer(OutputLayer): # output layer that implements cross entropy cost function: C(a) = y * ln(a) + (1 - y) * ln(1 - a)
    def __init__(self, size, af, prev): super().__init__(size, af, prev)

    def _CalcDz(self, y_exp): # calculates cost derivatives wrt weighted inputs given expected outputs
        if self._af is ActFunc.Sigmoid: return self._a - y_exp # shortcut for sigmoid activation function
        return super()._CalcDz(np.divide(self._a - y_exp, self._af.phi_prime(self._a)))

    @staticmethod
    def Deserialise(json_data, prev): # create cross entropy output layer from json layer data
        layer = XentOutputLayer(json_data['size'], ActFunc.map_from_json[json_data['act_func']], prev) # create layer
        layer._w.Deserialise(json_data) # get weight values
        layer._b.Deserialise(json_data) # get bias values
        return layer

class SoftmaxOutputLayer(OutputLayer): # output layer that implements log-likelihood cost function with softmax activation: C(a) = -ln(a) (for a expected to be 1.0)
    def __init__(self, size, prev): super().__init__(size, ActFunc.Softmax, prev)

    def _CalcDz(self, y_exp): return self._a - y_exp # calculates cost derivatives wrt weighted inputs given expected outputs

    def Serialise(self): # convert layer to json data (ie a dict)
        json_data = OrderedDict([('size', self._size)])
        json_data.update(self._w.Serialise()) # serialise weights
        json_data.update(self._b.Serialise()) # serialise biases
        return json_data
        
    @staticmethod
    def Deserialise(json_data, prev): # create softmax output layer from json layer data
        layer = SoftmaxOutputLayer(json_data['size'], prev) # create layer
        layer._w.Deserialise(json_data) # get weight values
        layer._b.Deserialise(json_data) # get bias values
        return layer

    def ToText(self): # convert layer attributes to display text
        return 'size={}'.format(self._size)
