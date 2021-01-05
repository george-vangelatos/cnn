from collections import OrderedDict
from Layer import Layer

class InputLayer(Layer): # input layer of neurons; must be first layer in network
    def __init__(self, shape): super().__init__(shape, None, None) # call base initialiser

    def _CalcActivations(self, x, tr_flag): 
        self._a = x if tr_flag else None # store input as activations if training (for subsequent back propagation)
        return x # return input as activations

    def BackProp(self, da, params): pass # no back propagation at this layer

    def Serialise(self): # convert layer to json data (ie a dict)
        return OrderedDict([('shape', self._shape)])

    @staticmethod
    def Deserialise(json_data, prev): # create input layer from json layer data
        return InputLayer(json_data['shape'])

    def ToText(self): # convert layer attributes to display text
        return 'shape={}'.format(self._shape)



