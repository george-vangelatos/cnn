from collections import OrderedDict
from Layer import Layer

class InputLayer(Layer): # input layer of neurons; must be first layer in network
    def __init__(self, shape): 
        shape = (tuple(shape) + (1, 1, 1))[:3] # make shape a tuple of at least 3 dimensions, ie (row x column x channel)
        super().__init__(shape, None, None) # call base initialiser

    # feeds input x forward through the network; returns ultimate activations; flag specifies whether training or not; 
    # activations are stored if training
    def FeedForward(self, x, tr_flag=False): 
        self._a = x if tr_flag else None # store input as activations if training (for subsequent back propagation)
        return self._next.FeedForward(x, tr_flag) # feed the input forward

    def BackProp(self, da): pass # no back propagation at this layer

    def Serialise(self): # convert layer to json data (ie a dict)
        return OrderedDict([('shape', self._shape)])

    @staticmethod
    def Deserialise(json_data, params, prev): # create input layer from json layer data
        return InputLayer(json_data['shape'])

    def ToText(self): # convert layer attributes to display text
        return 'shape={}'.format(self._shape)



