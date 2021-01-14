import numpy as np
from collections import OrderedDict
from Layer import Layer

class MaxPoolLayer(Layer): # 2D maxpool layer of neurons
    def __init__(self, p_shape, prev):
        assert (prev._shape[0] % p_shape[0] == 0) and (prev._shape[1] % p_shape[1] == 0) # check pool tiles input evenly
        shape = (prev._shape[0] // p_shape[0], prev._shape[1] // p_shape[1], prev._shape[-1],) # calculate shape of this layer
        super().__init__(shape, None, prev) # call base initialiser
        self._p_shape = tuple(p_shape) # store pool shape as tuple

    def _CalcActivations(self, x, tr_flag): # calculate activations; tr_flag specifies whether training or not
        # use reshaping and shuffle axes to form input into flattened pools
        x = np.reshape(x, (x.shape[0], self._shape[0], self._p_shape[0], self._shape[1], self._p_shape[1], self._shape[-1])) 
        x = np.transpose(x, (0,1,3,2,4,5)) # form pools
        x = np.reshape(x, (x.shape[0], self._shape[0], self._shape[1], -1, self._shape[-1])) # flatten pools

        # create index of maximum values for each pool and cache if training
        xi = np.expand_dims(np.argmax(x, 3), 3) 
        self._maxpool_idx = xi if tr_flag else None

        return np.reshape(np.take_along_axis(x, xi, 3), (x.shape[0], -1)) # return activations

    # calculates cost derivative wrt inputs and passes this back through the network to adjust previous layers
    def BackProp(self, da, params): 
        dx = np.zeros((da.shape[0], self._shape[0], self._shape[1], self._p_shape[0] * self._p_shape[1], self._shape[2])) # dx = 0
        da = np.reshape(da, self._maxpool_idx.shape) # unravel da completely (and add an extra dimension)
        np.put_along_axis(dx, self._maxpool_idx, da, 3) # transfer da values into dx according to cached index
        dx = np.reshape(dx, (da.shape[0], -1)) # reshape dx into mini-batch format
        self._prev.BackProp(dx, params) # back-propagate cost derivative wrt to inputs (= activations of previous layer)

    def Serialise(self): # convert layer to json data (ie a dict)
        return OrderedDict([('p_shape', self._p_shape)])

    @staticmethod
    def Deserialise(json_data, prev): # create maxpool layer from json layer data
        return MaxPoolLayer(json_data['p_shape'], prev) 

    def ToText(self): # convert layer attributes to display text
        return 'shape={}, p_shape={}'.format(self._shape, self._p_shape)
