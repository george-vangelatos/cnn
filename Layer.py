import numpy as np

class Layer(): # abstract class for a layer of neurons; derived classes must typically create: _w (weights), _b (biases), and _a (activations)
    def __init__(self, shape, af, prev):
        self._af = af # store activation function
        self._shape = shape # store shape of layer
        self._size = int(np.prod(shape)) # store number of nodes in this layer

        # layer storage
        self._w = None # weights
        self._b = None # biases
        self._a = None # activations (only used during training)

        # implement layer linking
        self._prev = prev # store previous layer
        if prev is not None: prev._next = self # if there is a previous layer, make it point to this layer
        self._next = None # no next layer yet

    # override these for derived classes
    def FeedForward(self, x, tr_flag): pass
    def _CalcDerivatives(self, dz, x): return (None, None, None)
    def Serialise(self): return None
    @staticmethod
    def Deserialise(json_data, params, prev): return None

    # updates weights and biases in this layer using cost derivative of (output) activations; calculates cost derivative wrt inputs
    # and passes this back through the network to adjust previous layers
    def BackProp(self, da):
        dz = self._CalcDz(da) # calculate cost derivatives wrt to weighted inputs
        (dw, db, dx) = self._CalcDerivatives(dz, self._prev._a) # calculate other derivatives
        self._w.GradDesc(dw) # update weights using gradient descent
        self._b.GradDesc(db) # update biases using gradient descent
        self._prev.BackProp(dx) # back-propagate cost derivative wrt to inputs (= activations of previous layer)

    # calculates cost derivatives wrt weighted inputs given those wrt activations using current layer activations
    def _CalcDz(self, da): return np.multiply(da, self._af.phi_prime(self._a)) 

        
    
