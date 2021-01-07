import numpy as np

# abstract class for a layer of neurons; derived classes must typically create: _w (weights), _b (biases), and _a (activations)
class Layer(): 
    def __init__(self, shape, af, prev):
        self._af = af # store activation function
        self._shape = tuple(shape) # store shape of layer
        self._size = int(np.prod(shape)) # store number of nodes in this layer

        # layer storage
        self._w = None # weights
        self._b = None # biases

        # implement layer linking
        self._prev = prev # store previous layer
        if prev is not None: prev._next = self # if there is a previous layer, make it point to this layer
        self._next = None # no next layer yet

    # override these for derived classes
    def _CalcActivations(self, x, tr_flag): return None
    def _CalcDerivatives(self, dz, x, no_dx): return (None, None, None)
    def Serialise(self): return None
    @staticmethod
    def Deserialise(json_data, prev): return None
    def ToText(self): return None

    # calculates activations then feeds them forward through the network; flag specifies whether training or not
    def FeedForward(self, x, tr_flag): 
        a = self._CalcActivations(x, tr_flag) # calculate activations
        self._a = a if tr_flag else None # cache activations if training
        return self._next.FeedForward(a, tr_flag) # feed activations forward, return ultimate output

    # updates weights and biases in this layer using cost derivative of (output) activations and provided parameters; 
    # calculates cost derivative wrt inputs and passes this back through the network to adjust previous layers
    def BackProp(self, da, params):
        dz = self._CalcDz(da) # calculate cost derivatives wrt to weighted inputs
        (dw, db, dx) = self._CalcDerivatives(dz, self._prev._a, self._prev._prev is None) # get other derivatives (except dx for 2nd layer)
        self._w.GradDesc(dw, params) # update weights using gradient descent
        self._b.GradDesc(db, params) # update biases using gradient descent
        self._prev.BackProp(dx, params) # back-propagate cost derivative wrt to inputs (= activations of previous layer)

    # calculates cost derivatives wrt weighted inputs given those wrt activations using current layer activations
    def _CalcDz(self, da): return da * self._af.phi_prime(self._a)

    # calculates the number of trainable parameters in this layer
    def num_params(self): 
        nw = 0 if self._w is None else self._w.num_params()
        nb = 0 if self._b is None else self._b.num_params() 
        return nw + nb

        
    
