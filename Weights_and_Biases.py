from collections import namedtuple
import numpy as np

# gradient descent parameters; eta = learning rate, L2 = L2 regularisation parameter (0 = no weight decay), mu = momentum parameter in [0, 1] (0 = none)
tr_params = namedtuple('tr_params', 'eta L2 mu')

class Weights(): # implements gradient descent for weights with L2 regularisation and momentum
    def __init__(self, shape, sigma, params):
        self.values = np.random.normal(0, sigma, shape) # create weights using random values from normal distribution
        self._vel = np.zeros(shape) # initialise corresponding velocities matrix to 0s
        self._params = params # store gradient descent parameters

    def GradDesc(self, dw): # adjust weights using gradient descent, given cost derivatives wrt to weights
        self._vel = self._params.mu * self._vel - dw * self._params.eta # update velocities for weights using cost derivatives
        self.values *= 1 - self._params.eta * self._params.L2 # decay the weights (L2 regularisation)
        self.values += self._vel # update weights using new velocities

    def Serialise(self): # return weights encoded as json data
        return { 'weights': np.reshape(self.values, self.values.size).tolist() }

    def Deserialise(self, json_data): # initialise weight values from json data, if it is present
        if 'weights' in json_data: self.values = np.reshape(np.array(json_data['weights']), self._vel.shape)

class Biases(Weights): # implements gradient descent for biases with momentum
    def __init__(self, shape, params): 
        params = params._replace(L2 = 0) # regularisation is n/a to biases
        super().__init__(shape, 1, params) # call base initialiser (without any scaling)

    def Serialise(self): # return biases encoded as json data
        return { 'biases': np.reshape(self.values, self.values.size).tolist() }

    def Deserialise(self, json_data): # initialise bias values from json data, if it is present
        if 'biases' in json_data: self.values = np.reshape(np.array(json_data['biases']), self._vel.shape)

