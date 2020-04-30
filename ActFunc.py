import math
import numpy as np

class ActFunc(object):
    # override these for derived classes
    @staticmethod
    def phi(z): return None # calculate activation a = phi(z) given weighted input z
    @staticmethod
    def phi_prime(a): return None # calculate derivatives phi'(z) directly from activations a
    sigma = None # calculate sigma for initialising weights (best choice depends on the activation function)

    # weight initialisation functions
    @staticmethod
    def _He(n): return math.sqrt(2.0 / n)
    @staticmethod
    def _Xavier(n): return math.sqrt(1.0 / n)

class Sigmoid(ActFunc): # sigmoid activation function
    @staticmethod
    def phi(z): return 1.0 / (1.0 + np.exp(-z))
    @staticmethod
    def phi_prime(a): return a * (1 - a) # phi'(z) = phi(z)(1 - phi(z)) = a(1 - a)
    sigma = ActFunc._Xavier

class Tanh(ActFunc): # Tanh activation function
    @staticmethod
    def phi(z): return np.tanh(z)
    @staticmethod
    def phi_prime(a): return 1 - a * a
    sigma = ActFunc._Xavier

class Relu(ActFunc): # ReLU activation function
    @staticmethod
    def phi(z): return np.where(z < 0, 0, z)
    @staticmethod
    def phi_prime(a): return np.where(a < 0, 0, 1)
    sigma = ActFunc._He

class LeakyRelu(ActFunc): # Leaky ReLU activation function
    alpha = 0.01 # leak parameter (non-negative)
    @staticmethod
    def phi(z): return np.where(z < 0, LeakyRelu.alpha * z, z)
    @staticmethod
    def phi_prime(a): return np.where(a < 0, LeakyRelu.alpha, 1)
    sigma = ActFunc._He

class Softmax(ActFunc): # Softmax activation function (not for general use--used to implement Softmax output layer)
    @staticmethod
    def phi(z): 
        terms = np.exp(z) # calculate e^z terms
        row_sums = np.sum(terms, 1).reshape(terms.shape[0], 1) # calculate sums of these terms for each input in the mini-batch (ie sum along rows)
        return np.divide(terms, row_sums) # calculate softmax values (divide terms by sum per input row)
    @staticmethod
    def phi_prime(a): return None # dummy function that is never called
    sigma = ActFunc._Xavier

# maps between json labels and activation function classes
map_from_json = { 
    'sigmoid': Sigmoid,
    'tanh': Tanh,
    'relu': Relu,
    'leaky_relu': LeakyRelu
    }
map_to_json = { val: key for (key, val) in map_from_json.items() }

