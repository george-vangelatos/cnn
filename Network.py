from collections import OrderedDict
import json
import InputLayer
import Conv2DLayer
import FullConLayer
import OutputLayer

class Network(): # implements network of layers; including saving and loading to file
    # map between json labels and corresponding layer classes
    _map_from_json = {
        'input': InputLayer.InputLayer, 
        'conv': Conv2DLayer.Conv2DLayer,
        'full_con': FullConLayer.FullConLayer,
        'quad_output': OutputLayer.QuadOutputLayer,
        'xent_output': OutputLayer.XentOutputLayer,
        'softmax_output': OutputLayer.SoftmaxOutputLayer
    }
    _map_to_json = { val: key for (key, val) in _map_from_json.items() }

    # loads network from provided source (a json string or file)
    def __init__(self, json_str=None, json_fn=None): 
        if json_fn is None: network_data = json.loads(json_str) # load network into a dictionary from json string, otherwise...
        else: 
            with open(json_fn) as f: network_data = json.load(f) # load network from json file
        self._first_layer, self._last_layer = None, None # initialise first and last layer pointers
        for layer_data in network_data['network']: # loop over each layer in the file
            layer_type = layer_data['layer'] # get the type of the next layer
            self._last_layer = self._map_from_json[layer_type].Deserialise(layer_data, self._last_layer) # create next layer
            if self._first_layer is None: self._first_layer = self._last_layer # set first layer, if required

    def FeedForward(self, X): # feeds batch of input vectors forward through network; returns ultimate activations
        return self._first_layer.FeedForward(X, False)

    def Train(self, X, Y_exp, params): # feeds forward batch of input vectors X; then back-propagates using expected output vectors Y_exp
        self._first_layer.FeedForward(X, True) 
        self._last_layer.BackProp(Y_exp, params) 

    def Save(self, fn): # saves network to file fn
        network = [] # list used to store json data for each layer
        layer = self._first_layer # start with first layer
        while layer is not None: # loop until no more layers 
            json_data = OrderedDict([('layer', self._map_to_json[layer.__class__])]) # serialise layer type
            json_data.update(layer.Serialise()) # add serialised layer data
            network.append(json_data) # add layer serialisation to network list
            layer = layer._next # get next layer
        with open(fn, 'w') as f: json.dump(OrderedDict([('network', network)]), f) # label network list and write it to json file

    def Print(self): # print network model
        layer = self._first_layer # start with first layer
        num_params = 0 # counts number of trainable parameters
        while layer is not None: # loop until no more layers 
            print('{}: {}'.format(self._map_to_json[layer.__class__], layer.ToText())) # print out layer description
            num_params += layer.num_params() # add number of parameters in layer
            layer = layer._next
        print('(total params={:,})'.format(num_params))
        


