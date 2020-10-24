from collections import namedtuple

HyperParameters = namedtuple('HyperParameters', 'eta L2 mu batch_size expand_data', defaults=(0.05, 0.0001, 0.25, 11, True))

# Parameters: 
#   eta: learning rate
#   L2: L2 regularisation parameter (0 = no weight decay)
#   mu: momentum parameter in [0, 1] (0 = none)
#   batch_size: size of mini-batches
#   expand_data: boolean flag specifying whether to augment data