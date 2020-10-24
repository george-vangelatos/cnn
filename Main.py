import sys
import os
import numpy as np
from Batcher import Batcher
from ProgressDisplay import ProgressDisplay
from Stopwatch import Stopwatch
from Network import Network
from HyperParameters import HyperParameters

def TestNetwork(net, ds, batch_size=500): # runs given dataset through network; counts number of correct outputs
    sum_correct = 0 # total number of correct outputs
    progress = ProgressDisplay(ds.num, 'Testing') # progress display for testing
    for start, num in Batcher(ds.num, batch_size): # batch up test cases
        progress.DisplayPercentage(start) # display progress percentage
        X, y_exp = ds.BuildMiniBatch(start, num) # build next batch of inputs
        Y_hat = net.FeedForward(X) # feed batch through the network, get output vectors for batch
        y_out = np.argmax(Y_hat, axis=1) # convert output vectors to labels
        sum_correct += sum(out == exp for (out, exp) in zip(y_out, y_exp)) # count correct outputs and add to total
    progress.DisplayPercentage(ds.num) # display 100%
    return sum_correct

def TrainEpoch(net, ds, params): # train network over a single epoch using provided dataset
    prog = ProgressDisplay(ds.num, 'Training') # initialise progress display
    ds.Shuffle() # shuffle training data
    for start, num in Batcher(ds.num, params.batch_size): # loop over mini-batches
        prog.DisplayPercentage(start) # display progress percentage            
        X, y = ds.BuildMiniBatch(start, num, params.expand_data) # build next batch of inputs
        Y_exp = ds.OneHotEncoding(y) # encode labels into one-hot vector rows
        net.Train(X, Y_exp) # feed batch through the network; back-propagate using expected outputs
    prog.DisplayPercentage(ds.num) # display 100%

def TrainNetwork(net, ds, num_epochs, params): # train network over multiple epochs using provided dataset
    sw_total, sw_epoch = Stopwatch(), Stopwatch() # start timing total training run and first epoch
    for epoch in range(1, num_epochs + 1): # loop through each epoch
        # run next epoch of training
        sw_epoch.Reset() # reset epoch stopwatch
        TrainEpoch(net, ds, params) # train network for a single epoch
        print('Epoch {} complete ({}).'.format(epoch, sw_epoch.FormatCurrentInterval())) # report time taken to complete epoch
    print('Training over {} epoch(s) complete ({}).'.format(num_epochs, sw_total.FormatCurrentInterval())) # report total time elapsed

# only execute the following if running as main module
if '__main__' == __name__: 
    # file names
    dir_work = 'C:\\Users\\georg\\Documents\\Visual Studio Code\\cnn\\projects' # working directory
    nn_in_fn = 'nn_in.json' # name of input network file
    nn_out_fn = 'nn_out.json' # name of output network file

    # training parameters
    params = HyperParameters(eta=0.05, L2=0.0001, mu=0.25, batch_size=11) # set hyper-parameters
    num_epochs = 2 # number of training epochs to run
    
    # load datasets; setup correct output vectors
    from projects.Mnist.Mnist import Mnist
    ds_tr, ds_va, ds_te = Mnist.Load(os.path.join(dir_work, 'Mnist\\data'))

    net = Network(os.path.join(os.path.join(dir_work, 'Mnist'), nn_in_fn), params) # load network from input file
    print('Starting precision: {:.2%}'.format(TestNetwork(net, ds_te)/ds_te.num)) # report starting precision
    TrainNetwork(net, ds_tr, num_epochs, params) # train network
    print('Ending precision: {:.2%}'.format(TestNetwork(net, ds_te)/ds_te.num)) # report ending precision
    net.Save(os.path.join(os.path.join(dir_work, 'Mnist'), nn_out_fn)) # save network to output file


