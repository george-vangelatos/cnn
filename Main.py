import os
import numpy as np
from Mnist import Mnist
from Weights_and_Biases import tr_params
from Batcher import Batcher
from ProgressDisplay import ProgressDisplay
from Stopwatch import Stopwatch
from Network import Network

# file names
dir_work = 'C:\\Users\\George\\Documents\\Visual Studio Code\\cnn\\working' # working directory
nn_in_fn = 'nn_in.json' # name of input network file
nn_out_fn = 'nn_out.json' # name of output network file

# load training and testing datasets; setup correct output vectors (y_values)
mnist_tr = Mnist(os.path.join(dir_work, 'mnist\\train-images-idx3-ubyte.gz'), os.path.join(dir_work, 'mnist\\train-labels-idx1-ubyte.gz'))
mnist_te = Mnist(os.path.join(dir_work, 'mnist\\t10k-images-idx3-ubyte.gz'), os.path.join(dir_work, 'mnist\\t10k-labels-idx1-ubyte.gz'))
y_values = np.identity(10) # identity matrix is equivalent to the correct output vectors for the network (indexed by digit)

# hyper-parameters
params = tr_params(0.05, 0.0001, 0.25) # eta = learning rate, L2 = regularisation parameter, mu = momentum parameter
batch_size = 11 # size of each input batch for training
test_batch_size = 500 # size of each input batch for testing
total_epochs = 10 # number of training epochs to run
expand_data = True # whether to expand training data by randomly shifting by up to 1 pixel along each axis

network = Network(os.path.join(dir_work, nn_in_fn), params) # load network from input file

def BuildInputBatch(mnist, start, num, expand=False): # builds batch of num inputs (and corresponding labels) from given mnist dataset
    x = mnist.images[start : start + num] # get images
    if expand: x = np.array([Mnist.RandomRoll(img) for img in x[:]]) # expand data if required
    x = Mnist.Map(x.reshape(num, mnist.pixels)) # reshape images to get 1 per row and map to floating point values in [0, 1]
    y = mnist.labels[start : start + num] # get labels
    return x, y

def TestNetwork(): # applies mnist test data to network; counts number of correct results
    sum_correct = 0
    progress = ProgressDisplay(mnist_te.num, 'Testing') # progress display for testing
    progress.DisplayPercentage(0) # display 0%
    for start, num in Batcher(mnist_te.num, test_batch_size): # batch up test cases
        x, y_exp = BuildInputBatch(mnist_te, start, num) # build next batch of inputs
        y_out = network.FeedForward(x) # feed batch through the network
        y_out = np.argmax(y_out, axis=1) # convert network output vectors to labels
        sum_correct += sum(out == exp for (out, exp) in zip(y_out, y_exp))
        progress.DisplayPercentage(start) # display progress percentage
    return sum_correct

def ProcessBatch(start, num): # trains network using mini-batch of size num starting at input start
    x, y_exp = BuildInputBatch(mnist_tr, start, num, expand_data) # build next batch of inputs
    y_exp = np.array([y_values[i,] for i in y_exp]) # convert labels to expected output vector rows
    network.Train(x, y_exp) # feed batch through the network; back-propagate using expected outputs

def TrainNetwork(): # train network using MNIST data
    sw_total, sw_epoch = Stopwatch(), Stopwatch() # start timing total training run and first epoch
    prog = ProgressDisplay(mnist_tr.num, 'Training') # progress display for training
    best_epoch, best_correct = 0, TestNetwork() # run initial test; initialise best epoch information
    prog.DisplayMessage('Start: {:.2%} ({})'.format(best_correct / mnist_te.num, sw_epoch.FormatCurrentInterval())) # display initial results

    for epoch in range(1, total_epochs + 1): # loop through each epoch
        # run next epoch of training
        sw_epoch.Reset() # reset epoch stopwatch
        prog.DisplayPercentage(0) # display 0%
        mnist_tr.Shuffle() # shuffle training data
        for start, num in Batcher(mnist_tr.num, batch_size): # loop over mini-batches
            prog.DisplayPercentage(start) # display progress percentage
            ProcessBatch(start, num) # process mini-batch
        prog.DisplayPercentage(mnist_tr.num) # display 100%

        # test network and display result; if network has been improved, save it then update best epoch
        num_correct = TestNetwork() # run test
        prog.DisplayMessage('Epoch {}: {:.2%} ({})'.format(epoch, num_correct / mnist_te.num, sw_epoch.FormatCurrentInterval()))
        if num_correct > best_correct: 
            best_epoch, best_correct = epoch, num_correct # update best epoch if improved
            if epoch != 0: network.Save(os.path.join(dir_work, nn_out_fn)) # save improved network to output file

    # report best epoch over training run
    if best_epoch == 0: prog.DisplayMessage('No improvement: {:.2%} (total: {})'.format(best_correct / mnist_te.num, sw_total.FormatCurrentInterval()))
    else: prog.DisplayMessage('Best: epoch {}, {:.2%} (total: {})'.format(best_epoch, best_correct / mnist_te.num, sw_total.FormatCurrentInterval()))

TrainNetwork() # kick-off the training


