import os
import numpy as np
import random
import gzip
from PIL import Image
from Dataset import Dataset

class Mnist(Dataset): # implements MNIST dataset access
    nc = 10 # number of classes (ie there are 10 digits to classify)
    def __init__(self, data, labels): super().__init__(data, labels, Mnist.nc) # call base initialiser

    @staticmethod
    def Load(path): # load datasets from path
        # load training dataset
        images, labels = Mnist.__LoadMnist(os.path.join(path, 'train-images-idx3-ubyte.gz'), os.path.join(path, 'train-labels-idx1-ubyte.gz'))
        ds_tr = Mnist(images, labels) # create dataset object

        # load testing dataset
        images, labels = Mnist.__LoadMnist(os.path.join(path, 't10k-images-idx3-ubyte.gz'), os.path.join(path, 't10k-labels-idx1-ubyte.gz'))
        ds_te = Mnist(images, labels) # create dataset object

        return ds_tr, None, ds_te

    @staticmethod
    def __LoadMnist(fn_img, fn_lb): # loads MNIST images and labels from provided files; returns images and labels
        # load image file header
        f = gzip.open(fn_img) # open image file
        Mnist.__ReadInt(f, 2051) # read (and check) magic number
        num = Mnist.__ReadInt(f, 60000, 10000) # read (and check) number of images
        rows = Mnist.__ReadInt(f, 28) # read (and check) number of rows per image
        cols = Mnist.__ReadInt(f, 28) # read (and check) number of columns per image

        # load image data and close file
        images = f.read(num * rows * cols) # load image data as bytes
        images = np.reshape(np.array(bytearray(images)), (num, rows, cols)) # convert to numpy array
        f.close

        # load label file header
        f = gzip.open(fn_lb) 
        Mnist.__ReadInt(f, 2049) # read (and check) magic number
        Mnist.__ReadInt(f, num) # read (and check) number of labels

        # load in labels and close file
        labels = np.array(bytearray(f.read(num)))
        f.close 

        return images, labels # return images and labels

    @staticmethod
    def __ReadInt(f, *allowed): # helper function that reads in integer from file f and returns it; checks integer is an allowed value
        i = int.from_bytes(f.read(4), byteorder="big") # read in integer
        assert i in allowed, "MNIST; file format" # check integer is an allowed value
        return i

    @staticmethod
    def Expand(img): # randomly rolls image by up to 1 pixel along each axis
        return np.roll(img, (random.randint(-1, 1), random.randint(-1, 1)), axis=(0, 1))

    # maps byte pixel values in [0x00, 0xFF] to floating point numbers in [0.0, 1.0]
    _b2f = np.linspace(0.0, 1.0, 0x100) # lookup table
    @staticmethod
    def Normalise(v): return Mnist._b2f[v]

    @staticmethod
    def Display(img, scale = 1): # displays image
        Image.fromarray(img, 'L').resize((img.shape[0] * scale, img.shape[1] * scale)).show() # scale and display image 
