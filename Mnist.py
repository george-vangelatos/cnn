import gzip
import random
import numpy as np
from PIL import Image

class Mnist(): # provides access to MNIST data
    def __init__(self, fn_img, fn_lb): # loads MNIST images and corresponding labels from provided files
        # load image file header
        f = gzip.open(fn_img) # open image file
        self.__ReadInt(f, 2051) # read (and check) magic number
        self.num = self.__ReadInt(f, 60000, 10000) # read (and check) number of images
        self.rows = self.__ReadInt(f, 28) # read (and check) number of rows per image
        self.cols = self.__ReadInt(f, 28) # read (and check) number of columns per image
        self.pixels = self.rows * self.cols # calculate number of pixels in each image

        # load image data and close file
        self.images = f.read(self.num * self.pixels) # load image data as bytes
        self.images = np.reshape(np.array(bytearray(self.images)), (self.num, self.rows, self.cols)) # convert to numpy array
        f.close

        # load label file header
        f = gzip.open(fn_lb) 
        self.__ReadInt(f, 2049) # read (and check) magic number
        self.__ReadInt(f, self.num) # read (and check) number of labels

        # load in labels and close file
        self.labels = np.array(bytearray(f.read(self.num)))
        f.close 

    def __ReadInt(self, f, *allowed): # reads in integer from file f and returns it; checks integer is an allowed value
        i = int.from_bytes(f.read(4), byteorder="big") # read in integer
        assert i in allowed, "MNIST; file format" # check integer is an allowed value
        return i

    def Shuffle(self): # shuffles images and labels (preserves correspondence)
        shuf = list(zip(self.images, self.labels))
        random.shuffle(shuf)
        self.images, self.labels = zip(*shuf) 
        self.images = np.array(self.images) 
        self.labels = np.array(self.labels) 

    @staticmethod
    def RandomRoll(img): # randomly rolls image by up to 1 pixel along each axis
        return np.roll(img, (random.randint(-1, 1), random.randint(-1, 1)), axis=(0, 1))

    @staticmethod
    def Display(img, scale = 1): # displays image
        Image.fromarray(img, 'L').resize((img.shape[0] * scale, img.shape[1] * scale)).show() # scale and display image 

    # helper function for mapping byte pixel values in [0x00, 0xFF] to floating point numbers in [0.0, 1.0]
    _b2f = np.linspace(0.0, 1.0, 0x100) # lookup table
    Map = lambda p : Mnist._b2f[p] 
