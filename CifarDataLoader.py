########################################################################
#
# Functions for downloading the CIFAR-10 data-set from the internet
# and loading it into memory.
#
# Implemented in Python 3.5
#
# Usage:
# 1) Set the variable data_path with the desired storage path.
# 2) Call maybe_download_and_extract() to download the data-set
#    if it is not already located in the given data_path.
# 3) Call load_class_names() to get an array of the class-names.
# 4) Call load_training_data() and load_test_data() to get
#    the images, class-numbers and one-hot encoded class-labels
#    for the training-set and test-set.
# 5) Use the returned data in your own program.
#
# Format:
# The images for the training- and test-sets are returned as 4-dim numpy
# arrays each with the shape: [image_number, height, width, channel]
# where the individual pixels are floats between 0.0 and 1.0.
#
########################################################################
#
# This file is part of the TensorFlow Tutorials available at:
#
# https://github.com/Hvass-Labs/TensorFlow-Tutorials
#
# Published under the MIT License. See the file LICENSE for details.
#
# Copyright 2016 by Magnus Erik Hvass Pedersen
#
########################################################################
#
# The only thing i did was use and object instead of functions
#
########################################################################
import pickle

import os
import numpy as np


class CifarDataLoader:
    def __init__(self):
        self.img_size = 32
        self.num_channels = 3 #(R,G,B)
        self.num_classes = 10
        self.num_files = 1
        self.images_per_file = 10000
        self.num_images_to_train = self.num_files * self.images_per_file
        self.data_path = "data/CIFAR-10/"

    def get_file_path(self,filename=""):
        """
        Return the full path of a data-file for the data-set.
        If filename=="" then return the directory of the files.
        """
        return os.path.join(self.data_path, "cifar-10-batches-py/", filename)

    def unpickle(self,filename):
        """
        Unpickle the given file and return the data.
        Note that the appropriate dir-name is prepended the filename.
        """

        # Create full path for the file.
        file_path = self.get_file_path(filename)

        print("Loading data: " + file_path)

        with open(file_path, mode='rb') as file:
            # In Python 3.X it is important to set the encoding,
            # otherwise an exception is raised here.
            data = pickle.load(file, encoding='bytes')

        return data

    def load_data(self,filename):
        """
        Load a pickled data-file from the CIFAR-10 data-set
        and return the converted images (see above) and the class-number
        for each image.
        """

        # Load the pickled data-file.
        data = self.unpickle(filename)

        # Get the raw images.
        raw_images = data[b'data']

        # Get the class-numbers for each image. Convert to numpy-array.
        cls = np.array(data[b'labels'])

        # Convert the images.
        images = self.convert_images(raw_images)

        return images, cls

    def convert_images(self,raw):
        """
        Convert images from the CIFAR-10 format and
        return a 4-dim array with shape: [image_number, height, width, channel]
        where the pixels are floats between 0.0 and 1.0.
        """

        # Convert the raw images from the data-files to floating-points.
        raw_float = np.array(raw, dtype=float) / 255.0

        # Reshape the array to 4-dimensions.
        images = raw_float.reshape([-1, self.num_channels, self.img_size, self.img_size])

        # Reorder the indices of the array.
        images = images.transpose([0, 2, 3, 1])

        return images

    def load_training_data(self):
        """
           Load all the training-data for the CIFAR-10 data-set.
           The data-set is split into 5 data-files which are merged here.
           Returns the images, class-numbers and one-hot encoded class-labels.
           """

        # Pre-allocate the arrays for the images and class-numbers for efficiency.
        images = np.zeros(shape=[self.num_images_to_train, self.img_size, self.img_size, self.num_channels], dtype=float)
        cls = np.zeros(shape=[self.num_images_to_train], dtype=int)

        # Begin-index for the current batch.
        begin = 0

        # For each data-file.
        for i in range(self.num_files):
            # Load the images and class-numbers from the data-file.
            images_batch, cls_batch = self.load_data(filename="data_batch_" + str(i + 1))

            # Number of images in this batch.
            num_images = len(images_batch)

            # End-index for the current batch.
            end = begin + num_images

            # Store the images into the array.
            images[begin:end, :] = images_batch

            # Store the class-numbers into the array.
            cls[begin:end] = cls_batch

            # The begin-index for the next batch is the current end-index.
            begin = end

        return images, cls

    def load_test_data(self):
        """
        Load all the test-data for the CIFAR-10 data-set.
        Returns the images, class-numbers.
        """

        images, cls = self.load_data(filename="test_batch")

        return images, cls

    def load_class_names(self):
        """
        Load the names for the classes in the CIFAR-10 data-set.
        Returns a list with the names. Example: names[3] is the name
        associated with class-number 3.
        """

        # Load the class-names from the pickled file.
        raw = self.unpickle(filename="batches.meta")[b'label_names']

        # Convert from binary strings.
        names = [x.decode('utf-8') for x in raw]

        return names