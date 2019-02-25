'''
Created on Wed May 31 14:48:46 2017
Modified on Jan 15 2018
Remodified on 2019-02-20

@author: Frederik Kratzert
@modified: Jiwung Hyun

Containes a helper class for image input pipelines in tensorflow.
I modified this datagenerator class on https://github.com/kratzert/finetune_alexnet_with_tensorflow/blob/master/datagenerator.py
'''

import tensorflow as tf
import numpy as np
import os

from tensorflow.python.data.ops.dataset_ops import Dataset
from tensorflow.python.framework import dtypes
from tensorflow.python.framework.ops import convert_to_tensor

IMAGENET_MEAN = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32)

nus_wide_path = "/media/jiwunghyun/DATA/nus-wide" # "mnt/hdd0/nus-wide"
img_list_path = os.path.join(nus_wide_path, "ImageList")
tags_path = os.path.join(nus_wide_path, "NUS_WID_Tags")


class ImageDataGenerator(object):
    """Wrapper class around the new Tensorflows dataset pipeline.
    Requires Tensorflow >= version 1.12rc0
    """

    def __init__(self, mode, batch_size, num_classes, shuffle=True,
                 buffer_size=1000):
        """Create a new ImageDataGenerator.
        Recieves a path string to a text file, which consists of many lines,
        where each line has first a path string to an image and seperated by
        a space an integer, referring to the class number. Using this data,
        this class will create TensrFlow datasets, that can be used to train
        e.g. a convolutional neural network.
        Args:
            mode: Either 'training' or 'inference'. Depending on this value,
                different parsing functions will be used.
            batch_size: Number of images per batch.
            num_classes: Number of classes in the dataset.
            shuffle: Wether or not to shuffle the data in the dataset and the
                initial file list.
            buffer_size: Number of images used as buffer for TensorFlows
                shuffling of the dataset.
        Raises:
            ValueError: If an invalid mode is passed.
        """

        self.num_classes = num_classes
        self.duplicated_tag_list = []

        fp = open("/media/jiwunghyun/DATA/nus-wide/Concepts81.txt", 'r')
        tag_list = []
        for i in range(81):
            a = fp.readline().split('\n')[0]
            tag_list.append(a)
        fp.close()

        fp2 = open("/media/jiwunghyun/DATA/nus-wide/NUS_WID_Tags/TagList1k.txt", 'r')
        tag1000_list = []
        for i in range(1000):
            a = fp2.readline().split('\n')[0]
            tag1000_list.append(a)
        fp2.close()

        for i in range(len(tag1000_list)):
            for j in range(len(tag_list)):
                if tag1000_list[i] == tag_list[j]:
                    self.duplicated_tag_list.append(i)

        # retrieve the data from the text file
        if mode == 'training':
            self._read_train_txt_file()
        elif mode == 'inference':
            self._read_test_txt_file()

        # number of samples in the dataset
        self.data_size = len(self.labels)

        # initial shuffling of the file and label lists (together!)
        if shuffle:
            self._shuffle_lists()

        # convert lists to TF tensor
        self.img_paths = convert_to_tensor(self.img_paths, dtype=dtypes.string)
        self.labels = convert_to_tensor(self.labels, dtype=dtypes.int32)
        self.noisy_tags = convert_to_tensor(self.noisy_tags, dtype=dtypes.int32)

        # create dataset
        data = Dataset.from_tensor_slices((self.img_paths, self.labels, self.noisy_tags))

        # distinguish between train/infer. when calling the parsing functions
        if mode == 'training':
            data = data.map(self._parse_function_train)

        elif mode == 'inference':
            data = data.map(self._parse_function_inference)

        else:
            raise ValueError("Invalid mode '%s'." % (mode))

        # shuffle the first `buffer_size` elements of the dataset
        if shuffle:
            data = data.shuffle(buffer_size=buffer_size)

        # create a new dataset with batches of images
        data = data.batch(batch_size)

        self.data = data


    def _read_train_txt_file(self):
        self.img_paths = []
        self.labels = []
        self.noisy_tags = []

        tag_path = os.path.join(tags_path, 'Train_Tags81.txt')
        tags = open(tag_path, 'r')
        training_img_list_path = os.path.join(img_list_path, 'TrainImagelist.txt')
        training_img_list = open(training_img_list_path, 'r')

        noisy_tag_path = os.path.join(tags_path, 'Train_Tags1k.dat')
        noisy_tag = open(noisy_tag_path, 'rb')

        for idx in range(161789):
            tag_line = tags.readline().split(' ')
            tag_line = [int(i) for i in tag_line[:-1]]

            noisy_tag_line = noisy_tag.readline().split('\t'.encode())
            noisy_tag_line = [int(i) for i in noisy_tag_line[:-1]]

            new_noisy_tag_line = []
            for i in range(1000):
                if i in self.duplicated_tag_list:
                    pass
                else:
                    new_noisy_tag_line.append(noisy_tag_line[i])

            line = training_img_list.readline()
            line = line.split('\\')
            line = line[0] + '/' + line[1]
            # line = "D:/nus-wide/image/" + line[:-1]
            line = os.path.join(nus_wide_path, "image/" + line[:-1])
            if os.path.exists(line) and sum(tag_line) != 0:  # and sum(tag_line) != 0
                self.labels.append(tag_line)
                self.noisy_tags.append(new_noisy_tag_line)
                self.img_paths.append(line)

        print("train setting end")
        print("the number of training data:", len(self.labels))

    def _read_test_txt_file(self):
        self.img_paths = []
        self.labels = []
        self.noisy_tags = []

        tag_path = os.path.join(tags_path, 'Test_Tags81.txt')
        tags = open(tag_path, 'r')

        test_img_list_path = os.path.join(img_list_path, 'TestImagelist.txt')
        test_img_list = open(test_img_list_path, 'r')

        noisy_tag_path = os.path.join(tags_path, 'Test_Tags1k.dat')
        noisy_tag = open(noisy_tag_path, 'rb')

        for idx in range(52421):
            tag_line = tags.readline().split(' ')
            tag_line = [int(i) for i in tag_line[:-1]]

            noisy_tag_line = noisy_tag.readline().split('\t'.encode())
            noisy_tag_line = [int(i) for i in noisy_tag_line[:-1]]

            new_noisy_tag_line = []
            for i in range(1000):
                if i in self.duplicated_tag_list:
                    pass
                else:
                    new_noisy_tag_line.append(noisy_tag_line[i])

            line = test_img_list.readline()
            line = line.split('\\')
            line = line[0] + '/' + line[1]
            # line = "D:/nus-wide/image/" + line[:-1]
            line = os.path.join(nus_wide_path, "image/" + line[:-1])
            if os.path.exists(line) and sum(tag_line) != 0:  # and sum(tag_line) != 0
                # print(line)
                self.labels.append(tag_line)
                self.noisy_tags.append(new_noisy_tag_line)
                self.img_paths.append(line)

        print("test setting end")
        print("the number of test data:", len(self.labels))

    def _shuffle_lists(self):
        """Conjoined shuffling of the list of paths and labels."""
        path = self.img_paths
        labels = self.labels
        noisy_tags = self.noisy_tags
        permutation = np.random.permutation(self.data_size)
        self.img_paths = []
        self.labels = []
        self.noisy_tags = []
        for i in permutation:
            self.img_paths.append(path[i])
            self.noisy_tags.append(noisy_tags[i])
            self.labels.append(labels[i])

    def _parse_function_train(self, filename, label, noisy_tag):
        """Input parser for samples of the training set."""
        # convert label number into one-hot-encoding
        #one_hot = tf.one_hot(label, self.num_classes)

        # load and preprocess the image
        img_string = tf.read_file(filename)
        img_decoded = tf.image.decode_jpeg(img_string, channels=3)
        img_resized = tf.image.resize_images(img_decoded, [224, 224])
        """
        Dataaugmentation comes here.
        """
        img_centered = tf.subtract(img_resized, IMAGENET_MEAN)

        # RGB -> BGR
        img_bgr = img_centered[:, :, ::-1]

        return img_bgr, label, noisy_tag


    def _parse_function_inference(self, filename, label, noisy_tag):
        """Input parser for samples of the validation/test set."""
        # convert label number into one-hot-encoding
        # one_hot = tf.one_hot(label, self.num_classes)

        # load and preprocess the image
        img_string = tf.read_file(filename)
        img_decoded = tf.image.decode_png(img_string, channels=3)
        img_resized = tf.image.resize_images(img_decoded, [224, 224])
        img_centered = tf.subtract(img_resized, IMAGENET_MEAN)

        # RGB -> BGR
        img_bgr = img_centered[:, :, ::-1]

        return img_bgr, label, noisy_tag

