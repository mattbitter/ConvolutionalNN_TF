#Matthew Bitter
#CS498 - HW9
#MCS-DS
#May 7th, 2018

#REFERENCES: Big thank you to this github (nealwu) for details on how to implement tensorboard tracking
#using tf.layers architecture and to convert to format TFRecords
#https://github.com/tensorflow/models/blob/5ddd7e55c9ab6ae10a1459afc6309a2a417398d9/official/mnist/mnist.py#L211

"""Converts MNIST data to TFRecords file format with Example protos.
To read about optimizations that can be applied to the input preprocessing
stage, see: https://www.tensorflow.org/performance/performance_guide#input_pipeline_optimization.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

import tensorflow as tf

from tensorflow.contrib.learn.python.learn.datasets import mnist

parser = argparse.ArgumentParser()

parser.add_argument('--directory', type=str, default='/tmp/mnist_data',
                    help='Directory to download data files and write the '
                    'converted result.')

parser.add_argument('--validation_size', type=int, default=0,
                    help='Number of examples to separate from the training '
                    'data for the validation set.')


def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert_to(data_set, name):
  """Converts a dataset to TFRecords."""
  images = data_set.images
  labels = data_set.labels
  num_examples = data_set.num_examples

  if images.shape[0] != num_examples:
    raise ValueError('Images size %d does not match label size %d.' %
                     (images.shape[0], num_examples))
  rows = images.shape[1]
  cols = images.shape[2]
  depth = images.shape[3]

  filename = os.path.join(FLAGS.directory, name + '.tfrecords')
  print('Writing', filename)
  writer = tf.python_io.TFRecordWriter(filename)
  for index in range(num_examples):
    image_raw = images[index].tostring()
    example = tf.train.Example(features=tf.train.Features(feature={
        'height': _int64_feature(rows),
        'width': _int64_feature(cols),
        'depth': _int64_feature(depth),
        'label': _int64_feature(int(labels[index])),
        'image_raw': _bytes_feature(image_raw)}))
    writer.write(example.SerializeToString())
  writer.close()


def main(unused_argv):
  # Get the data.
  data_sets = mnist.read_data_sets(FLAGS.directory,
                                   dtype=tf.uint8,
                                   reshape=False,
                                   validation_size=FLAGS.validation_size)

  # Convert to Examples and write the result to TFRecords.
  convert_to(data_sets.train, 'train')
  convert_to(data_sets.validation, 'validation')
  convert_to(data_sets.test, 'test')


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)