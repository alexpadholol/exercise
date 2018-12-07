import tensorflow as tf

from tensorflow_e.cifar import cifar10

FLAGS = tf.app.flags.FLAGS
FLAGS.data_dir = 'G:\data\cifar10\\'
cifar10.maybe_download_and_extract()