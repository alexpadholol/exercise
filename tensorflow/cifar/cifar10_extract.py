import tensorflow as tf
import os
import scipy
from tensorflow_e.cifar import cifar10_input

def inputs_origin(data_dir):
    filenames = [os.path.join(data_dir,'data_batch_%d.bin' % (i + 1 )) for i in range(5)]
    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to fine file: %s' % f)
    filename_queue = tf.train.string_input_producer(filenames)
    read_input = cifar10_input.read_cifar10(filename_queue)
    return tf.cast(read_input.uint8image,tf.float32)

if __name__ == '__main__':
    with tf.Session() as sess:
        reshape_image = inputs_origin('G:\data\cifar10\cifar-10-batches-bin')
        threads = tf.train.start_queue_runners(sess=sess)
        sess.run(tf.global_variables_initializer())
        if not os.path.exists('G:\data\cifar10\\raw\\'):
            os.makedirs('G:\data\cifar10\\raw\\')
        for i in range(30):
            image_array = sess.run(reshape_image)
            scipy.misc.toimage(image_array).save('G:\data\cifar10\\raw\\%d.jpg' % i)