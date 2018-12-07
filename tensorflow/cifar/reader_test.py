import tensorflow as tf

with tf.Session() as sess:
    filenames = ['G:\data\\film_frame\Godfather1\\frame_000001.jpg','G:\data\\film_frame\Godfather1\\frame_000101.jpg','G:\data\\film_frame\Godfather1\\frame_100001.jpg']
    filename_queue = tf.train.string_input_producer(filenames,shuffle=True,num_epochs=3)

    reader = tf.WholeFileReader()
    key,value = reader.read(filename_queue)
    tf.local_variables_initializer().run()
    threads = tf.train.start_queue_runners(sess=sess)
    i = 0
    while True:
        i += 1
        image_data = sess.run(value)
        with open('read/test_%d.jpg' % i,'wb') as f:
            f.write(image_data)

if __name__ == '__main__':
    with tf.Session() as sess:
        reshape_image = inputs_origin('G:\data\cifar10\cifar-10-batches-bin')
        threads = tf.train.start_queue_runners(sess=sess)
        sess.run(tf.global_variables_initializer())
        if not os.path.exits('G:\data\cifar10\\raw\\'):
            os.makedirs('G:\data\cifar10\\raw\\')
        for i in range(30):
            image_array = sess.run(reshape_image)
            scipy.misc.toimage(image_array).save('G:\data\cifar10\\raw\\%d.jpg' % i)