
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('G:\data\MNIST\\',one_hot=True)

X = tf.placeholder(tf.float32,[None,28*28])

W = tf.Variable(tf.zeros([28*28,10]))
b = tf.Variable(tf.zeros([10]))

# y 表示模型的输出
y = tf.nn.softmax(tf.matmul(X, W) + b)
y_ = tf.placeholder(tf.float32, [None, 10])

cross_entropy = tf.reduce_mean(-tf.reduce_mean(y_*tf.log(y)))

train_step = tf.train.GradientDescentOptimizer(0.01).minimize( cross_entropy)

sess = tf.InteractiveSession()
init_op = tf.global_variables_initializer()
sess.run(init_op)

for _ in range(10000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={X:batch_xs, y_:batch_ys})
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
print(sess.run(accuracy, feed_dict={X: mnist.test.images, y_: mnist.test.labels}))


