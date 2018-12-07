# coding:utf-8

from __future__ import print_function
import numpy as np
import tensorflow as tf

# 创建图和会话
graph = tf.Graph()
sess = tf.InteractiveSession(graph=graph)
# tensorflow_inception_graph.pb 文件中，既存储了inception 的网络结构，也存储了对应的数据
#使用下面的语句导入
model_fn = 'tensorflow_inception_graph.pb'
with tf.gfile.FastGFile(model_fn,'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

# 定义t_input 为输入的图像
t_input = tf.placeholder(np.float32,name='input')
image_mean = 117.0

#输入图像需要经过处理才能送入网络中
# expand_dims 是增加一维，从[height,width,channel]变成[1,height,width,channel]
t_preprocessed = tf.expand_dims(t_input - image_mean,0)
tf.import_graph_def(graph_def,{'input':t_preprocessed})

# find all conv layers
layers = [op.name for op in graph.get_operations() if op.type == 'Conv2D' and 'import/' in op.name]

print('Nnmber of layers',len(layers))
name = 'mixed4d_5x5_pre_relu'
print('shape of %s:%s' % (name,str(graph.get_tensor_by_name('import/'+ name + ':0').get_shape())))
print(layers)