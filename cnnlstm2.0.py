from tensorlayer.layers import *
from tool.util import plot_confusion_matrix
from net import cnn_lstm
import tensorlayer as tl
import tensorflow as tf
from Data2 import Data
import math
from config.config import *
from tool.lib import log2

train = False
x = tf.placeholder(tf.float32, [None, num_steps, image_size, image_size, image_channel])
S = tf.placeholder(tf.float32, [attr_num, None])
inputs = tf.reshape(x, shape=[-1, image_size, image_size, image_channel])
network = cnn_lstm.build_net(None, inputs, num_steps, attr_num)
a = network.outputs

y = tf.matmul(a, S)
y_ = tf.placeholder(dtype=tf.int32, shape=[None])
cost = tf.losses.sparse_softmax_cross_entropy(y_, y)
train_op = tf.train.AdamOptimizer(lr).minimize(cost)
y_op = tf.argmax(y, 1)
correct_num = tf.reduce_sum(tf.cast(tf.equal(tf.cast(y_op, tf.int32), y_), tf.float32))
acc = tf.reduce_mean(tf.cast(tf.equal(tf.cast(y_op, tf.int32), y_), tf.float32))

ld = Data(attr_num, mode='rgb')
train_videos, train_videos_labels, test_videos, test_videos_labels = ld.get_train_test()

sess = tf.Session()
initialize_global_variables(sess)
if train:
    params = tl.files.load_npz(path='npz/', name=params_file_name)
    # params = tl.files.load_npz(name=params_file_name)
    tl.files.assign_params(sess, params, network)
    for i in range(500):
        for iter_num in range(int(math.ceil(len(train_videos)/batch_size))):
            video = train_videos[iter_num * batch_size:(iter_num + 1) * batch_size]
            label = train_videos_labels[iter_num * batch_size:(iter_num + 1) * batch_size]
            feed_dict = {x: video, y_: label, S: ld.S(True)}
            feed_dict.update(network.all_drop)
            _, cost_val, acc_val = sess.run([train_op, cost, acc], feed_dict=feed_dict)
            log2(i, iter_num, cost_val, acc_val, 'log/' + log_name)
        if should_record(i):
            tl.files.save_npz(network.all_params, 'npz/'+params_file_name, sess)
else:
    params = tl.files.load_npz(path='npz/', name=params_file_name)
    tl.files.assign_params(sess, params, network)
    dp_dict = tl.utils.dict_to_one(network.all_drop)
    print 'ok'
    right = 0
    all_y = []
    all_y_val = []
    batch_size = 1
    for iter_num in range(int(math.ceil(len(test_videos)/batch_size))):
        video = test_videos[iter_num*batch_size:(iter_num+1)*batch_size]
        label = test_videos_labels[iter_num*batch_size:(iter_num+1)*batch_size]
        all_y.append(label)
        feed_dict = {x: video, y_: label, S: ld.S(False)}
        feed_dict.update(dp_dict)
        correct_num_val = sess.run(correct_num, feed_dict=feed_dict)
        right += correct_num_val
    print right/float(len(test_videos))
    #plot_confusion_matrix(all_y, all_y_val, range(3), 'confusion_matrix/' + confusion_matrix_name)
