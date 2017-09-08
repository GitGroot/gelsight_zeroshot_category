from tensorlayer.layers import *
from tool.util import plot_confusion_matrix
from net import depth_cnn
import tensorlayer as tl
import tensorflow as tf
from depth_cnn_data import Data
from config.depth_config.config import *
from tool.lib import log2
import math

train = False
x = tf.placeholder(tf.float32, [None, image_size, image_size, image_channel])
S = tf.placeholder(tf.float32, [attr_num, None])
network = depth_cnn.build_net(None, x, attr_num)
a = network.outputs

y = tf.matmul(a, S)
y_ = tf.placeholder(dtype=tf.int32, shape=[None])
a_ = tf.placeholder(dtype=tf.float32, shape=[None, attr_num])

y_op = tf.argmax(y, 1)
correct_num = tf.reduce_sum(tf.cast(tf.equal(tf.cast(y_op, tf.int32), y_), tf.float32))
acc = tf.reduce_mean(tf.cast(tf.equal(tf.cast(y_op, tf.int32), y_), tf.float32))
class_cost = tf.losses.sparse_softmax_cross_entropy(y_, y)
attr_cost = tf.reduce_mean(tf.square(a_ - a))


ld = Data(attr_num)
train_videos, train_videos_labels, train_videos_attr_labels, test_videos, test_videos_labels \
    = ld.get_train_test()
sess = tf.Session()
lam = 1
cost = class_cost +  lam*attr_cost
train_op = tf.train.AdamOptimizer(lr).minimize(cost)
initialize_global_variables(sess)
if train:
    params = tl.files.load_npz(name=params_file_name)
    tl.files.assign_params(sess, params, network)
    for i in range(300):
        for iter_num in range(int(math.ceil(len(train_videos)/batch_size))):
            video = train_videos[iter_num * batch_size:(iter_num + 1) * batch_size]
            label = train_videos_labels[iter_num * batch_size:(iter_num + 1) * batch_size]
            attr = train_videos_attr_labels[iter_num * batch_size:(iter_num + 1) * batch_size]
            feed_dict = {x: video, y_: label, a_: attr, S: ld.S(True)}
            feed_dict.update(network.all_drop)
            _, cost_val, acc_val = sess.run([train_op, cost, acc], feed_dict=feed_dict)
            log2(i, iter_num, cost_val, acc_val, '../log/depth/' + log_name + str(lam))
            print sess.run(attr_cost, feed_dict=feed_dict)
        if should_record(i):
            tl.files.save_npz(network.all_params, params_file_name, sess)
else:
    params = tl.files.load_npz(name=params_file_name)
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
