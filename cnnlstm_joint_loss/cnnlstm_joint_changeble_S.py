from tensorlayer.layers import *
import sys
sys.path.append('../')
from tool.util import plot_confusion_matrix
from net import cnn_lstm
import tensorlayer as tl
import tensorflow as tf
from Data2 import Data
import math
from config.config import *
from tool.lib import log2
import sys

test_large_label = [0, 5, 7]
train_large_label = [1, 2, 3, 4, 6]
test_label_num = 3
train_label_num = 5

train = True
x = tf.placeholder(tf.float32, [None, num_steps, image_size, image_size, image_channel])
S = tf.placeholder(tf.float32, [attr_num, None])
inputs = tf.reshape(x, shape=[-1, image_size, image_size, image_channel])
network = cnn_lstm.build_net(None, inputs, num_steps, attr_num)
a = network.outputs

y = tf.matmul(a, S)
y_ = tf.placeholder(dtype=tf.int32, shape=[None])
a_ = tf.placeholder(dtype=tf.float32, shape=[None, attr_num])

y_op = tf.argmax(y, 1)
correct_num = tf.reduce_sum(tf.cast(tf.equal(tf.cast(y_op, tf.int32), y_), tf.float32))
acc = tf.reduce_mean(tf.cast(tf.equal(tf.cast(y_op, tf.int32), y_), tf.float32))
class_cost = tf.losses.sparse_softmax_cross_entropy(y_, y)
attr_cost = tf.reduce_mean(loss_fn(a_ - a))


ld = Data(attr_num, mode='rgb')
train_videos, train_videos_labels, train_videos_attr_labels, test_videos, test_videos_labels \
    = ld.get_train_test()
sess = tf.Session()
#lam = 1#0,0.1(73.48%),1(%),10(%)
lam = eval(sys.argv[1])
print lam
lr = 1e-5

cost = class_cost + lam*attr_cost
train_op = tf.train.AdamOptimizer(lr).minimize(cost)
initialize_global_variables(sess)
def to_str(a):
    r = '['
    for i in a:
        r += str(i)
    r+=']'
    return r
def get_filename():
    return 'rand('+str(lam)+')'+ to_str(test_large_label)
print get_filename()
if train:
    # params = tl.files.load_npz(name=params_file_name + '057dis' + str(lam) + '.npz')
    # tl.files.assign_params(sess, params, network)
    for i in range(1):
        for iter_num in range(int(math.ceil(len(train_videos)/batch_size))):
            video = train_videos[iter_num * batch_size:(iter_num + 1) * batch_size]
            label = train_videos_labels[iter_num * batch_size:(iter_num + 1) * batch_size]
            attr = train_videos_attr_labels[iter_num * batch_size:(iter_num + 1) * batch_size]
            feed_dict = {x: video, y_: label, a_: attr, S: ld.S_normal(True)}
            feed_dict.update(network.all_drop)
            _, cost_val, acc_val = sess.run([train_op, cost, acc], feed_dict=feed_dict)
            log2(i, iter_num, cost_val, acc_val, '../log/cnnlstm_joint_loss/' + log_name +'cos' + str(lam))
            print sess.run(attr_cost, feed_dict=feed_dict)
        if should_record(i):
            tl.files.save_npz(network.all_params, params_file_name + 'cos057' + str(lam), sess)
else:
    params = tl.files.load_npz(name=params_file_name + 'cos057' + str(lam) + '.npz')
    #params = tl.files.load_npz(name=params_file_name + '057'+'.npz')
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
        feed_dict = {x: video, y_: label, S: ld.S_normal(False)}
        feed_dict.update(dp_dict)
        correct_num_val = sess.run(correct_num, feed_dict=feed_dict)
        right += correct_num_val
    print right/float(len(test_videos))
    #plot_confusion_matrix(all_y, all_y_val, range(3), 'confusion_matrix/' + confusion_matrix_name)
