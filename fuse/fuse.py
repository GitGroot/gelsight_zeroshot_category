from tensorlayer.layers import *
from tool.util import plot_confusion_matrix
from net import fuse_net
import tensorlayer as tl
import tensorflow as tf
from fuse_data import Data
import math
from config.fuse_config.config import *
from tool.lib import log2

train = False
x1 = tf.placeholder(tf.float32, [None, num_steps, image_size, image_size, image_channel])
x2 = tf.placeholder(tf.float32, [None, image_size, image_size, 1])
S = tf.placeholder(tf.float32, [attr_num, None])
video_inputs = tf.reshape(x1, shape=[-1, image_size, image_size, image_channel])
network, video_net, depth_net = fuse_net.build_net(video_inputs, x2, num_steps, attr_num)
a = network.outputs

y = tf.matmul(a, S)
y_ = tf.placeholder(dtype=tf.int32, shape=[None])
a_ = tf.placeholder(dtype=tf.float32, shape=[None, attr_num])

y_op = tf.argmax(y, 1)
correct_num = tf.reduce_sum(tf.cast(tf.equal(tf.cast(y_op, tf.int32), y_), tf.float32))
acc = tf.reduce_mean(tf.cast(tf.equal(tf.cast(y_op, tf.int32), y_), tf.float32))
class_cost = tf.losses.sparse_softmax_cross_entropy(y_, y)
attr_cost = tf.reduce_mean(loss_fn(a_ - a))


ld = Data(attr_num)
train_videos, train_depimas, train_labels, train_attr_labels, test_videos, test_depimas, test_labels \
    = ld.get_train_test()
sess = tf.Session()
lam = 1
cost = class_cost + lam * attr_cost
train_op = tf.train.AdamOptimizer(lr).minimize(cost, var_list=network.all_params[:])
batch_size = 10
lr = 1e-5

initialize_global_variables(sess)
if train:
    # params = tl.files.load_npz(path='../npz/fuse/', name=params_file_name + str(lam) + '.npz')
    # tl.files.assign_params(sess, params, network)
    for i in range(300):
        for iter_num in range(int(math.ceil(len(train_videos)/batch_size))):
            video, depima = ld.get_content(train_videos[iter_num * batch_size:(iter_num + 1) * batch_size],
                                           train_depimas[iter_num * batch_size:(iter_num + 1) * batch_size])
            label = train_labels[iter_num * batch_size:(iter_num + 1) * batch_size]
            attr = train_attr_labels[iter_num * batch_size:(iter_num + 1) * batch_size]
            feed_dict = {x1: video, x2: depima, y_: label, a_: attr, S: ld.S(True)}
            feed_dict.update(network.all_drop)
            _, cost_val, acc_val, attr_cost_val = sess.run([train_op, cost, acc, attr_cost], feed_dict=feed_dict)
            log2(i, iter_num, cost_val, acc_val, '../log/fuse/' + log_name + str(lam))
            print attr_cost_val
        tl.files.save_npz(network.all_params, '../npz/fuse/'+params_file_name + str(lam), sess)
        #tl.files.save_npz(network.all_params, '../npz/fuse/'+params_file_name + str(lam) + '-2', sess)
else:
    # params = tl.files.load_npz(path='../npz/fuse/', name=params_file_name + str(lam) + '.npz')
    # tl.files.assign_params(sess, params, network)
    params = tl.files.load_npz(path='../npz/fuse/', name=params_file_name + str(lam) + '.npz')
    tl.files.assign_params(sess, params, network)
    dp_dict = tl.utils.dict_to_one(network.all_drop)
    print 'ok'
    right = 0
    all_y = []
    all_y_val = []
    batch_size = 1
    for iter_num in range(int(math.ceil(len(test_videos)/batch_size))):
        video, depima = ld.get_content(test_videos[iter_num*batch_size:(iter_num+1)*batch_size],
                                       test_depimas[iter_num*batch_size:(iter_num+1)*batch_size])
        label = test_labels[iter_num*batch_size:(iter_num+1)*batch_size]
        all_y.append(label[0])
        feed_dict = {x1: video, x2: depima, y_: label, S: ld.S(False)}
        feed_dict.update(dp_dict)
        correct_num_val = sess.run(correct_num, feed_dict=feed_dict)

        right += correct_num_val
    print right/float(len(test_videos))
#    plot_confusion_matrix(all_y, all_y_val, range(3), 'confusion_matrix/' + confusion_matrix_name)
# else:
#     params = tl.files.load_npz(path='../npz/fuse/', name=params_file_name + str(lam) + '.npz')
#     tl.files.assign_params(sess, params, network)
#     dp_dict = tl.utils.dict_to_one(network.all_drop)
#     print 'ok'
#     test_attr_labels = ld.get_test_attr_label()
#     right = 0
#     all_y = []
#     all_y_val = []
#     for iter_num in range(len(test_videos)):
#         video, depima= ld.get_content(test_videos[iter_num:iter_num + 1], test_depimas[iter_num:iter_num+1])
#         label = test_labels[iter_num:iter_num + 1]
#         all_y.append(label[0])
#         feed_dict = {x1: video, x2: depima}
#         feed_dict.update(dp_dict)
#         y_val = sess.run(a, feed_dict=feed_dict)
#         y_val = np.reshape(y_val, [-1])
#         #
#         #pred = ((test_attr_labels - y_val)**2).sum(axis=1).reshape(-1).argmin()
#         #S = test_attr_labels.T
#         # print S
#         S = ld.S(False)
#         # print S
#         #print '+++++++++++++++', S
#         # pred = y_val.dot(S/[(S[:,i]**2).sum()**0.5 for i in range(S.shape[1])]).argmax()
#         pred = y_val.dot(S).argmax()
#         #
#         # if pred1 != pred:
#         #     print '++++++++++++'
#         #     print y_val.astype(np.float32)
#         #     print S
#         all_y_val.append(pred)
#         #print pred, pred1, label[0]
#         if pred == label[0]:
#             right += 1
#     print right/float(len(test_videos))
#     #plot_confusion_matrix(all_y, all_y_val, range(3), 'confusion_matrix/' + confusion_matrix_name)
