from tensorlayer.layers import *
from tool.util import plot_confusion_matrix
from net import cnn_lstm
import tensorlayer as tl
import tensorflow as tf
from Data import Data
import math
from config.config import *
from tool.lib import log

train = False
x = tf.placeholder(tf.float32, [None, num_steps, image_size, image_size, image_channel])
inputs = tf.reshape(x, shape=[-1, image_size, image_size, image_channel])
network = cnn_lstm.build_net(None, inputs, num_steps, attr_num)
y = network.outputs
y_ = tf.placeholder(dtype=tf.float32, shape=[None, attr_num])
cost = tf.reduce_mean(loss_fn(y_ - y))
train_op = tf.train.AdamOptimizer(lr).minimize(cost)

ld = Data(attr_num, mode='rgb')
train_videos, train_videos_attr_labels, test_videos, test_videos_labels = ld.get_train_test()

sess = tf.Session()
initialize_global_variables(sess)
if train:
    params = tl.files.load_npz(name=params_file_name)
    tl.files.assign_params(sess, params, network)
    for i in range(500):
        for iter_num in range(int(math.ceil(len(train_videos)/batch_size))):
            video = train_videos[iter_num * batch_size:(iter_num + 1) * batch_size]
            label = train_videos_attr_labels[iter_num * batch_size:(iter_num + 1) * batch_size]
            feed_dict = {x: video, y_: label}
            feed_dict.update(network.all_drop)
            _, cost_val = sess.run([train_op, cost], feed_dict=feed_dict)
            log(i, iter_num, cost_val, 'log/' + log_name)
        if should_record(i):
            tl.files.save_npz(network.all_params,params_file_name, sess)

else:
    params = tl.files.load_npz(name=params_file_name)
    tl.files.assign_params(sess, params, network)
    dp_dict = tl.utils.dict_to_one(network.all_drop)
    print 'ok'
    test_attr_labels = ld.get_test_attr_label()
    right = 0
    all_y = []
    all_y_val = []
    for iter_num in range(len(test_videos)):
        video = test_videos[iter_num:iter_num + 1]
        label = test_videos_labels[iter_num:iter_num + 1]
        all_y.append(label[0])
        feed_dict = {x: video}
        feed_dict.update(dp_dict)
        y_val = sess.run(y, feed_dict=feed_dict)
        y_val = np.reshape(y_val, [-1])
        #
        #pred = ((test_attr_labels - y_val)**2).sum(axis=1).reshape(-1).argmin()
        S = test_attr_labels.T
        #print '+++++++++++++++', S
        pred = y_val.dot(S/[(S[:,i]**2).sum()**0.5 for i in range(S.shape[1])]).argmax()
        #
        # if pred1 != pred:
        #     print '++++++++++++'
        #     print y_val.astype(np.float32)
        #     print S
        all_y_val.append(pred)
        #print pred, pred1, label[0]
        if pred == label[0]:
            right += 1
    print right/float(len(test_videos))
    #plot_confusion_matrix(all_y, all_y_val, range(3), 'confusion_matrix/' + confusion_matrix_name)
