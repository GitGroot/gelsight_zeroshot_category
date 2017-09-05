import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *

def conv_layers_simple_api(input):
    net_in = InputLayer(input, name='input')
    """ conv1 """
    network = Conv2d(net_in, n_filter=64, filter_size=(3, 3),
            strides=(1, 1), act=tf.nn.relu,padding='SAME', name='conv1_1')
    network = Conv2d(network, n_filter=64, filter_size=(3, 3),
            strides=(1, 1), act=tf.nn.relu,padding='SAME', name='conv1_2')
    network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2),
            padding='SAME', name='pool1')
    """ conv2 """
    network = Conv2d(network, n_filter=128, filter_size=(3, 3),
            strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv2_1')
    network = Conv2d(network,n_filter=128, filter_size=(3, 3),
            strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv2_2')
    network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2),
            padding='SAME', name='pool2')
    """ conv3 """
    network = Conv2d(network, n_filter=256, filter_size=(3, 3),
            strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv3_1')
    network = Conv2d(network, n_filter=256, filter_size=(3, 3),
            strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv3_2')
    network = Conv2d(network, n_filter=256, filter_size=(3, 3),
            strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv3_3')
    network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2),
            padding='SAME', name='pool3')
    """ conv4 """
    network = Conv2d(network, n_filter=512, filter_size=(3, 3),
            strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv4_1')
    network = Conv2d(network, n_filter=512, filter_size=(3, 3),
            strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv4_2')
    network = Conv2d(network, n_filter=512, filter_size=(3, 3),
            strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv4_3')
    network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2),
            padding='SAME', name='pool4')
    """ conv5 """
    network = Conv2d(network, n_filter=512, filter_size=(3, 3),
            strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv5_1')
    network = Conv2d(network, n_filter=512, filter_size=(3, 3),
            strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv5_2')
    network = Conv2d(network, n_filter=512, filter_size=(3, 3),
            strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv5_3')
    network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2),
            padding='SAME', name='pool5')
    return network

def fc_layers(net):
    network = FlattenLayer(net, name='flatten')
    network = DenseLayer(network, n_units=4096, act=tf.nn.relu, name='fc1_relu')
    network = DenseLayer(network, n_units=200, act=tf.nn.relu, name='fc2_relu')
    #network = DenseLayer(network, n_units=8, act=tf.identity, name='fc3_relu')
    return network

def build_net(session, input, class_num):
    net_cnn = conv_layers_simple_api(input)
    network = fc_layers(net_cnn)

    # npz = np.load('../gel_lbl_fold.npz')
    # params = []
    # for val in sorted(npz.items()):
    #     # print val
    #     print("  Loading %s" % str(val[1].shape))
    #     params.append(val[1])
    # params = tl.files.load_npz(path='../',name='gel_lbl_fold.npz')
    # #print params
    # tl.files.assign_params(session, params[:-2], network)

    network = DenseLayer(network, n_units=class_num, act=tf.nn.sigmoid, name='output_layer')


    return network
