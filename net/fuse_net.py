import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *


def conv_layers_simple_api(input):
    net_in = InputLayer(input, name='depth_input')
    """ conv1 """
    network = Conv2d(net_in, n_filter=64, filter_size=(3, 3),
            strides=(1, 1), act=tf.nn.relu,padding='SAME', name='depth_conv1_1')
    network = Conv2d(network, n_filter=64, filter_size=(3, 3),
            strides=(1, 1), act=tf.nn.relu,padding='SAME', name='depth_conv1_2')
    network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2),
            padding='SAME', name='depth_pool1')
    """ conv2 """
    network = Conv2d(network, n_filter=128, filter_size=(3, 3),
            strides=(1, 1), act=tf.nn.relu, padding='SAME', name='depth_conv2_1')
    network = Conv2d(network,n_filter=128, filter_size=(3, 3),
            strides=(1, 1), act=tf.nn.relu, padding='SAME', name='depth_conv2_2')
    network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2),
            padding='SAME', name='depth_pool2')
    """ conv3 """
    network = Conv2d(network, n_filter=256, filter_size=(3, 3),
            strides=(1, 1), act=tf.nn.relu, padding='SAME', name='depth_conv3_1')
    network = Conv2d(network, n_filter=256, filter_size=(3, 3),
            strides=(1, 1), act=tf.nn.relu, padding='SAME', name='depth_conv3_2')
    network = Conv2d(network, n_filter=256, filter_size=(3, 3),
            strides=(1, 1), act=tf.nn.relu, padding='SAME', name='depth_conv3_3')
    network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2),
            padding='SAME', name='depth_pool3')
    """ conv4 """
    network = Conv2d(network, n_filter=512, filter_size=(3, 3),
            strides=(1, 1), act=tf.nn.relu, padding='SAME', name='depth_conv4_1')
    network = Conv2d(network, n_filter=512, filter_size=(3, 3),
            strides=(1, 1), act=tf.nn.relu, padding='SAME', name='depth_conv4_2')
    network = Conv2d(network, n_filter=512, filter_size=(3, 3),
            strides=(1, 1), act=tf.nn.relu, padding='SAME', name='depth_conv4_3')
    network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2),
            padding='SAME', name='depth_pool4')
    """ conv5 """
    network = Conv2d(network, n_filter=512, filter_size=(3, 3),
            strides=(1, 1), act=tf.nn.relu, padding='SAME', name='depth_conv5_1')
    network = Conv2d(network, n_filter=512, filter_size=(3, 3),
            strides=(1, 1), act=tf.nn.relu, padding='SAME', name='depth_conv5_2')
    network = Conv2d(network, n_filter=512, filter_size=(3, 3),
            strides=(1, 1), act=tf.nn.relu, padding='SAME', name='depth_conv5_3')
    network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2),
            padding='SAME', name='depth_pool5')
    return network

def fc_layers(net):
    network = FlattenLayer(net, name='depth_flatten')
    network = DenseLayer(network, n_units=4096, act=tf.nn.relu, name='depth_fc1_relu')
    network = DenseLayer(network, n_units=200, act=tf.nn.relu, name='depth_fc2_relu')
    return network

def build_depth_net(input):
    net_cnn = conv_layers_simple_api(input)
    network = fc_layers(net_cnn)
    return network


def conv_layers_simple_api_video(input):
    net_in = InputLayer(input, name='video_input')
    with tf.name_scope('preprocess') as scope:
        """
        Notice that we include a preprocessing layer that takes the RGB image
        with pixels values in the range of 0-255 and subtracts the mean image
        values (calculated over the entire ImageNet training set).
        """
        mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
        net_in.outputs = net_in.outputs - mean
    """ conv1 """
    network = Conv2d(net_in, n_filter=64, filter_size=(3, 3),
            strides=(1, 1), act=tf.nn.relu,padding='SAME', name='video_conv1_1')
    network = Conv2d(network, n_filter=64, filter_size=(3, 3),
            strides=(1, 1), act=tf.nn.relu,padding='SAME', name='video_conv1_2')
    network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2),
            padding='SAME', name='video_pool1')
    """ conv2 """
    network = Conv2d(network, n_filter=128, filter_size=(3, 3),
            strides=(1, 1), act=tf.nn.relu, padding='SAME', name='video_conv2_1')
    network = Conv2d(network,n_filter=128, filter_size=(3, 3),
            strides=(1, 1), act=tf.nn.relu, padding='SAME', name='video_conv2_2')
    network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2),
            padding='SAME', name='pool2')
    """ conv3 """
    network = Conv2d(network, n_filter=256, filter_size=(3, 3),
            strides=(1, 1), act=tf.nn.relu, padding='SAME', name='video_conv3_1')
    network = Conv2d(network, n_filter=256, filter_size=(3, 3),
            strides=(1, 1), act=tf.nn.relu, padding='SAME', name='video_conv3_2')
    network = Conv2d(network, n_filter=256, filter_size=(3, 3),
            strides=(1, 1), act=tf.nn.relu, padding='SAME', name='video_conv3_3')
    network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2),
            padding='SAME', name='video_pool3')
    """ conv4 """
    network = Conv2d(network, n_filter=512, filter_size=(3, 3),
            strides=(1, 1), act=tf.nn.relu, padding='SAME', name='video_conv4_1')
    network = Conv2d(network, n_filter=512, filter_size=(3, 3),
            strides=(1, 1), act=tf.nn.relu, padding='SAME', name='video_conv4_2')
    network = Conv2d(network, n_filter=512, filter_size=(3, 3),
            strides=(1, 1), act=tf.nn.relu, padding='SAME', name='video_conv4_3')
    network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2),
            padding='SAME', name='video_pool4')
    """ conv5 """
    network = Conv2d(network, n_filter=512, filter_size=(3, 3),
            strides=(1, 1), act=tf.nn.relu, padding='SAME', name='video_conv5_1')
    network = Conv2d(network, n_filter=512, filter_size=(3, 3),
            strides=(1, 1), act=tf.nn.relu, padding='SAME', name='video_conv5_2')
    network = Conv2d(network, n_filter=512, filter_size=(3, 3),
            strides=(1, 1), act=tf.nn.relu, padding='SAME', name='video_conv5_3')
    network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2),
            padding='SAME', name='video_pool5')
    return network

def fc_layers_video(net):
    network = FlattenLayer(net, name='video_flatten')
    network = DenseLayer(network, n_units=4096, act=tf.nn.relu, name='video_fc1_relu')
    network = DenseLayer(network, n_units=4096, act=tf.nn.relu, name='video_fc2_relu')
    #network = DenseLayer(network, n_units=8, act=tf.identity, name='fc3_relu')
    return network

def build_video_net(input, num_steps):
    net_cnn = conv_layers_simple_api_video(input)
    network = fc_layers_video(net_cnn)

    network = ReshapeLayer(network, shape=[-1, num_steps, int(network.outputs._shape[-1])])
    rnn1 = RNNLayer(network, cell_fn=tf.contrib.rnn.BasicLSTMCell, cell_init_args={}, n_hidden=200,
                    initializer=tf.random_uniform_initializer(-0.1, 0.1), n_steps=num_steps,
                    return_last=True, name='rnn_layer')
    return rnn1

def build_net(video_input, depima_input, num_steps, class_num):
    video_net = build_video_net(video_input, num_steps)
    depth_net = build_depth_net(depima_input)
    network = ConcatLayer([video_net, depth_net])
    network = DenseLayer(network, n_units=200, act=tf.nn.relu, name='fuse_dense1')
    network = DenseLayer(network, n_units=class_num, act=tf.nn.sigmoid, name='output')
    return network, video_net, depth_net
