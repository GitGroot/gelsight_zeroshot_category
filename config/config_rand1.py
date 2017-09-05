import tensorflow as tf
# basic params
image_size = 224
image_channel = 3
num_steps = 5
batch_size = 16

# train params
epoch = 500
lr = 1e-5
attr_num = 8
loss_fn = tf.square

# data params
test_large_label = [0, 4, 7]
train_large_label = [1, 2, 3, 5, 6]
test_label_num = 3
train_label_num = 5
data_path = '/home/unreal/gelsight/fuse2/fuse/data/gel_rand3_224'


# file params
params_file_name = 'cnnlstm_rand1.npz'
confusion_matrix_name = 'cnnlstm_rand1.jpeg'
log_name = 'cnnlstm_rand1.txt'
def should_record(i):
    return i % 5 == 0 and i != 0
