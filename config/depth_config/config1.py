import tensorflow as tf
# basic params
image_size = 224
image_channel = 1
batch_size = 32

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


# file params
params_file_name = 'depthcnn1.npz'
confusion_matrix_name = 'depthcnn1.jpeg'
log_name = 'depthcnn1.txt'
def should_record(i):
    return i % 5 == 0 and i != 0
