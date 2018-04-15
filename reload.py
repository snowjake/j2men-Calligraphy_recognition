import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


tf.set_random_seed(1)
np.random.seed(1)

new_data = pd.read_csv("j2mentest.csv")
new_data = new_data.values.astype(np.float32)
np.random.shuffle(new_data)

test_x = new_data

def reload():
    print('This is reload')
    # build entire net again and restore
    tf_x = tf.placeholder(tf.float32, [None, 128*128]) / 255.
    image = tf.reshape(tf_x, [-1, 128, 128, 1])              # (batch, height, width, channel)
    tf_y = tf.placeholder(tf.int32, [None, 1])            # input y

    # CNN
    conv1 = tf.layers.conv2d(   # shape (128, 128, 1)
        inputs=image,
        filters=16,
        kernel_size=5,
        strides=1,
        padding='same',
        activation=tf.nn.relu
    )           # -> (128, 128, 16)
    pool1 = tf.layers.max_pooling2d(
        conv1,
        pool_size=2,
        strides=2,
    )           # -> (64, 64, 16)
    conv2 = tf.layers.conv2d(pool1, 32, 5, 1, 'same', activation=tf.nn.relu)    # -> (64, 64, 32)
    pool2 = tf.layers.max_pooling2d(conv2, 2, 2)    # -> (32, 32, 32)
    flat = tf.reshape(pool2, [-1, 32*32*32])          # -> (7*7*32, )
    output = tf.layers.dense(flat, 1)              # output layer

    #loss = tf.losses.softmax_cross_entropy(onehot_labels=tf_y, logits=output)           # compute cost
    #train_op = tf.train.AdamOptimizer(LR).minimize(loss)

    sess = tf.Session()
    # don't need to initialize variables, just restoring trained variables
    saver = tf.train.Saver()  # define a saver for saving and restoring
    saver.restore(sess, 'cnnnet/j2men')
    test_output = sess.run(output, {tf_x: test_x[:100]})
    pred_y = np.argmax(test_output, 1)
    print(pred_y, 'prediction number')


# destroy previous net
tf.reset_default_graph()

reload()
