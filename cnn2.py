import tensorflow as tf
import numpy as np
import pandas as pd

tf.set_random_seed(1)
np.random.seed(1)

BATCH_SIZE = 50
LR = 0.001              # learning rate

new_data = pd.read_csv("j2men.csv")
new_data = new_data.values.astype(np.float32)
np.random.shuffle(new_data)
#new_data = new_data.reshape(-1,16385)
test_x = new_data[:, 1:]
test_y = new_data[:, :1]
print(test_x.shape,test_y.shape)
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

loss = tf.losses.mean_squared_error(labels=tf_y, predictions=output)           # compute cost
train_op = tf.train.AdamOptimizer(LR).minimize(loss)

#accuracy = tf.metrics.accuracy(          # return (acc, update_op), and create 2 local variables
#    labels=tf.argmax(tf_y, axis=1), predictions=tf.argmax(output, axis=1),)[1]

sess = tf.Session()
init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()) # the local var is for accuracy_op
sess.run(init_op)     # initialize var in graph
saver = tf.train.Saver()  # define a saver for saving and restoring
for step in range(780):
    print(step)    
    b_x, b_y = test_x[step*BATCH_SIZE:(step+1)*BATCH_SIZE,: ],test_y[step*BATCH_SIZE:(step+1)*BATCH_SIZE,:]
    #print(b_x.shape,b_y.shape)
    _, loss_ = sess.run([train_op, loss], {tf_x: b_x, tf_y: b_y})
    #if step % 50 == 0:
        #accuracy_, flat_representation = sess.run([accuracy, flat], {tf_x: test_x, tf_y: test_y})
        #print('Step:', step, '| train loss: %.4f' % loss_, '| test accuracy: %.2f' % accuracy_)

saver.save(sess, 'cnnnet/j2men', write_meta_graph=False)  # meta_graph is not recommended
# print 1 predictions from test data
test_output = sess.run(output, {tf_x: test_x[:10]})
pred_y =test_output
print(pred_y, 'prediction number')
print(test_y[:10], 'real number')
