import numpy as np
import tensorflow as tf
import pandas as pd
#import loadImage

new_data = pd.read_csv("num1.csv") #通过pandas读取csv中的数据
print(len(new_data))
# prepare training data
new_data = new_data.values.astype(np.float32)       # change to numpy array and float32
np.random.shuffle(new_data)         #随机打乱数据
sep = int(0.7*len(new_data))        #设置训练数据的百分比
print(len(new_data))
train_data = new_data[:sep]    
print(len(train_data))                     # training data (70%)
test_data = new_data[sep:]                          # test data (30%)


# build network
tf_input = tf.placeholder(tf.float32, [None, 16385], "input")   #用placeholder将输入的张量hold住
tfx = tf_input[:, 1:]   #输入的是从第二位开始到结束
tfy = tf_input[:, :1]   #输出时第一位数据

l1 = tf.layers.dense(tfx, 128, tf.nn.relu, name="l1")   #第一层全连接网络
l2 = tf.layers.dense(l1, 128, tf.nn.relu, name="l2")   #第二层全连接网络
out = tf.layers.dense(l2, 1, name="l3")   #第三层全连接网络 用来输出1位数据
prediction = tf.nn.softmax(out, name="pred")  #使用softmax计算每个预测结果的概率值

loss = tf.losses.softmax_cross_entropy(onehot_labels=tfy, logits=out) #计算损失率
accuracy = tf.metrics.accuracy(          # return (acc, update_op), and create 2 local variables 
    labels=tf.argmax(tfy, axis=1), predictions=tf.argmax(out, axis=1),)[1]
opt = tf.train.GradientDescentOptimizer(learning_rate=0.1)
train_op = opt.minimize(loss)  #上面创建了优化器 这里使损失率最小

sess = tf.Session()
sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())) #初始化变量

# training
accuracies, steps = [], []
for t in range(300):
    # training
    batch_index = np.random.randint(len(train_data), size=32)  #训练的batch size
    #print(batch_index)
    sess.run(train_op, {tf_input: train_data[batch_index]})  #采用batch data训练

    if t % 50 == 0:
        # testing
        acc_, pred_, loss_ = sess.run([accuracy, prediction, loss], {tf_input: test_data})
        accuracies.append(acc_)
        steps.append(t)
        print("pred_" % pred_)
        print("Step: %i" % t,"| Accurate: %.2f" % acc_,"| Loss: %.2f" % loss_,)


