import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import loadImage3 as li
import math
import configparser

tf.set_random_seed(1)
np.random.seed(1)

#new_data = pd.read_csv("j2mentest.csv")
#new_data = new_data.values.astype(np.float32)
#np.random.shuffle(new_data)

#test_x = new_data

def reload():
    print('This is reload')
    # build entire net again and restore
    tf_x = tf.placeholder(tf.float32, [None, 128*128]) / 255.
    image = tf.reshape(tf_x, [-1, 128, 128, 1])              # (batch, height, width, channel)
    #tf_y = tf.placeholder(tf.int32, [None, 1])            # input y

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
   # train_op = tf.train.AdamOptimizer(LR).minimize(loss)
    cf = configparser.ConfigParser()
    cf.read("j2men.cfg")
    sess = tf.Session()
    # don't need to initialize variables, just restoring trained variables
    saver = tf.train.Saver()  # define a saver for saving and restoring
    saver.restore(sess, 'cnnnet/j2men')
    c=li.get_imlist(r"test1")    #r""是防止字符串转译
    d=len(c)    #这可以以输出图像个数
    data=np.empty((d,128*128)) #建立d*（128*128）的矩阵
    #for jpgfile in glob.glob("images\*.jpg"):  
        #img=convertjpg(jpgfile)
    f=open('result.csv','ab')
    while d>0:
        img=li.convertjpg(c[d-1])  #打开图像
        img_ndarray=np.asarray(img,dtype='float64')/256  #将图像转化为数组并将像素转化到0-1之间
        data[d-1]=np.ndarray.flatten(img_ndarray)    #将图像的矩阵形式转化为一维数组保存到data中
        
        A=np.array(data[d-1]).reshape(1,128*128)   #将一维数组转化为1,128*128矩阵
        test_output = sess.run(output, {tf_x: A})
        pred_y = int(round(test_output[0,0]))
        if pred_y<112:
            pred_y=112
        print(c[d-1],'prediction number',pred_y,cf.get("j2men", str(pred_y)))
        
        f.write(bytes(str(c[d-1]).replace("test1\\","")+","+cf.get("j2men", str(pred_y-2))+cf.get("j2men", str(pred_y-1))+cf.get("j2men", str(pred_y))+cf.get("j2men", str(pred_y+1))+cf.get("j2men", str(pred_y+2))+"\r\n", encoding = "utf8")) #输出结果
        #print(pred_y, 'prediction number')
        d=d-1


# destroy previous net
tf.reset_default_graph()

reload()
