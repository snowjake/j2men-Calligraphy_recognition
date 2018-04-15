import tensorflow as tf
import pandas as pd
import numpy as np
import csv
import os
def get_imlist(path):   #此函数读取特定文件夹下的bmp格式图像
    return [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.jpg')]

def convert2onehot(data):
    return pd.get_dummies(data)

with tf.Session() as sess:
    c=get_imlist(r"images")    #r""是防止字符串转译
    print (c)     #这里以list形式输出bmp格式的所有图像（带路径）
    d=len(c)    #这可以以输出图像个数
    f=open('image.csv','ab')
    while d>0:
        imagerawdata=tf.gfile.FastGFile(c[d-1], 'rb').read()
        img_data = tf.image.decode_jpeg(imagerawdata)
        #print(img_data.eval())
        
        img_data = tf.image.convert_image_dtype(img_data, dtype=tf.float32)
        print(img_data.eval().shape)
        resultarray=img_data.eval().reshape(1,-1)[:, :14750]
        resultarray=resultarray+[110]
        print(resultarray.shape)
        np.savetxt(f, resultarray,delimiter=',')
        d=d-1


    #fileobj=open('test.csv','wb')
    #writer = csv.writer(fileobj)
    #writer.writerow(img_data.eval().flatten().tolist())

    #data = pd.read_csv("image.csv")
    #new_data = convert2onehot(data)
    #print(data.head())
    #print("\nNum of data: ", len(data), "\n") 
    #print("\n", new_data.head(2))
    #new_data.to_csv("image_onehot.csv", index=False)
