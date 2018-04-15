import os
import numpy
from PIL import Image   #导入Image模块
from pylab import *     #导入savetxt模块
#import glob


#以下代码看可以读取文件夹下所有文件
# def getAllImages(folder):
#     assert os.path.exists(folder)
#     assert os.path.isdir(folder)
#     imageList = os.listdir(folder)
#     imageList = [os.path.abspath(item) for item in imageList if os.path.isfile(os.path.join(folder, item))]
#     return imageList

# print getAllImages(r"D:\\test")


def convertjpg(jpgfile,width=128,height=128):  
    img=Image.open(jpgfile)  
    try:  
        new_img=img.resize((width,height),Image.BILINEAR)     
        return new_img
    except Exception as e:  
        print(e)  

def get_imlist(path):   #此函数读取特定文件夹下的jpg格式图像
    return [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.jpg')]

rootdir = "images"
rootdir = os.path.abspath(rootdir)
for parent, dirnames, filenames in os.walk(rootdir, topdown=False):
    for dirname in dirnames:
        print(dirname)
        c=get_imlist(r"images/"+dirname)    #r""是防止字符串转译
        print (c)     #这里以list形式输出jpg格式的所有图像（带路径）
        d=len(c)    #这可以以输出图像个数

        data=numpy.empty((d,128*128)) #建立d*（128*128）的矩阵
        #for jpgfile in glob.glob("images\*.jpg"):  
            #img=convertjpg(jpgfile)
        while d>0:
            img=convertjpg(c[d-1])  #打开图像
            #img_ndarray=numpy.asarray(img)
            img_ndarray=numpy.asarray(img,dtype='float64')/256  #将图像转化为数组并将像素转化到0-1之间
            data[d-1]=numpy.ndarray.flatten(img_ndarray)    #将图像的矩阵形式转化为一维数组保存到data中
            d=d-1
        print (data)
        f=open('j2men.csv','ab')
        for d in data:
            A=numpy.array(d).reshape(1,128*128)   #将一维数组转化为1,128*128矩阵
            #A = np.concatenate((A,[p_])) # 先将p_变成list形式进行拼接，注意输入为一个tuple
            #A = np.append(A,1)
            #print A
            f.write(bytes(dirname+",", encoding = "utf8")) #在前面追加一个字符
            savetxt(f,A,fmt="%.0f",delimiter=',') #将矩阵保存到文件中
