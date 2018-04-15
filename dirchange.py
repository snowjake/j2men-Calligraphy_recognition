import os
import os.path
import shutil
import configparser
 
#读入指定目录并转换为绝对路径
rootdir = "images/train"
rootdir = os.path.abspath(rootdir)
print('absolute root path:\n*** ' + rootdir + ' ***')
 
         
#后修改目录名，这里注意topdown参数。
#topdown决定遍历的顺序，如果topdown为True，则先列举top下的目录，然后是目录的目录，依次类推；
#反之，则先递归列举出最深层的子目录，然后是其兄弟目录，然后父目录。
#我们需要先修改深层的子目录
i =110

#生成config对象  
conf = configparser.ConfigParser()  
#增加新的section  
conf.add_section('j2men')  
  
for parent, dirnames, filenames in os.walk(rootdir, topdown=False):
    for dirname in dirnames:
        pathdir = os.path.join(parent, dirname)
        print(pathdir + ' --> ' + str(i))
        os.rename(pathdir, str(i))
        
        #写配置文件  
        #更新指定section，option的值  
        conf.set("j2men",str(i), dirname)
        #写回配置文件  
        conf.write(open("j2men.cfg", "w"))
        i+=1
