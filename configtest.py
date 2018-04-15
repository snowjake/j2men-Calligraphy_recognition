# -* - coding: UTF-8 -* -  

import configparser
#生成config对象  
conf = configparser.ConfigParser()  
  
#写配置文件  
#增加新的section  
conf.add_section('j2men')  
#更新指定section，option的值  
conf.set("j2men", "b_key3", "new-$r")  
#写入指定section增加新option和值  
conf.set("j2men", "b_newkey", "new-value")  
conf.set('j2men', 'new_key', 'new_value')  
#写回配置文件  
conf.write(open("j2men.cfg", "w"))
