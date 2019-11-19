# coding=utf-8
import os
import pathlib

#os.path.abspath(__file__)当前py文件的绝对路径
root=pathlib.Path(os.path.abspath(__file__)).parent.parent.parent
_data_source='datasource'
_stop_words='stopwords'
train_data_path=os.path.join(root,_data_source,'AutoMaster_TestSet.csv')
test_data_path=os.path.join(root,_data_source,'AutoMaster_TrainSet.csv')
stop_word_path=os.path.join(root,_data_source,_stop_words,'哈工大停用词表.txt')

user_dict=os.path.join(root,_data_source,'user_dict.txt')

train_seg_path=os.path.join(root,_data_source)