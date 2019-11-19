#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   __init__.py
@Contact :   cxbwater@163.com
@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2019/11/18 下午7:50   cxb      1.0         None
'''

# import lib
import os
import pathlib

#os.path.abspath(__file__)当前py文件的绝对路径
root=pathlib.Path(os.path.abspath(__file__)).parent.parent.parent
_data_source='datasource'
_stop_words='stopwords'
#训练数据路径
train_data_path=os.path.join(root,_data_source,'AutoMaster_TrainSet.csv')
#测试数据路径
test_data_path=os.path.join(root,_data_source,'AutoMaster_TestSet.csv')
#停用词路径
stop_word_path=os.path.join(root,_data_source,_stop_words,'哈工大停用词表.txt')
#自定义词典，建议根据使用数据制作，本次数据可以使用brand，以及model制作
user_dict=os.path.join(root,_data_source,'user_dict.txt')
#预处理后的训练数据
train_seg_path=os.path.join(root,_data_source, 'train_seg_data.csv')
#预处理后的测试数据
test_seg_path=os.path.join(root,_data_source, 'test_seg_data.csv')
#合并训练集和测试集的数据
merged_seg_path=os.path.join(root,_data_source,'merged_train_test_seg_data.csv')


