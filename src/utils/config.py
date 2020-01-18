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
_results_dir="results"
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
#word2vec训练模型的路径
save_wv_model_path=os.path.join(root,_data_source,'word2vec.model')
#fasttext训练模型的路径
fasttext_model_path=os.path.join(root,_data_source,'fasttext.model')
#embedding_matrix保存路径
embedding_matrix_path=os.path.join(root,_data_source,'embedding_matrix')
# 数据标签分离
train_x_seg_path = os.path.join(root, _data_source, 'train_X_seg_data.csv')
train_y_seg_path = os.path.join(root, _data_source, 'train_Y_seg_data.csv')
test_x_seg_path = os.path.join(root, _data_source, 'test_X_seg_data.csv')
val_x_seg_path = os.path.join(root, _data_source, 'val_X_seg_data.csv')
val_y_seg_path = os.path.join(root, _data_source, 'val_Y_seg_data.csv')
# pad oov处理后的数据
train_x_pad_path = os.path.join(root, _data_source, 'train_X_pad_data.csv')
train_y_pad_path = os.path.join(root, _data_source, 'train_Y_pad_data.csv')
test_x_pad_path = os.path.join(root, _data_source, 'test_X_pad_data.csv')

# numpy 转换后的数据
train_x_path = os.path.join(root, _data_source, 'train_X')
train_y_path = os.path.join(root, _data_source, 'train_Y')
test_x_path = os.path.join(root, _data_source, 'test_X')

# 词典路径
vocab_path = os.path.join(root, _data_source, 'vocab.txt')
reverse_vocab_path = os.path.join(root, _data_source, 'reverse_vocab.txt')

# 模型保存文件夹
checkpoint_dir = os.path.join(root, _data_source, 'checkpoints', 'training_checkpoints_mask_loss_dim500_seq')

checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')

# 结果保存文件夹
save_result_dir = os.path.join(root, _results_dir)

# 词向量训练轮数
wv_train_epochs = 1
#词向量维度
embedding_dim=300

sample_total = 82871

batch_size = 3


epochs = 5

vocab_size = 32201