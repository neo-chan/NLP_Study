#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   __init__.py
@Contact :   cxbwater@163.com
@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2019/11/19 下午9:20   cxb      1.0         None
'''

# import lib
import pandas as pd
import jieba
import re
import jieba.posseg as pseg
import sys
sys.path.append(r"..")
from src.utils.file_path import test_data_path,train_data_path,stop_word_path, user_dict
from src.utils.file_path import test_seg_path, train_seg_path, merged_seg_path
from src.utils.multiprocessing_utils import parallelize

jieba.load_userdict(user_dict)

def clear(comment):
    comment = comment.strip()
    comment = comment.replace('、', '')
    comment = comment.replace('，', '。')
    comment = comment.replace('《', '。')
    comment = comment.replace('》', '。')
    comment = comment.replace('～', '')
    comment = comment.replace('…', '')
    comment = comment.replace('\r', '')
    comment = comment.replace('\t', ' ')
    comment = comment.replace('\f', ' ')
    comment = comment.replace('/', '')
    comment = comment.replace('、', ' ')
    comment = comment.replace('/', '')
    comment = comment.replace('。', '')
    comment = comment.replace('（', '')
    comment = comment.replace('"', '')
    comment = comment.replace('“', '')
    comment = comment.replace('）', '')
    comment = comment.replace('_', '')
    comment = comment.replace('-', '')
    comment = comment.replace('?', ' ')
    comment = comment.replace('？', ' ')
    comment = comment.replace('|', '')
    comment = comment.replace('：', '')
    comment = comment.replace('！', '')
    comment = comment.replace('!', '')
    comment = comment.replace('[语音]', '')
    comment = comment.replace('[图片]', '')
    comment = comment.replace('技师说', '')
    comment = comment.replace('车主说', '')
    return comment

def load_dataset(train_data_path,test_data_path):
    '''
    数据数据集
    :param train_data_path:训练集路径
    :param test_data_path: 测试集路径
    :return:
    '''
    # 读取数据集
    train_data=pd.read_csv(train_data_path)
    test_data=pd.read_csv(test_data_path)
    return train_data,test_data

def clean_sentence(sentence):
    '''
    用正则表达式去除特殊符号或者用前面的clear函数做替换
    :param sentence: 待处理的字符串
    :return: 过滤特殊字符后的字符串
    '''
    if isinstance(sentence,str):
        return re.sub(r'[\s+\-\|\!\/\[\]\{\}_,.$%^*(+\"\')]+|[:：+——()?【】“”！，。？、~@#￥%……&*（）]+|车主说|技师说|语音|图片|你好|您好',
                      '', sentence)
    else:
        return ''

def load_stop_words(stop_word_path):
    '''
    加载停用词
    :param stop_word_path:停用词路径
    :return: 停用词列表
    '''
    #打开文件
    with open(stop_word_path,'r',encoding='utf-8') as f:
        #读取所有行
        stop_words=f.readlines()
        #去除每一个停用词前后的空格，换行符
        stop_words=[stop_word.strip() for stop_word in stop_words]
        return stop_words

def filter_stopwords(words):
    '''
    过滤停用词
    :param words:切好词的列表[word1，word2，...]
    :return: 过滤停用词后的词列表
    '''
    return [word for word in words if word not in stop_words]

def sentence_process(sentence):
    '''
    预处理模块
    :param sentence:待处理的字符串
    :return: 处理后的字符串
    '''
    # 1.清除无用词
    sentence=clean_sentence(sentence)
    # 2.切词,默认精确模式，全模式为cut_all=True
    words=jieba.cut(sentence)
    # 3.过滤停用词
    words=filter_stopwords(words)
    # 4.拼接成一个空格分隔的字符串
    return ' '.join(words)

def data_frame_process(df):
    '''
    对数据集中的每一帧进行处理
    :param df: 数据集
    :return: 处理后的数据集
    '''
    #批量处理测试集和训练集中的几列
    for each_col in ['Brand', 'Model', 'Question','Dialogue']:
        df[each_col]=df[each_col].apply(sentence_process)
    # 对只有训练集中有的Report进行处理
    if'Report' in df.columns:
        df['Report']=df['Report'].apply(sentence_process)
    return df

def data_generate(train_data_path,test_data_path):
    '''
    数据加载和预处理
    :param train_data_path:训练集路径
    :param test_data_path: 测试集路径
    :return: 训练数据，测试数据，合并的数据
    '''
    # 1.加载数据
    train_df,test_df=load_dataset(train_data_path,test_data_path)
    print('train data size {},test data size {}'.format(len(train_df), len(test_df)))
    # 2.空值处理,删除subset中的列字段为空的行
    train_df.dropna(subset=['Question', 'Dialogue','Report'], how='any',inplace=True)
    test_df.dropna(subset=['Question', 'Dialogue'], how='any',inplace=True)
    # 3.多进程批量处理数据
    train_df=parallelize(train_df,data_frame_process)
    test_df=parallelize(test_df,data_frame_process)
    # 4.保存处理好的训练集 测试集
    train_df.to_csv(train_seg_path, index=None, header=True)
    test_df.to_csv(test_seg_path, index=None, header=True)
    # 5.合并训练集和测试集,将数据集中一行的某几个列值合并后放到merged列，axis=1 apply function to each row, 0 apply function to each column
    train_df['merged']=train_df[['Question', 'Dialogue', 'Report']].apply(lambda x: ' '.join(x), axis=1)
    test_df['merged']=test_df[['Question', 'Dialogue']].apply(lambda x: ' '.join(x), axis=1)
    merged_df=pd.concat([train_df['merged'],test_df['merged']],axis=0)
    merged_df.to_csv(merged_seg_path, index=None, header=True)
    print('train data size {},test data size {},merged_df data size {}'.format(len(train_df), len(test_df),
                                                                               len(merged_df)))
    return train_df,test_df,merged_df




if __name__ == '__main__':
    stop_words=load_stop_words(stop_word_path)
    data_generate(train_data_path,test_data_path)




