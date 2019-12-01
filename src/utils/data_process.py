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
import numpy as np
import jieba
import re
import jieba.posseg as pseg
import sys
sys.path.append(r"..")
from src.utils.file_path import test_data_path,train_data_path,stop_word_path, user_dict
from src.utils.file_path import test_seg_path, train_seg_path, merged_seg_path, wv_train_epochs,train_x_seg_path,\
    train_y_seg_path,test_x_seg_path,train_x_pad_path,train_y_pad_path,test_x_pad_path,word2vec_model_path
from src.utils.multiprocessing_utils import parallelize
from gensim.models.word2vec import LineSentence,Word2Vec

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

jieba.load_userdict(user_dict)

def load_dataset():
    '''
    数据数据集
    :param train_data_path:训练集路径
    :param test_data_path: 测试集路径
    :return:
    '''
    # 读取数据集,将csv文件的第一列命名
    train_X=pd.read_csv(train_x_pad_path,header=None).rename(columns={0: 'X'})
    train_Y=pd.read_csv(train_y_pad_path,header=None).rename(coiumns={0: 'Y'})
    test_X=pd.read_csv(test_x_pad_path,header=None).rename(columns={0: 'X'})
    # 获取保存的词向量模型
    wv_moodel=Word2Vec.load(word2vec_model_path)
    return train_X['X'],train_Y['Y'],test_X['X'],wv_moodel

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

def pad_process(sentence, max_len, vocab):
    '''
    < START > < END > < PAD > < UNK > max_lens
    '''
    # 0.按空格统计切分出词
    words = sentence.strip().split(' ')
    # 1. 截取规定长度的词数
    words = words[:max_len]
    # 2. 填充< unk > ,判断是否在vocab中, 不在填充 < unk >
    sentence = [word if word in vocab else '<UNK>' for word in words]
    # 3. 填充< start > < end >
    sentence = ['<START>'] + sentence + ['<END>']
    # 4. 判断长度，填充　< pad >
    sentence = sentence + ['<PAD>'] * (max_len + 2 - len(words))
    return ' '.join(sentence)

def get_max_len(data):
    '''
    获得合适的最大长度值
    :param data: 待统计的数据  train_df['Question']
    :return: 最大长度值
    '''
    max_lens = data.apply(lambda x: x.count(' '))
    return int(np.mean(max_lens) + 2 * np.std(max_lens))


def data_generate(train_data_path,test_data_path):
    '''
    数据加载和预处理
    :param train_data_path:训练集路径
    :param test_data_path: 测试集路径
    :return: 训练数据，测试数据，合并的数据
    '''
    # 1.加载数据
    train_df=pd.read_csv(train_data_path)
    test_df=pd.read_csv(test_data_path)
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
    # 6.保存合并数据
    merged_df.to_csv(merged_seg_path, index=None, header=True)
    print('train data size {},test data size {},merged_df data size {}'.format(len(train_df), len(test_df),
                                                                               len(merged_df)))
    # 7. 训练词向量
    print('start build w2v model')
    wv_model = Word2Vec(LineSentence(merged_seg_path), size=300, negative=5, workers=4, iter=wv_train_epochs, window=3,
                        min_count=5)
    # 8. 分离数据和标签
    train_df['X'] = train_df[['Question', 'Dialogue']].apply(lambda x: ' '.join(x), axis=1)
    test_df['X'] = test_df[['Question', 'Dialogue']].apply(lambda x: ' '.join(x), axis=1)

    # 9. 填充开始结束符号,未知词填充 oov, 长度填充
    # 使用GenSim训练得出的vocab
    vocab = wv_model.wv.vocab

    # 训练集X处理
    # 获取适当的最大长度
    train_x_max_len = get_max_len(train_df['X'])
    test_X_max_len = get_max_len(test_df['X'])
    X_max_len = max(train_x_max_len, test_X_max_len)
    train_df['X'] = train_df['X'].apply(lambda x: pad_process(x, X_max_len, vocab))

    # 测试集X处理
    # 获取适当的最大长度
    test_df['X'] = test_df['X'].apply(lambda x: pad_process(x, X_max_len, vocab))

    # 训练集Y处理
    # 获取适当的最大长度
    train_y_max_len = get_max_len(train_df['Report'])
    train_df['Y'] = train_df['Report'].apply(lambda x: pad_process(x, train_y_max_len, vocab))

    # 10. 保存pad oov处理后的,数据和标签
    train_df['X'].to_csv(train_x_pad_path, index=None, header=False)
    train_df['Y'].to_csv(train_y_pad_path, index=None, header=False)
    test_df['X'].to_csv(test_x_pad_path, index=None, header=False)

    # 11. 词向量再次训练
    print('start retrain w2v model')
    wv_model.build_vocab(LineSentence(train_x_pad_path), update=True)
    wv_model.train(LineSentence(train_x_pad_path), epochs=wv_train_epochs, total_examples=wv_model.corpus_count)
    print('1/3')
    wv_model.build_vocab(LineSentence(train_y_pad_path), update=True)
    wv_model.train(LineSentence(train_y_pad_path), epochs=wv_train_epochs, total_examples=wv_model.corpus_count)
    print('2/3')
    wv_model.build_vocab(LineSentence(test_x_pad_path), update=True)
    wv_model.train(LineSentence(test_x_pad_path), epochs=wv_train_epochs, total_examples=wv_model.corpus_count)

    # 保存词向量模型
    wv_model.save(word2vec_model_path)
    print('finish retrain w2v model')
    print('final w2v_model has vocabulary of ', len(wv_model.wv.vocab))
    return train_df['X'], train_df['Y'], test_df['X'], wv_model




if __name__ == '__main__':
    stop_words=load_stop_words(stop_word_path)
    data_generate(train_data_path,test_data_path)




