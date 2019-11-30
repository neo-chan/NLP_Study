#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   wv_loader.py    
@Contact :   cxbwater@163.com
@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2019/11/29 上午8:10   cxb      1.0         None
'''

# import lib
from gensim.models.word2vec import LineSentence, Word2Vec
import numpy as np
import codecs
# 引入日志配置
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def build_pad_vocab(vocab):
    '''
    加入< START > < END > < PAD > < UNK >后的词典
    :param vocab:
    :return: 处理后的词典
    '''
    start_token=u"<START>"
    end_token=u"<END>"
    unk_token=u"<UNK>"

    #按词典索引排序
    vocab=sorted([(vocab[i].index, i) for i in vocab])
    #排序后的词表
    sorted_words=[word for index,word in vocab]
    #拼接标志位的词
    sorted_words=[start_token,end_token,unk_token]+sorted_words

    #构建新索引表
    vocab={index:word for index,word in enumerate(sorted_words)}
    rever_vocab={word:index for index,word in enumerate(sorted_words)}

    return vocab,rever_vocab

