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
from src.utils.file_path import embedding_matrix_path
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def get_vocab_from_model(wv_model_path):
    '''
    从保存的词向量模型中获取字典
    :param vocab:
    :return: word_index,index_word
    '''
    wv_model=Word2Vec.load(wv_model_path)
    vocab={word: index for index, word in enumerate(wv_model.wv.index2word)}
    reverse_vocab={index: word for index,word in enumerate(wv_model.wv.index2word)}
    return vocab,reverse_vocab

def get_pad_vocab(vocab):
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

def save_embedding_matrix(w2v_model):
    '''
    获取embedding矩阵
    :param w2v_model:词向量模型
    :return: embedding矩阵
    '''
    vocab_size=len(w2v_model.wv.vocab)
    embedding_dim=len(w2v_model.wv['<START>'])
    print('vocab_size, embedding_dim:', vocab_size, embedding_dim)
    embedding_matrix=np.zeros(vocab_size,embedding_dim)
    for i in range(vocab_size):
        embedding_matrix[i,:]=w2v_model.wv[w2v_model.wv.index2word[i]]
        embedding_matrix=embedding_matrix.astype('float32')
    assert embedding_matrix.shape==(vocab_size,embedding_dim)
    np.save(embedding_matrix_path,embedding_matrix,fmt='%0.8f')
    print('embedding matrix extracted')
    return embedding_matrix

def get_embedding_matrix(save_wv_model_path):
    '''
    从词向量模型中获取embeding矩阵
    :param save_wv_model_path:保存的词向量路径
    :return: embedding矩阵
    '''
    wv_model=Word2Vec(save_wv_model_path)
    embedding_matrix=wv_model.wv.vectors
    return embedding_matrix


def load_embedding_matrix():
    """
    加载 embedding_matrix_path
    """
    return np.load(embedding_matrix_path + '.npy')