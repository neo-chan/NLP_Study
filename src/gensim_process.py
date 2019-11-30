#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   gensim.py    
@Contact :   cxbwater@163.com
@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2019/11/20 上午8:00   cxb      1.0         None
'''

# import lib
import pandas as pd
import logging
import numpy as np
from src.utils.file_path import merged_seg_path,word2vec_model_path,fasttext_model_path,embedding_matrix_path,embedding_dim
from gensim.models.word2vec import LineSentence
from gensim.models import word2vec
from gensim.models import FastText
import gensim

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def gensim_process(merged_seg_path,word2vec_model_path,fasttext_model_path):
    '''
    生成词向量模型
    :param merged_seg_path:词典的路径
    :param word2vec_model_path: 模型保存的路径
    :return:
    '''
    #word2vec训练词向量
    word2vec_model=word2vec.Word2Vec(LineSentence(merged_seg_path),workers=4,min_count=5,size=embedding_dim)
    word2vec_model.save(word2vec_model_path)
    #fasttext训练词向量
    fasttext_model=FastText(LineSentence(merged_seg_path),workers=4,min_count=5,size=embedding_dim)
    fasttext_model.save(fasttext_model_path)
    #print(word2vec_model.wv.most_similar(['奔驰'],topn=10))

def build_embedding_matrix(vocab,model_path):
    '''
    建立embedding_matrix
    :param vocab: 词表
    :param model_path: 词向量保存的路径
    :return:embedding_matrix
    '''
    # 加载词向量
    model=word2vec.Word2Vec.load(model_path)
    #构建初始矩阵
    embedding_matrix=np.zeros((len(vocab),embedding_dim))
    #遍历填充矩阵
    for i,word in enumerate(vocab):
        try:
            embedding_vector=model[word]
            embedding_matrix[i]=embedding_vector
        except:
            continue
    #保存矩阵
    np.savetxt(embedding_matrix_path,embedding_matrix,fmt="%s",delimiter=",")
    return embedding_matrix


if __name__ == '__main__':
    gensim_process(merged_seg_path, word2vec_model_path,fasttext_model_path)
    build_embedding_matrix(merged_seg_path,word2vec_model_path)
