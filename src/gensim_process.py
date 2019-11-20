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
from src.utils.file_path import merged_seg_path,word2vec_model_path
from gensim.models.word2vec import LineSentence
from gensim.models import word2vec
import gensim
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def gensim_process(merged_seg_path,word2vec_model_path):
    model=word2vec.Word2Vec(LineSentence(merged_seg_path),workers=4,min_count=5,size=200)
    model.save(word2vec_model_path)
    #print(model.wv.most_similar(['奔驰'],topn=10))

if __name__ == '__main__':
    gensim_process(merged_seg_path, word2vec_model_path)