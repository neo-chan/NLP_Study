#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   layers.py    
@Contact :   cxbwater@163.com
@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2019/11/28 上午7:27   cxb      1.0         None
'''

# import lib
import tensorflow as tf

class Encoder(tf.keras.Model):
    def __int__(self,vocab_size,embedding_dim,embeddings,encoder_units,batch_size):
        '''
        Encoder初始化
        :param vocab_size: 词表大小
        :param embedding_dim: 词向量维度
        :param encoder_units: encoder的单元数
        :param batch_size: 本次训练的数据大小
        :param embeddings: 词向量的矩阵
        :return:
        '''
        super(Encoder,self).__init__()
        self.encoder_units=encoder_units
        self.batch_size=batch_size
        self.embedding=tf.keras.layers.Embedding(vocab_size,embedding_dim,weights=[embeddings],trainable=False)
        self.gru=tf.keras.layers.GRU(self.encoder_units,
                                     return_sequence=True,
                                     return_state=True,
                                     recurrent_initializer='glorot_uniform')



    def call(self, x, hidden):
        '''
        调用encoder
        :param x: 输入
        :param hidden: gru初始状态单元
        :return:
        '''
        # embedding层 将输入的词的index转换成词向量
        x=self.embedding(x)
        output , state=self.gru(x,initial_state=hidden)
        return output, state


    def initialize_hidden_state(self):
        '''
        初始化gru单元
        :return:
        '''
        return tf.zeros(self.batch_size, self.encoder_units)


class Decoder(tf.keras.Model):
    def __int__(self,vocab_size, embedding_dim,embeddings, decoder_units, batch_size):
        '''
        Decoder初始化
        :param vocab_size: 词表大小
        :param embedding_dim: 词向量维度
        :param encoder_units: encoder的单元数
        :param batch_size: 本次训练的数据大小
        :return:
        '''
        super(Decoder,self).__init__()
        self.decoder_units=decoder_units
        self.batch_size=batch_size
        self.embedding=tf.keras.layers.Embedding(vocab_size,embedding_dim,weights=[embeddings],trainable=False)
        self.gru=tf.keras.layers.GRU(self.decoder_units,
                                     return_sequence=True,
                                     return_state=True,
                                     recurrent_initializer='glorot_uniform')
        # 全连接层
        self.fc=tf.keras.layers.Dense(vocab_size)

        # 加入attention
        self.attention=BahdanauAttention(decoder_units)

    def call(self,x,hidden,enc_output):
        '''
        decoder调用
        :param x: 输入
        :param hidden:encoder的隐藏层 用于attention
        :param enc_output: 输出
        :return:
        '''
        # enc_output shape == (batch_size, max_length, hidden_size)
        context_vector, attention_weights = self.attention(hidden, enc_output)

        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        x=self.embedding(x)

        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        # passing the concatenated vector to the GRU
        output, state=self.gru(x)

        # output shape == (batch_size * 1, hidden_size)
        output = tf.reshape(output, (-1, output.shape[2]))

        # output shape == (batch_size, vocab)
        x=self.fc(output)
        return x, state,attention_weights

class BahdanauAttention():
    def __int__(self, units):
        super(BahdanauAttention,self).__init__()
        self.W1=tf.keras.layers.Dense(units)
        self.W2=tf.keras.layers.Dense(units)
        self.V=tf.keras.layers.Dense(1)

    def call(self,query, values):
        # hidden shape == (batch_size, hidden size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden size)
        # we are doing this to perform addition to calculate the score
        hidden_with_time_axis=tf.expand_dims(query,1)

        # score shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying score to self.V
        # the shape of the tensor before applying self.V is (batch_size, max_length, units)
        score=self.V(tf.nn.tanh(self.W1(values)+self.W2(hidden_with_time_axis)))

        # attention_weights shape == (batch_size, max_length, 1)
        attention_weights=tf.nn.softmax(score,axis=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights*values
        context_vector = tf.reduce_sum(context_vector,axis=1)

        return context_vector, attention_weights