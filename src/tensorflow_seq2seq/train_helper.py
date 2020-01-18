# -*- coding:utf-8 -*-
import tensorflow as tf
from src.tensorflow_seq2seq.batcher import train_batch_generator
from src.tensorflow_seq2seq.seq2seq_model import Seq2Seq
from src.utils.config import save_wv_model_path
from src.utils.gpu_utils import config_gpu
import time


def train_model(model, vocab, params, checkpoint_manager):
    epochs = params['epochs']
    batch_size = params['batch_size']

    pad_index = vocab.word2id[vocab.PAD_TOKEN]
    start_index = vocab.word2id[vocab.START_DECODING]

    # 计算vocab size
    params['vocab_size'] = vocab.count
    #优化器选择
    optimizer = tf.keras.optimizers.Adam(name='Adam', learning_rate=0.001)
    #损失函数 交叉熵
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

    # 定义损失函数
    def loss_function(real, pred):
        #去除pad的影响
        mask = tf.math.logical_not(tf.math.equal(real, pad_index))
        loss_ = loss_object(real, pred)
        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask

        return tf.reduce_mean(loss_)

    # 训练
    @tf.function
    def train_step(enc_inp, dec_target):
        batch_loss = 0
        with tf.GradientTape() as tape:
            enc_output, enc_hidden = model.call_encoder(enc_inp)
            # 第一个decoder输入 开始标签
            dec_input = tf.expand_dims([start_index] * batch_size, 1)
            # 第一个隐藏层输入
            dec_hidden = enc_hidden
            # 逐个预测序列,调用Seq2Seq中的call函数
            predictions, _ = model(dec_input, dec_hidden, enc_output, dec_target)
            # 计算当前batch的loss
            batch_loss = loss_function(dec_target[:, 1:], predictions)
            # 获取模型中的训练参数
            variables = model.encoder.trainable_variables + model.decoder.trainable_variables + model.attention.trainable_variables
            # 计算梯度
            gradients = tape.gradient(batch_loss, variables)
            # 将梯度应用到模型参数中 zip：将一个梯度与一个参数打包成一个元组
            optimizer.apply_gradients(zip(gradients, variables))
            #返回当前batch的loss
            return batch_loss

    dataset, steps_per_epoch = train_batch_generator(batch_size)

    for epoch in range(epochs):
        start = time.time()
        total_loss = 0

        for (batch, (inputs, target)) in enumerate(dataset.take(steps_per_epoch)):
            batch_loss = train_step(inputs, target)
            total_loss += batch_loss

            if batch % 5 == 0:
                print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                             batch,
                                                             batch_loss.numpy()))
        # saving (checkpoint) the model every 2 epochs
        if (epoch + 1) % 1 == 0:
            ckpt_save_path = checkpoint_manager.save()
            print('Saving checkpoint for epoch {} at {}'.format(epoch + 1,
                                                                ckpt_save_path))

        print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                            total_loss / steps_per_epoch))
        print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
