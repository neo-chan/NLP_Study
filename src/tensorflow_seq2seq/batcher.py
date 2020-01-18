# -*- coding:utf-8 -*-

from src.utils.data_process import load_train_dataset, load_test_dataset
import tensorflow as tf


def train_batch_generator(batch_size, max_enc_len=200, max_dec_len=50, sample_sum=None):
    # 加载数据集
    train_X, train_Y = load_train_dataset(max_enc_len, max_dec_len)
    if sample_sum:
        train_X = train_Y[:sample_sum]
        train_Y = train_Y[:sample_sum]
    #根据训练集生成tf需要的数据格式并调用shuffle打乱
    dataset = tf.data.Dataset.from_tensor_slices((train_X, train_Y)).shuffle(len(train_X))
    #根据batch_size划分成不同的batch
    dataset = dataset.batch(batch_size, drop_remainder=True)
    # 一个batch的数据大小
    steps_per_epoch = len(train_X) // batch_size
    return dataset, steps_per_epoch


def beam_test_batch_generator(beam_size, max_enc_len=200):
    # 加载数据集
    test_X = load_test_dataset(max_enc_len)
    # 利用生成器，每一次获取beam_size大小的数据
    for row in test_X:
        beam_search_data = tf.convert_to_tensor([row for i in range(beam_size)])
        yield beam_search_data


if __name__ == '__main__':
    beam_test_batch_generator(4)
