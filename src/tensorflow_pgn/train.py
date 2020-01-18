# -*- coding:utf-8 -*-


from src.utils.gpu_utils import config_gpu

import tensorflow as tf

from src.tensorflow_pgn.batcher import batcher
from src.tensorflow_pgn.pgn_model import PGN
from src.tensorflow_pgn.train_helper import train_model
from src.utils.params_utils import get_params
from src.utils.wv_loader import Vocab


def train(params):
    # GPU资源配置
    #config_gpu(use_cpu=True, gpu_memory=params['gpu_memory'])
    # 读取vocab训练
    print("Building the model ...")
    vocab = Vocab(params["vocab_path"], params["vocab_size"])
    params['vocab_size'] = vocab.count

    # 构建模型
    print("Building the model ...")
    # model = Seq2Seq(params)
    model = PGN(params)

    print("Creating the batcher ...")
    dataset = batcher(vocab, params)
    # print('dataset is ', dataset)

    # 获取保存管理者
    print("Creating the checkpoint manager")
    checkpoint = tf.train.Checkpoint(PGN=model)
    checkpoint_manager = tf.train.CheckpointManager(checkpoint, params['checkpoint_dir'], max_to_keep=5)
    checkpoint.restore(checkpoint_manager.latest_checkpoint)
    if checkpoint_manager.latest_checkpoint:
        print("Restored from {}".format(checkpoint_manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")
    # 训练模型
    print("Starting the training ...")
    train_model(model, dataset, params, checkpoint_manager)


if __name__ == '__main__':
    # 获得参数m
    params = get_params()
    # 训练模型
    train(params)
