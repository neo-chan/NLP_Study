#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   test.py    
@Contact :   cxbwater@163.com
@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2019/12/14 下午4:31   cxb      1.0         None
'''

# import lib
from src.tensorflow.test_helper import beam_decode
import tensorflow as tf
from src.tensorflow.batcher import beam_test_batch_generator
from src.tensorflow.seq2seq_model import Seq2Seq
from src.tensorflow.test_helper import beam_decode, greedy_decode
from src.utils.config import checkpoint_dir, test_data_path
from src.utils.data_process import load_test_dataset
from src.utils.gpu_utils import config_gpu
from src.utils.params_utils import get_params
from src.utils.wv_loader import Vocab
import pandas as pd


def test(params):
    assert params["mode"].lower() in ["test", "eval"], "change training mode to 'test' or 'eval'"
    assert params["beam_size"] == params["batch_size"], "Beam size must be equal to batch_size, change the params"
    # GPU资源配置
    config_gpu(use_cpu=True)

    print("Building the model ...")
    model = Seq2Seq(params)

    print("Creating the vocab ...")
    vocab = Vocab(params["vocab_path"], params["vocab_size"])

    print("Creating the checkpoint manager")
    print("Creating the checkpoint manager")
    checkpoint = tf.train.Checkpoint(Seq2Seq=model)
    checkpoint_manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=5)
    checkpoint.restore(checkpoint_manager.latest_checkpoint)
    if checkpoint_manager.latest_checkpoint:
        print("Restored from {}".format(checkpoint_manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")
    print("Model restored")

    if params['greedy_decode']:
        predict_result(model, params, vocab, params['result_save_path'])
    else:
        b = beam_test_batch_generator(params["beam_size"])
        results = []
        for batch in b:
            best_hyp = beam_decode(model, batch, vocab, params)
            results.append(best_hyp.abstract)
        save_predict_result(results, params['result_save_path'])
        print('save result to :{}'.format(params['result_save_path']))


def predict_result(model, params, vocab, result_save_path):
    test_X = load_test_dataset(params['max_enc_len'])
    # 预测结果
    results = greedy_decode(model, test_X, params['batch_size'], vocab, params)
    # 保存结果
    save_predict_result(results, result_save_path)


def save_predict_result(results, result_save_path):
    # 读取结果
    test_df = pd.read_csv(test_data_path)
    # 填充结果
    test_df['Prediction'] = results
    # 　提取ID和预测结果两列
    test_df = test_df[['QID', 'Prediction']]
    # 保存结果.
    test_df.to_csv(result_save_path, index=None, sep=',')


if __name__ == '__main__':
    # 获得参数
    params = get_params()
    params["greedy_decode"]=True
    # 获得参数
    test(params)