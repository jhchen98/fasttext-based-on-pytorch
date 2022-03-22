# coding: UTF-8
import time
import torch
import numpy as np
from train_eval import train, init_network
from importlib import import_module
from data_process import build_dataset, build_iterator, get_time_dif


if __name__ == '__main__':
    dataset = 'Data'  # 数据集
    embedding = 'random'
    x = import_module('models.FastText')
    config = x.Config(dataset, embedding)      # 加载神经网络模型的参数
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样
 
    start_time = time.time()
    print("Loading data...")
    vocab, train_data, dev_data, test_data = build_dataset(config)   # 每句中字对应的id, 标签, 句子长度, bigram, trigram)
    train_iter = build_iterator(train_data, config)   # (x, seq_len, bigram, trigram), y
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # train
    config.n_vocab = len(vocab)
    model = x.Model(config).to(config.device)
    print(model.parameters)
    train(config, model, train_iter, dev_iter, test_iter)
