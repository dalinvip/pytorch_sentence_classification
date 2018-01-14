# @Author : bamtercelboo
# @Datetime : 2018/1/14 22:45
# @File : hyperparams.py
# @Last Modify Time : 2018/1/14 22:45
# @Contact : bamtercelboo@{gmail.com, 163.com}

import torch
import random
# random seed num
seed_num = 233
torch.manual_seed(seed_num)
random.seed(seed_num)

"""
    set hyperparams for load Data、model、train
"""


class Hyperparams():
    def __init__(self):

        # Data path
        self.train_path = "./pos_test_data/train.ctb60.pos.hwc"
        self.dev_path = "./pos_test_data/dev.ctb60.pos.hwc"
        self.test_path = "./pos_test_data/test.ctb60.pos.hwc"
        self.shuffle = True
        self.epochs_shuffle = True

        # model
        self.CNN = True
        self.kernel_num = 200
        self.kernel_sizes = "1,2,3,4"
        self.dropout = 0.6
        self.dropout_embed = 0.6
        self.max_norm = None
        self.clip_max_norm = 5

        # select optim algorhtim for train
        self.Adam = True
        self.learning_rate = 0.001
        self.learning_rate_decay = 1   # value is 1 means not change lr
        self.epochs = 150
        self.batch_size = 16
        self.log_interval = 1
        self.test_interval = 100
        self.save_interval = 100
        self.save_dir = "snapshot"

        # min freq to include during built the vocab, default is 1
        self.min_freq = 1

        # word_Embedding
        self.word_Embedding = True
        self.word_Embedding_Path = "./word2vec/glove.sentiment.conj.pretrained.txt"

        # GPU
        self.use_cuda = False
        self.gpu_device = 0
        self.num_threads = 1

        self.snapshot = None
        self.num_threads = 1

        # L2 weight_decay
        self.weight_decay = 1e-8   # default value is zero in Adam SGD
        # self.weight_decay = 0   # default value is zero in Adam SGD

        # whether to delete the model after test acc so that to save space
        self.rm_model = True
