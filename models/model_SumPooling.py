
# @Author : bamtercelboo
# @Datetime : 2018/1/16 22:21
# @File : model_SumPooling.py
# @Last Modify Time : 2018/1/16 22:21
# @Contact : bamtercelboo@{gmail.com, 163.com}

"""
    FILE :  model_SumPooling.py
    FUNCTION : Neutral Network Model that only has sum_pooling
                    sum_pooling:add all word embedding to backward
    REFERENCE ; Li et al. 2017. Neural Bag-of-Ngrams
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import torch.nn.init as init
import hyperparams
torch.manual_seed(hyperparams.seed_num)
random.seed(hyperparams.seed_num)


class SumPooling(nn.Module):
    
    def __init__(self, args):
        super(SumPooling, self).__init__()
        self.args = args
        
        V = args.embed_num
        D = args.embed_dim
        C = args.class_num
        Co = args.kernel_num
        Ks = args.kernel_sizes

        self.embed = nn.Embedding(V, D)

        if args.word_Embedding:
            self.embed.weight.data.copy_(args.pretrained_weight)
            self.embed.weight.require_grads = False

        self.dropout = nn.Dropout(args.dropout)
        self.dropout_embed = nn.Dropout(args.dropout_embed)
        in_fea = len(Ks) * Co
        self.fc = nn.Linear(in_features=in_fea, out_features=C, bias=True)

    def forward(self, x):
        x = self.embed(x)  # (N,W,D)
        x = self.dropout_embed(x)

        return ""

    def sum_pooling(self, embed):
        return ""
