
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
from torch.autograd import Variable as Variable
import random
import torch.nn.init as init
import numpy as np
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

        self.embed = nn.Embedding(V, D)

        if args.word_Embedding:
            self.embed.weight.data.copy_(args.pretrained_weight)
            self.embed.weight.require_grads = False

        self.linear = nn.Linear(in_features=D, out_features=C, bias=True)

    def sum_pooling(self, embed):
        assert embed.dim() == 3
        assert isinstance(embed, Variable)
        embed_sum = torch.sum(embed, 2)
        # print(embed_sum)
        return embed_sum

    def forward(self, x):
        x = self.embed(x)  # (N,W,D)
        x = x.permute(0, 2, 1)
        # print(x.size())
        x = self.sum_pooling(x)
        logit = self.linear(x)
        return logit


