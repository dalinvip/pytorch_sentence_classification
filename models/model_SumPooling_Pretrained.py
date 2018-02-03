
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
import sys
import os
torch.manual_seed(hyperparams.seed_num)
random.seed(hyperparams.seed_num)
pad = "<pad>"
unk = "<unk>"
judge_flag = "##$$"


class SumPooling(nn.Module):
    
    def __init__(self, args):
        super(SumPooling, self).__init__()
        self.args = args
        
        V = args.embed_num
        D = args.embed_dim
        C = args.class_num

        # self.embed = nn.Embedding(V, D, padding_idx=PaddingID)
        # self.embed, self.pretrained_embed_dim = self.load_pretrain(file=args.word_Embedding_Path, args=args)
        self.embed, self.pretrained_embed_dim = args.embed, args.pretrained_embed_dim

        # if args.word_Embedding:
        #     self.embed.weight.data.copy_(args.pretrained_weight)
        self.embed.weight.requires_grad = False

        self.dropout_embed = nn.Dropout(args.dropout_embed)
        self.dropout = nn.Dropout(args.dropout)

        self.linear = nn.Linear(in_features=D, out_features=C, bias=True)
        init.xavier_uniform(self.linear.weight)
        self.linear.bias.data.uniform_(-np.sqrt(6 / (D + 1)), np.sqrt(6 / (D + 1)))

    def sum_pooling(self, embed):
        assert embed.dim() == 3
        assert isinstance(embed, Variable)
        embed_sum = torch.sum(embed, 2)
        # print(embed_sum)
        return embed_sum

    def word_n_gram(self, word=None, feat_embedding_dict=None):
        # print("n-gram")
        feat_embedding = 0
        feat_count = 0
        word = "<" + word + ">"
        feat_embedding_list = []
        # print(word)
        for feat_num in range(3, 7):
            for i in range(0, len(word) - feat_num + 1):
                feat = word[i:(i + feat_num)]
                if feat.strip() in feat_embedding_dict:
                    feat_count += 1
                    # print(feat)
                    featID = feat_embedding_dict[feat.strip()]
                    # print(featID)
                    list_float = self.embed.weight.data[featID]
                    # list_float = [float(i) for i in feat_embedding_dict[feat.strip()]]
                    # print(np.array(list_float))
                    feat_embedding_list.append(np.array(list_float))
                    # feat_embedding = np.array(feat_embedding) + np.array(list_float)
        feat_embedding = np.sum(feat_embedding_list, axis=0)
        return feat_embedding, feat_count

    def handle_word_context(self, sentence=None, word=None, windows_size=5):
        data_dict = {}
        index = (len(sentence) // 2)
        left = sentence[:index]
        right = sentence[(index + 1):]
        context_dict = {}
        for i in range(len(left)):
            if left[i] == judge_flag:
                continue
            context_dict["F-" + str(len(left) - i) + "@" + left[i]] = 0
        for i in range(len(right)):
            if right[i] == judge_flag:
                continue
            context_dict["F" + str(i + 1) + "@" + right[i]] = 0
        data_dict[word] = set(context_dict)
        return data_dict

    def handle_word_context_1(self, sentence=None, windows_size=5):
        data_dict = {}
        for word_index, word in enumerate(sentence):
            context_dict = {}
            for i in range(windows_size):
                if (word_index - i) > 0:
                    context_dict["F-" + str(i + 1) + "@" + sentence[word_index - i - 1]] = 0
            for i in range(windows_size):
                if (word_index + i) < len(sentence) - 1:
                    context_dict["F" + str(i + 1) + "@" + sentence[word_index + i + 1]] = 0
            data_dict[word] = set(context_dict)
        return data_dict

    def context(self, context_dict=None, stoi=None, itos=None):
        context_num = 0
        context_embed_list = []
        context_embed = 0
        for context in context_dict:
            if context in stoi:
                context_num += 1
                contextID = stoi[context]
                # print(contextID)
                list_float = self.embed.weight.data[contextID]
                context_embed_list.append(np.array(list_float))
        context_embed = np.sum(context_embed_list, axis=0)
        return context_embed, context_num

    def handle_embedding_input(self, x):
        windows_size = 5
        itos = self.args.text_field.vocab.itos
        stoi = self.args.pretrained_text_field.vocab.stoi
        feat_context_embed = torch.zeros(x.size(0), x.size(1), self.pretrained_embed_dim)
        for id_batch in range(x.size(0)):
            sentence = [itos[word] for word in x.data[id_batch]]
            sentence_set = set(sentence)
            if pad in sentence_set:
                sentence = sentence[:sentence.index(pad)]

            # context_dict = self.handle_word_context(sentence=sentence, windows_size=5)
            for id_word in range(x.size(1)):
                word = itos[x.data[id_batch][id_word]]
                if word != pad:
                    start = id_word
                    sentence_paded = []
                    for i in range((start - windows_size), (start + windows_size + 1)):
                        if i >= len(sentence):
                            break
                        if i < 0:
                            sentence_paded.append(judge_flag)
                            continue
                        else:
                            sentence_paded.extend([sentence[i]])
                    sentence_paded_len = (2 * windows_size + 1 - len(sentence_paded))
                    if sentence_paded_len > 0:
                        sentence_paded.extend([judge_flag] * sentence_paded_len)
                    context_dict = self.handle_word_context(sentence=sentence_paded, word=word, windows_size=windows_size)

                    feat_sum_embedding, feat_ngram_num = self.word_n_gram(word=word, feat_embedding_dict=stoi)
                    if not isinstance(feat_sum_embedding, np.ndarray):
                        # if the word no n-gram in feature, replace with zero
                        feat_sum_embedding = np.array(list([0] * self.pretrained_embed_dim))
                        feat_ngram_num = 1
                    # print(context_dict)
                    context_embed, context_num = self.context(context_dict=context_dict[word], stoi=stoi)
                    feat_embed = np.divide(np.add(feat_sum_embedding, context_embed), np.add(feat_ngram_num, context_num))
                    # print(feat_embed)
                    feat_context_embed[id_batch][id_word] = torch.from_numpy(feat_embed)
                    # print(feat_context_embed)
        if self.args.use_cuda is True:
            feat_context_embed = Variable(feat_context_embed).cuda()
        else:
            feat_context_embed = Variable(feat_context_embed)
        return feat_context_embed

    def forward(self, x):
        x = self.handle_embedding_input(x)
        # print(x)
        # x = self.embed(x)  # (N,W,D)
        # x = self.dropout_embed(x)
        # x = Variable(x.data, requires_grad=False)
        x = x.permute(0, 2, 1)
        # print(x.size())
        x = self.sum_pooling(x)
        # x = self.dropout_embed(x)
        logit = self.linear(x)
        return logit


def load_pretrain(file, args):
    print("load pretrained embedding from {}".format(file))
    f = open(file, encoding='utf-8')
    allLines = f.readlines()
    indexs = set()
    info = allLines[0].strip().split(' ')
    embed_dim = len(info) - 1
    emb = nn.Embedding(args.embed_num, embed_dim)

    # init.uniform(emb.weight, a=-np.sqrt(3 / embed_dim), b=np.sqrt(3 / embed_dim))
    oov_emb = torch.zeros(1, embed_dim).type(torch.FloatTensor)
    now_line = 0
    for line in allLines:
        now_line += 1
        sys.stdout.write("\rhandling with the {} line.".format(now_line))
        info = line.split(" ")
        wordID = args.pretrained_text_field.vocab.stoi[info[0]]
        if wordID >= 0:
            indexs.add(wordID)
            for idx in range(embed_dim):
                val = float(info[idx + 1])
                emb.weight.data[wordID][idx] = val
                # oov_emb[0][idx] += val
    f.close()
    print("handle finished")

    unkID = args.pretrained_text_field.vocab.stoi[unk]
    paddingID = args.pretrained_text_field.vocab.stoi[pad]
    for idx in range(embed_dim):
        emb.weight.data[paddingID][idx] = 0
        emb.weight.data[unkID][idx] = 0

    return emb, embed_dim


