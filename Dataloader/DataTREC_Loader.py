# @Author : bamtercelboo
# @Datetime : 2018/1/22 9:38
# @File : DataTREC_Loader.py
# @Last Modify Time : 2018/1/22 9:38
# @Contact : bamtercelboo@{gmail.com, 163.com}

"""
    FILE :  DataTREC_Loader.py
    FUNCTION : TREC Dataset
"""

import re
import os
from torchtext import data
import random
import torch
import hyperparams
from imp import reload
import sys
defaultencoding = 'utf-8'
if sys.getdefaultencoding() != defaultencoding:
    reload(sys)
    sys.setdefaultencoding(defaultencoding)
torch.manual_seed(hyperparams.seed_num)
random.seed(hyperparams.seed_num)


class DataTREC(data.Dataset):

    def __init__(self, text_field, label_field, path=None, examples=None, **kwargs):
        """
        Arguments:
            text_field: The field that will be used for text data.
            label_field: The field that will be used for label data.
            path: Path to the data file.
            examples: The examples contain all the data.
            char_data: The char level to solve
            Remaining keyword arguments: Passed to the constructor of data.Dataset.
        """
        def clean_str(string):
            """
            Tokenization/string cleaning for all datasets except for SST.
            Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
            """
            string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
            string = re.sub(r"\'s", " \'s", string)
            string = re.sub(r"\'ve", " \'ve", string)
            string = re.sub(r"n\'t", " n\'t", string)
            string = re.sub(r"\'re", " \'re", string)
            string = re.sub(r"\'d", " \'d", string)
            string = re.sub(r"\'ll", " \'ll", string)
            string = re.sub(r",", " , ", string)
            string = re.sub(r"!", " ! ", string)
            string = re.sub(r"\(", " \( ", string)
            string = re.sub(r"\)", " \) ", string)
            string = re.sub(r"\?", " \? ", string)
            string = re.sub(r"\s{2,}", " ", string)

            return string.strip()

        text_field.preprocessing = data.Pipeline(clean_str)
        fields = [('text', text_field), ('label', label_field)]

        if examples is None:
            path = None if os.path.join(path) is None else os.path.join(path)
            examples = []
            with open(path, encoding="utf-8") as f:
                a, b, c, d, e, g = 0, 0, 0, 0, 0, 0
                for line in f:
                    # sentence, flag = line.strip().split(' ||| ')
                    # print(line)
                    label, seq, sentence = line.partition(" ")
                    # clear string in every sentence
                    sentence = clean_str(sentence)
                    if label == '0':
                        a += 1
                        examples += [data.Example.fromlist([sentence, '0'], fields=fields)]
                    elif label == '1':
                        b += 1
                        examples += [data.Example.fromlist([sentence, '1'], fields=fields)]
                    elif label == "2":
                        c += 1
                        examples += [data.Example.fromlist([sentence, '2'], fields=fields)]
                    elif label == "3":
                        d += 1
                        examples += [data.Example.fromlist([sentence, '3'], fields=fields)]
                    elif label == "4":
                        e += 1
                        examples += [data.Example.fromlist([sentence, '4'], fields=fields)]
                    elif label == "5":
                        g += 1
                        examples += [data.Example.fromlist([sentence, '5'], fields=fields)]
                print("0 {}, 1 {} ".format(a, b))
                print("2 {}, 3 {} ".format(c, d))
                print("4 {}, 5 {} ".format(e, g))
                print("**************************************")
        super(DataTREC, self).__init__(examples, fields, **kwargs)

    @classmethod
    def splits(cls, train_path, test_path, text_field, label_field, shuffle=True, **kwargs):
        """Create dataset objects for splits of the MR dataset.
        Arguments:
            text_field: The field that will be used for the sentence.
            label_field: The field that will be used for label data.
            dev_ratio: The ratio that will be used to get split validation dataset.
            shuffle: Whether to shuffle the data before split.
            root: The root directory that the dataset's zip archive will be
                expanded into; therefore the directory in whose trees
                subdirectory the data files will be stored.
            train: The filename of the train data. Default: 'train.txt'.
            Remaining keyword arguments: Passed to the splits method of
                Dataset.
        """
        print("train_path {}  \ntest_path {}".format(train_path, test_path))
        examples_train = cls(text_field, label_field, path=train_path, **kwargs).examples
        examples_test = cls(text_field, label_field, path=test_path, **kwargs).examples

        if shuffle:
            print("shuffle data examples......")
            random.shuffle(examples_train)
            random.shuffle(examples_test)

        return (cls(text_field, label_field, examples=examples_train),
                cls(text_field, label_field, examples=examples_test))