
# @Author : bamtercelboo
# @Datetime : 2018/1/14 22:45
# @File : hyperparams.py
# @Last Modify Time : 2018/1/14 22:45
# @Contact : bamtercelboo@{gmail.com, 163.com}

"""
    File : main_hyperparams.py
    Function : main function
"""

import os
import sys
import argparse
import datetime
import torch
import torchtext.data as data
from Dataloader.Data_Loader import *
from Dataloader.Data_Loader_CV import *
from Dataloader.Load_Pretrained_Embed import *
import train_ALL_CNN
import train_CV
from models import model_CNN
from models import model_SumPooling
import shutil
import random
import hyperparams as hy
# solve default encoding problem
from imp import reload
defaultencoding = 'utf-8'
if sys.getdefaultencoding() != defaultencoding:
    reload(sys)
    sys.setdefaultencoding(defaultencoding)

# random seed
torch.manual_seed(hy.seed_num)
random.seed(hy.seed_num)

# init hyperparams instance
hyperparams = hy.Hyperparams()

parser = argparse.ArgumentParser(description="Text Classification for sentence level.")
# Data path
parser.add_argument('-train_path', type=str, default=hyperparams.train_path, help='train data path')
parser.add_argument('-dev_path', type=str, default=hyperparams.dev_path, help='dev data path')
parser.add_argument('-test_path', type=str, default=hyperparams.test_path, help='test data path')
parser.add_argument('-nfold', type=int, default=hyperparams.nfold, help='cv')
# shuffle data
parser.add_argument('-shuffle', action='store_true', default=hyperparams.shuffle, help='shuffle the data when load data' )
parser.add_argument('-epochs_shuffle', action='store_true', default=hyperparams.epochs_shuffle, help='shuffle the data every epoch' )
# model params
parser.add_argument("-CNN", action='store_true', default=hyperparams.CNN, help="CNN neural network model")
parser.add_argument("-wide_conv", action='store_true', default=hyperparams.wide_conv, help="wide CNN neural network model")
parser.add_argument('-embed_dim', type=int, default=hyperparams.embed_dim, help='embedding dim')
parser.add_argument('-kernel_num', type=int, default=hyperparams.kernel_num, help='number of each kind of kernel')
parser.add_argument('-kernel_sizes', type=str, default=hyperparams.kernel_sizes, help='comma-separated kernel size to use for conv')
parser.add_argument('-dropout', type=float, default=hyperparams.dropout, help='dropout')
parser.add_argument('-dropout_embed', type=float, default=hyperparams.dropout_embed, help='dropout')
parser.add_argument('-max_norm', type=float, default=hyperparams.max_norm, help='max_norm params in nn.Embedding()')
parser.add_argument('-clip_max_norm', type=float, default=hyperparams.clip_max_norm, help='clip_norm params in train')
# Train
parser.add_argument("-Adam", action="store_true", default=hyperparams.Adam, help="elf.Adam = optimizer for train")
parser.add_argument('-lr', type=float, default=hyperparams.learning_rate, help='initial learning rate [default: 0.001]')
parser.add_argument('-learning_rate_decay', type=float, default=hyperparams.learning_rate_decay, help='learn rate decay')
parser.add_argument('-weight_decay', type=float, default=hyperparams.weight_decay, help='weight_decay')
parser.add_argument('-epochs', type=int, default=hyperparams.epochs, help="The number of iterations for train")
parser.add_argument('-batch_size', type=int, default=hyperparams.train_batch_size, help="The number of batch_size for train")
parser.add_argument('-dev_batch_size', type=int, default=hyperparams.dev_batch_size, help='batch size for dev [default: None]')
parser.add_argument('-test_batch_size', type=int, default=hyperparams.test_batch_size, help='batch size for test [default: None]')
parser.add_argument('-log_interval',  type=int, default=hyperparams.log_interval,   help='how many steps to wait before logging training status [default: 1]')
parser.add_argument('-dev_interval', type=int, default=hyperparams.dev_interval, help='how many steps to wait before testing [default: 100]')
parser.add_argument('-test_interval', type=int, default=hyperparams.test_interval, help='how many steps to wait before saving [default:500]')
parser.add_argument('-save_dir', type=str, default=hyperparams.save_dir, help='save model')
parser.add_argument('-rm_model', action="store_true", default=hyperparams.rm_model, help='remove model after test')
# build vocab
parser.add_argument("-min_freq", type=int, default=hyperparams.min_freq, help="build vocab for cut off")
# word_Embedding
parser.add_argument("-word_Embedding", action="store_true", default=hyperparams.word_Embedding, help="whether to use pretrained word embedding")
parser.add_argument("-word_Embedding_Path", type=str, default=hyperparams.word_Embedding_Path, help="Pretrained Embedding Path")
# GPU
parser.add_argument('-use_cuda', action='store_true', default=hyperparams.use_cuda, help='use gpu')
parser.add_argument("-gpu_device", type=int, default=hyperparams.gpu_device, help="gpu device number")
parser.add_argument("-num_threads", type=int, default=hyperparams.num_threads, help="threads number")
# option
args = parser.parse_args()

assert args.test_interval == args.dev_interval
assert args.CV is True, "Not CrossFold Dataset, Please Run main_hyperparams.py"


def load_data(text_field, label_field, path_file, **kargs):
    train_data, test_data = DataCV.splits(path_file, text_field, label_field, shuffle=args.shuffle)
    print("len(train_data) {} ".format(len(train_data)))
    print("len(test_data) {} ".format(len(test_data)))
    # print("all word")
    # text_field.build_vocab(train_data.text, min_freq=args.min_freq)
    text_field.build_vocab(train_data.text, test_data.text, min_freq=args.min_freq)
    label_field.build_vocab(train_data.label, test_data.label)
    train_iter, test_iter = create_Iterator(train_data, test_data, batch_size=args.batch_size, **kargs)
    return train_iter, test_iter


# create Iterator
def create_Iterator(train_data, test_data, batch_size, **kargs):
    train_iter, test_iter = data.Iterator.splits((train_data, test_data),
                                                 batch_sizes=(batch_size, len(test_data)), **kargs)
    return train_iter, test_iter


# cross CV
def cv_spilit_file(path, nfold, test_id):
    assert (nfold > 1) and (test_id >= 0) and (test_id < nfold)
    print("CV file......")
    if os.path.exists("./temp_train.txt"):
        os.remove("./temp_train.txt")
    if os.path.exists("./temp_test.txt"):
        os.remove("./temp_test.txt")
    file_train = open("./temp_train.txt", "w", encoding="utf-8")
    file_test = open("./temp_test.txt", "w", encoding="utf-8")
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            # print(i, line)
            file_train.writelines(line) if i % nfold != test_id else file_test.writelines(line)
    file_train.close()
    file_test.close()


def show_params():
    print("\nParameters:")
    if os.path.exists("./Parameters.txt"):
        os.remove("./Parameters.txt")
    file = open("Parameters.txt", "a", encoding="UTF-8")
    for attr, value in sorted(args.__dict__.items()):
        if attr.upper() != "PRETRAINED_WEIGHT":
            print("\t{}={}".format(attr.upper(), value))
        file.write("\t{}={}\n".format(attr.upper(), value))
    file.close()
    shutil.copy("./Parameters.txt", args.save_dir)
    shutil.copy("./hyperparams.py", args.save_dir)


def calculate_result(id):
    resultlist = []
    if os.path.exists("./Test_Result.txt"):
        file = open("./Test_Result.txt")
        for line in file.readlines():
            if line[:10] == "Evaluation":
                resultlist.append(float(line[34:41]))
        result = sorted(resultlist)
        file.close()
        file = open("./Test_Result.txt", "a")
        file.write("\nThe Best Result is : " + str(result[len(result) - 1]))
        file.write("\n")
        file.close()
        shutil.copy("./Test_Result.txt", "./snapshot/" + args.mulu)
        shutil.copy("./Test_Result.txt", "./Temp_Test_Result/Test_Result_" + str(id) + ".txt")
    best_result = result[len(result) - 1]
    return best_result


# calculate the all cv means
def cal_mean(list):
    sum = 0
    for i in list:
        sum += i
    avg = sum / len(list)
    return avg


def cal_result():
    resultlist = []
    if os.path.exists("./Test_Result.txt"):
        file = open("./Test_Result.txt")
        for line in file.readlines():
            if line[:10] == "Evaluation":
                resultlist.append(float(line[34:41]))
        result = sorted(resultlist)
        file.close()
        file = open("./Test_Result.txt", "a")
        file.write("\nThe Best Result is : " + str(result[len(result) - 1]))
        file.write("\n")
        file.close()
    shutil.copy("./Test_Result.txt", str(args.save_dir) + "/Test_Result.txt")


def main():
    # if args.use_cuda is True:
        # use deterministic algorithm for cnn
        # torch.backends.cudnn.deterministic = True
    args.kernel_sizes = [int(k) for k in args.kernel_sizes.split(',')]
    # save file
    mulu = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    args.mulu = mulu
    args.save_dir = os.path.join(args.save_dir, mulu)
    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)

    Temp_Test_Result = "./Temp_Test_Result"
    if os.path.exists(Temp_Test_Result):
        shutil.rmtree(Temp_Test_Result)
    if not os.path.isdir(Temp_Test_Result):
        os.makedirs(Temp_Test_Result)

    cv_result = []
    # CV loop start
    for id in range(args.nfold):
        print("\nthe {} CV file".format(id))
        # build vocab and iterator
        text_field = data.Field(lower=True)
        label_field = data.Field(sequential=False)
        cv_spilit_file(args.train_path, args.nfold, test_id=id)
        train_iter, test_iter = load_data(text_field, label_field, path_file=args.train_path, device=args.gpu_device,
                                          repeat=False, shuffle=args.epochs_shuffle, sort=False)
        args.embed_num = len(text_field.vocab)
        args.class_num = len(label_field.vocab) - 1
        args.PaddingID = text_field.vocab.stoi[text_field.pad_token]
        print("embed_num : {}, class_num : {}".format(args.embed_num, args.class_num))
        print("PaddingID {}".format(args.PaddingID))
        # pretrained word embedding
        if args.word_Embedding:
            pretrain_embed = load_pretrained_emb_zeros(path=args.word_Embedding_Path,
                                                       text_field_words_dict=text_field.vocab.itos,
                                                       pad=text_field.pad_token)
            calculate_oov(path=args.word_Embedding_Path, text_field_words_dict=text_field.vocab.itos,
                          pad=text_field.pad_token)
            args.pretrained_weight = pretrain_embed

        # print params
        show_params()

        # load model and start train
        if args.CNN is True:
            print("loading CNN model.....")
            # model = model_CNN.CNN_Text(args)
            model = model_SumPooling.SumPooling(args)
            # for param in model.parameters():
            #     param.requires_grad = False
            shutil.copy("./models/model_CNN.py", args.save_dir)
            print(model)
            if args.use_cuda is True:
                print("using cuda......")
                model = model.cuda()
            print("CNN training start......")
            if os.path.exists("./Test_Result.txt"):
                os.remove("./Test_Result.txt")
            model_count = train_CV.train(train_iter, test_iter, model, args)
        cv_result.append(calculate_result(id=id))

    print(cv_result)
    cv_mean = cal_mean(cv_result)
    print("The best result is {:.6f} ".format(cv_mean))
    file = open("./Temp_Test_Result/Final_Result.txt", "a")
    for index, value in enumerate(cv_result):
        stra = str(index + 1) + "   " + str(value)
        file.write(stra)
        file.write("\n")
    file.write("mean_value  " + str(cv_mean))
    file.close()
    shutil.copytree("./Temp_Test_Result/", "./snapshot/" + mulu + "/Temp_Test_Result")


if __name__ == "__main__":
    main()
