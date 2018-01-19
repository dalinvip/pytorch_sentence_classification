# @Author : bamtercelboo
# @Datetime : 2018/1/15 10:22
# @File : train.py
# @Last Modify Time : 2018/1/15 10:22
# @Contact : bamtercelboo@{gmail.com, 163.com}

"""
    FILE :  train.py
    FUNCTION : train, eval, test
"""
import os
import sys
import torch
import torch.nn.functional as F
import torch.nn.utils as utils
import shutil
import random
import hyperparams
torch.manual_seed(hyperparams.seed_num)
random.seed(hyperparams.seed_num)


def train(train_iter, dev_iter, test_iter, model, args):
    if args.use_cuda:
        model.cuda()

    if args.Adam is True:
        print("Adam Training......")
        # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,
                                     weight_decay=args.weight_decay)
        # optimizer = torch.optim.Adagrad(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,
        #                              weight_decay=args.weight_decay)

    steps = 0
    model_count = 0
    model.train()
    max_dev_acc = -1
    for epoch in range(1, args.epochs+1):
        print("\n## 第{} 轮迭代，共计迭代 {} 次 ！##\n".format(epoch, args.epochs))
        print("now lr is {} \n".format(optimizer.param_groups[0].get("lr")))
        for batch in train_iter:
            feature, target = batch.text, batch.label
            feature.data.t_(), target.data.sub_(1)  # batch first, index align
            if args.use_cuda is True:
                feature, target = feature.cuda(), target.cuda()
            optimizer.zero_grad()
            logit = model(feature)
            # print(target)
            loss = F.cross_entropy(logit, target)
            loss.backward()
            # if args.clip_max_norm is not None:
            #     utils.clip_grad_norm(model.parameters(), max_norm=args.clip_max_norm)
            optimizer.step()
            steps += 1
            if steps % args.log_interval == 0:
                train_size = len(train_iter.dataset)
                corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
                accuracy = float(corrects)/batch.batch_size * 100.0
                sys.stdout.write(
                    '\rBatch[{}/{}] - loss: {:.6f}  acc: {:.4f}%({}/{})'.format(steps,
                                                                            train_size,
                                                                             loss.data[0], 
                                                                             accuracy,
                                                                             corrects,
                                                                             batch.batch_size))

            if steps % args.dev_interval == 0:
                model_count += 1
                dev_accuracy = eval(dev_iter, model, model_count, args)
                if dev_accuracy > max_dev_acc:
                    max_dev_acc = dev_accuracy
            if steps % args.test_interval == 0:
                if not os.path.isdir(args.save_dir):
                    os.makedirs(args.save_dir)
                save_prefix = os.path.join(args.save_dir, 'snapshot')
                save_path = '{}_steps{}.pt'.format(save_prefix, steps)
                torch.save(model, save_path)
                print(save_path, end=" ")
                test_model = torch.load(save_path)
                # model_count += 1
                test_eval(test_iter, test_model, save_path, args, model_count, max_dev_acc)
    return model_count


def eval(data_iter, model, model_count, args):
    model.eval()
    corrects, avg_loss = 0, 0
    for batch in data_iter:
        feature, target = batch.text, batch.label
        feature.data.t_(), target.data.sub_(1)  # batch first, index align
        if args.use_cuda:
            feature, target = feature.cuda(), target.cuda()

        logit = model(feature)
        loss = F.cross_entropy(logit, target, size_average=False)
        avg_loss += loss.data[0]
        corrects += (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()

    size = len(data_iter.dataset)
    avg_loss = loss.data[0]/size
    # accuracy = float(corrects)/size * 100.0
    accuracy = 100.0 * corrects/size
    model.train()
    print('\nEvaluation - loss: {:.6f}  acc: {:.4f}%({}/{}) \n'.format(avg_loss,
                                                                       accuracy,
                                                                       corrects,
                                                                       size))
    if os.path.exists("./Test_Result.txt"):
        file = open("./Test_Result.txt", "a")
    else:
        file = open("./Test_Result.txt", "w")
    file.write("Dev_Evaluation - loss: {:.6f}  acc: {:.4f}%({}/{}) --- modelCount: {} \n".format(avg_loss, accuracy,
                                                                                                corrects, size, model_count))
    return round(accuracy, 4)


def test_eval(data_iter, model, save_path, args, model_count, max_dev_acc):
    model.eval()
    corrects, avg_loss = 0, 0
    for batch in data_iter:
        feature, target = batch.text, batch.label
        feature.data.t_(), target.data.sub_(1)  # batch first, index align
        if args.use_cuda:
            feature, target = feature.cuda(), target.cuda()
        logit = model(feature)
        loss = F.cross_entropy(logit, target, size_average=False)
        avg_loss += loss.data[0]
        corrects += (torch.max(logit, 1)
                     [1].view(target.size()).data == target.data).sum()

    size = len(data_iter.dataset)
    avg_loss = loss.data[0]/size
    # accuracy = float(corrects)/size * 100.0
    accuracy = 100.0 * corrects/size
    model.train()
    print('\nEvaluation - loss: {:.6f}  acc: {:.4f}%({}/{}) \n'.format(avg_loss,
                                                                       accuracy,
                                                                       corrects,
                                                                       size))
    print("model_count {}".format(model_count))
    # test result
    if os.path.exists("./Test_Result.txt"):
        file = open("./Test_Result.txt", "a")
    else:
        file = open("./Test_Result.txt", "w")
    file.write("model " + save_path + "\n")
    file.write("Evaluation - loss: {:.6f}  acc: {:.4f}%({}/{}) \n".format(avg_loss, accuracy, corrects, size))
    file.write("model_count {} \n".format(model_count))
    # file.write("\n")
    file.close()
    # calculate the best score in current file
    resultlist = []
    if os.path.exists("./Test_Result.txt"):
        file = open("./Test_Result.txt")
        modelCount = -1
        test_result = -1
        for line in file.readlines():
            if line[:14] == "Dev_Evaluation" and float(line[(line.find("acc") + 5):line.find("%")]) == max_dev_acc:
                modelCount = int(line[(line.find("modelCount") + 12):-1])
            if line[:10] == "Evaluation":
                resultlist.append(float(line[34:41]))
                if modelCount != -1:
                    test_result = resultlist[modelCount - 1]
        result = sorted(resultlist)
        file.close()
        file = open("./Test_Result.txt", "a")
        file.write("\nThe Current Best Test Result is : " + str(result[len(result) - 1]))
        file.write("\nThe Current Best Dev Result is {}, For Test Result is {}: ".format(max_dev_acc, test_result))
        file.write("\n\n")
        file.close()
    shutil.copy("./Test_Result.txt", "./snapshot/" + args.mulu + "/Test_Result.txt")
    # whether to delete the model after test acc so that to save space
    if os.path.isfile(save_path) and args.rm_model is True:
        os.remove(save_path)
