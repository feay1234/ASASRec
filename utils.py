import glob
import random
import torch
import pandas as pd
from Dataset import Dataset
import time
import os, math
import numpy as np
import tensorflow as tf
from multiprocessing import Pool
from multiprocessing import cpu_count
import argparse
import logging
from time import time
from time import strftime
from time import localtime

def write2file(path, name, output):
    print(output)
    if not os.path.exists(path):
        os.makedirs(path)
    thefile = open(path+name, 'a')
    thefile.write("%s\n" % output)
    thefile.close()

def prediction2file(path, name, pred):
    if not os.path.exists(path):
        os.makedirs(path)
    thefile = open(path+name, 'w')
    for item in pred:
        thefile.write("%f\n" % item)
    thefile.close()

def set_seed(seed, cuda=False):

    np.random.seed(seed)
    random.seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)
    else:
        torch.manual_seed(seed)


def getDataset(data, path, evalMode):
    # if data in ["ml-1m", "yelp", "pinterest-20"]:
    if data in ["brightkite", "fsq11", "yelp"]:
        columns = ['uid', 'iid', 'rating', 'hour', 'day', 'month', 'timestamp']
        train = pd.read_csv(path + "data/%sTrain" % data, names=columns, sep="\t")
        test = pd.read_csv(path + "data/%sTest" % data, names=columns, sep="\t")
        df = train.append(test)
        df.sort_values(["uid", "timestamp"], inplace=True)
        dataset = Dataset(df, evalMode)

    elif data in ["ml-1m", "yelp-he"]:
        names = ["uid", "iid", "rating", "timestamp"]
        data = "yelp" if data == "yelp-he" else data
        train = pd.read_csv(path + "data/%s.train.rating" % data, sep="\t", names=names)
        test = pd.read_csv(path + "data/%s.test.rating" % data, sep="\t", names=names)
        df = train.append(test)
        dataset = Dataset(df, evalMode)

    elif data in ["beauty", "steam", "video", "ml-sas"]:
        names = ["uid", "iid"]
        if data == "beauty":
            df = pd.read_csv(path + "data/Beauty.txt", sep=" ", names=names)
        elif data == "steam":
            df = pd.read_csv(path + "data/Steam.txt", sep=" ", names=names)
        elif data == "video":
            df = pd.read_csv(path + "data/Video.txt", sep=" ", names=names)
        else:
            df = pd.read_csv(path + "data/ml-1m.txt", sep=" ", names=names)
        dataset = Dataset(df, evalMode)

    elif data == "test":
        columns = ["uid", "timestamp", "lat", "lng", "iid"]
        df = pd.read_csv(path + "data/brightkite.txt", names=columns, sep="\t", nrows=10000)
        dataset = Dataset(df, evalMode)

    return dataset

def output_evaluate(model, sess, dataset, train_batches, eval_feed_dicts, epoch_count, batch_time, train_time, prev_acc,
                    runName, args, output_adv):
    loss_begin = time()
    train_loss, post_acc = training_loss_acc(model, sess, train_batches, output_adv)
    loss_time = time() - loss_begin

    eval_begin = time()
    result, raw_result = evaluate(model, sess, dataset, eval_feed_dicts, output_adv, args)
    eval_time = time() - eval_begin

    # check embedding
    embedding_P, embedding_Q = sess.run([model.embedding_P, model.embedding_Q])

    hr, ndcg, auc = np.swapaxes(result, 0, 1)[-1]
    res = "Epoch %d [%.1fs + %.1fs]: HR = %.4f, NDCG = %.4f ACC = %.4f ACC_adv = %.4f [%.1fs], |P|=%.2f, |Q|=%.2f" % \
          (epoch_count, batch_time, train_time, hr, ndcg, prev_acc,
           post_acc, eval_time, np.linalg.norm(embedding_P), np.linalg.norm(embedding_Q))

    write2file(args.path + "out/" + args.opath, runName + ".out", res)

    return post_acc, ndcg, result, raw_result


# input: batch_index (shuffled), model, sess, batches
# do: train the model optimizer
def training_batch(model, sess, batches, adver=False):
    user_input, item_input_pos, user_dns_list, item_dns_list = batches
    # dns for every mini-batch
    # dns = 1, i.e., BPR
    if model.dns == 1:
        item_input_neg = item_dns_list
        # for BPR training
        for i in range(len(user_input)):
            feed_dict = {model.user_input: user_input[i],
                         model.item_input_pos: item_input_pos[i],
                         model.item_input_neg: item_input_neg[i]}
            if adver:
                sess.run([model.update_P, model.update_Q], feed_dict)
            sess.run(model.optimizer, feed_dict)
    # dns > 1, i.e., BPR-dns
    elif model.dns > 1:
        item_input_neg = []
        for i in range(len(user_input)):
            # get the output of negtive sample
            feed_dict = {model.user_input: user_dns_list[i],
                         model.item_input_neg: item_dns_list[i]}
            output_neg = sess.run(model.output_neg, feed_dict)
            # select the best negtive sample as for item_input_neg
            item_neg_batch = []
            for j in range(0, len(output_neg), model.dns):
                item_index = np.argmax(output_neg[j: j + model.dns])
                item_neg_batch.append(item_dns_list[i][j: j + model.dns][item_index][0])
            item_neg_batch = np.array(item_neg_batch)[:, None]
            # for mini-batch BPR training
            feed_dict = {model.user_input: user_input[i],
                         model.item_input_pos: item_input_pos[i],
                         model.item_input_neg: item_neg_batch}
            sess.run(model.optimizer, feed_dict)
            item_input_neg.append(item_neg_batch)
    return user_input, item_input_pos, item_input_neg


# calculate the gradients
# update the adversarial noise
def adv_update(model, sess, train_batches):
    user_input, item_input_pos, item_input_neg = train_batches
    # reshape mini-batches into a whole large batch
    user_input, item_input_pos, item_input_neg = \
        np.reshape(user_input, (-1, 1)), np.reshape(item_input_pos, (-1, 1)), np.reshape(item_input_neg, (-1, 1))
    feed_dict = {model.user_input: user_input,
                 model.item_input_pos: item_input_pos,
                 model.item_input_neg: item_input_neg}

    return sess.run([model.update_P, model.update_Q], feed_dict)


# input: model, sess, batches
# output: training_loss
def training_loss_acc(model, sess, train_batches, output_adv):
    train_loss = 0.0
    acc = 0
    num_batch = len(train_batches[1])
    user_input, item_input_pos, item_input_neg = train_batches
    for i in range(len(user_input)):
        # print user_input[i][0]. item_input_pos[i][0], item_input_neg[i][0]
        feed_dict = {model.user_input: user_input[i],
                     model.item_input_pos: item_input_pos[i],
                     model.item_input_neg: item_input_neg[i]}
        if output_adv:
            loss, output_pos, output_neg = sess.run([model.loss_adv, model.output_adv, model.output_neg_adv], feed_dict)
        else:
            loss, output_pos, output_neg = sess.run([model.loss, model.output, model.output_neg], feed_dict)
        train_loss += loss
        acc += ((output_pos - output_neg) > 0).sum() / len(output_pos)
    return train_loss / num_batch, acc / num_batch


def init_eval_model(dataset, args):
    begin_time = time()
    global _dataset
    # global _model
    global _args
    global _candidates
    _dataset = dataset
    # _model = model
    _args = args
    _candidates = dataset.df.iid.tolist()

    pool = Pool(cpu_count())
    feed_dicts = pool.map(_evaluate_input, list(range(_dataset.num_users)))
    pool.close()
    pool.join()

    # print(("Load the evaluation model done [%.1f s]" % (time() - begin_time)))
    return feed_dicts


def _evaluate_input(user):
    # generate items_list
    test_item = _dataset.testRatings[user][1]
    if _args.eval_mode == "sample":

        random.seed(2019)
        item_input = []
        for i in range(100):
            r = random.choice(_candidates)
            while r in _dataset.trainList[user] or test_item == r:
                r = random.choice(_candidates)
            item_input.append(r)
    else:
        item_input = set(range(_dataset.num_items)) - set(_dataset.trainList[user])
        if test_item in item_input:
            item_input.remove(test_item)
    item_input = list(item_input)
    item_input.append(test_item)
    user_input = np.full(len(item_input), user, dtype='int32')[:, None]
    item_input = np.array(item_input)[:, None]
    return user_input, item_input


def evaluate(model, sess, dataset, feed_dicts, output_adv, args):
    global _model
    global _K
    global _sess
    global _dataset
    global _feed_dicts
    global _output
    _dataset = dataset
    _model = model
    _sess = sess
    _K = 100 if args.eval_mode == "all" else 10
    _feed_dicts = feed_dicts
    _output = output_adv

    res = []
    for user in range(_dataset.num_users):
        res.append(_eval_by_user(user))
    res = np.array(res)
    hr, ndcg, auc = (res.mean(axis=0)).tolist()

    return (hr, ndcg, auc), res


def _eval_by_user(user):
    # get prredictions of data in testing set
    user_input, item_input = _feed_dicts[user]
    feed_dict = {_model.user_input: user_input, _model.item_input_pos: item_input}
    if _output:
        predictions = _sess.run(_model.output_adv, feed_dict)
    else:
        predictions = _sess.run(_model.output, feed_dict)

    neg_predict, pos_predict = predictions[:-1], predictions[-1]
    position = (neg_predict >= pos_predict).sum()

    # calculate from HR@1 to HR@100, and from NDCG@1 to NDCG@100, AUC
    hr, ndcg, auc = [], [], []
    for k in range(1, _K + 1):
        hr.append(position < k)
        ndcg.append(math.log(2) / math.log(position + 2) if position < k else 0)
        auc.append(1 - (position / len(neg_predict)))  # formula: [#(Xui>Xuj) / #(Items)] = [1 - #(Xui<=Xuj) / #(Items)]
    # k = 10
    # hr.append(position < k)
    # ndcg.append(math.log(2) / math.log(position + 2) if position < k else 0)
    # auc.append(1 - (position / len(neg_predict)))  # formula: [#(Xui>Xuj) / #(Items)] = [1 - #(Xui<=Xuj) / #(Items)]

    return hr, ndcg, auc


def init_logging(args, time_stamp):
    path = "Log/%s_%s/" % (strftime('%Y-%m-%d_%H', localtime()), args.task)
    if not os.path.exists(path):
        os.makedirs(path)
    logging.basicConfig(filename=path + "%s_log_embed_size%d_%s" % (args.dataset, args.embed_size, time_stamp),
                        level=logging.INFO)
    logging.info(args)
    print(args)



def run_normal_model(epoch_start, epoch_end, max_ndcg, best_res, ranker, dataset, args, eval_feed_dicts, runName, time_stamp):
    with tf.Session() as sess:
        ranker.init(dataset.trainSeq, args.batch_size, sess)

        # initialized the save op

        if args.adver:
            ckpt_save_path = "Pretrain/%s/ASASREC/embed_%d/%s/" % (args.dataset, args.embed_size, time_stamp)
            ckpt_restore_path = "Pretrain/%s/SASREC/embed_%d/%s/" % (args.dataset, args.embed_size, time_stamp)
            # mylist = [f for f in glob.glob("Pretrain/save/%s/SASREC/embed_%d/*" % (args.dataset, args.embed_size))]
            # args.restore = mylist[0] + "/"
            # ckpt_restore_path = mylist[0] + "/"

        else:
            ckpt_save_path = "Pretrain/%s/SASREC/embed_%d/%s/" % (args.dataset, args.embed_size, time_stamp)
            ckpt_restore_path = 0 if args.restore is None else "Pretrain/%s/SASREC/embed_%d/%s/" % (
            args.dataset, args.embed_size, args.restore)

            # Pretrain/fsq11-sort/SASREC/embed_64/2020_01_24_12_50_17/

        if not os.path.exists(ckpt_save_path):
            os.makedirs(ckpt_save_path)
        if ckpt_restore_path and not os.path.exists(ckpt_restore_path):
            os.makedirs(ckpt_restore_path)

        saver_ckpt = tf.train.Saver()
        sess.run(tf.global_variables_initializer())

        # restore the weights when pretrained
        if args.restore is not None or epoch_start:
            ckpt = tf.train.get_checkpoint_state(os.path.dirname(str(ckpt_restore_path) + 'checkpoint'))
            if ckpt and ckpt.model_checkpoint_path:
                # TODO add feature to get input from user then start from pre-trained.
                saver_ckpt.restore(sess, ckpt.model_checkpoint_path)
                print("restored", ckpt_restore_path)
        # initialize the weights
        else:
            print("Initialized from scratch")

        # train by epoch
        for epoch_count in range(epoch_start, epoch_end + 1):

            x_train, y_train = ranker.get_train_instances(dataset.trainMatrix)

            # training the model
            train_begin = time()
            loss = ranker.train(x_train, y_train, args.batch_size)
            # loss = 0
            train_time = time() - train_begin

            if epoch_count % args.verbose == 0:

                eval_begin = time()
                res = []
                for user in range(dataset.num_users):
                    user_input, item_input = eval_feed_dicts[user]
                    predictions = ranker.rank(user_input, item_input)

                    neg_predict, pos_predict = predictions[:-1], predictions[-1]
                    position = (neg_predict >= pos_predict).sum()

                    # calculate from HR@1 to HR@100, and from NDCG@1 to NDCG@100, AUC
                    hr, ndcg, auc = [], [], []
                    K = 100 if args.eval_mode == "all" else 10
                    for k in range(1, K + 1):
                        hr.append(position < k)
                        ndcg.append(math.log(2) / math.log(position + 2) if position < k else 0)
                        auc.append(1 - (
                                position / len(
                            neg_predict)))  # formula: [#(Xui>Xuj) / #(Items)] = [1 - #(Xui<=Xuj) / #(Items)]
                    res.append((hr, ndcg, auc))
                    # for dev time
                    # break


                res = np.array(res)
                hr, ndcg, auc = (res.mean(axis=0)).tolist()
                cur_res = (hr, ndcg, auc)
                hr, ndcg, auc = np.swapaxes((hr, ndcg, auc), 0, 1)[-1]

                eval_time = time() - eval_begin

                output = "Epoch %d [%.1fs + %.1fs]: HR = %.4f, NDCG = %.4f ACC = %.4f ACC_adv = %.4f [%.1fs], |P|=%.2f, |Q|=%.2f" % \
                         (epoch_count, train_time, 0, hr, ndcg, loss,
                          loss, eval_time, 0, 0)

                write2file(args.path + "out/" + args.opath, runName + ".out", output)

            # print and log the best result
            if max_ndcg < ndcg:
                max_ndcg = ndcg
                best_res['result'] = cur_res
                best_res['epoch'] = epoch_count

                _hrs = res[:, 0, -1]
                _ndcgs = res[:, 1, -1]
                prediction2file(args.path + "out/" + args.opath, runName + ".hr", _hrs)
                prediction2file(args.path + "out/" + args.opath, runName + ".ndcg", _ndcgs)
            # save the embedding weights
            if args.ckpt > 0 and epoch_count % args.ckpt == 0:
                saver_ckpt.save(sess, ckpt_save_path + 'weights', global_step=epoch_count)

        return max_ndcg, best_res

def run_keras_model(epoch_start, epoch_end, max_ndcg, best_res, ranker, args, dataset, eval_feed_dicts, runName):

    # train by epoch
    for epoch_count in range(epoch_start, epoch_end + 1):

        x_train, y_train = ranker.get_train_instances(dataset.trainMatrix)
        # training the model
        train_begin = time()
        loss = ranker.train(x_train, y_train, args.batch_size)
        train_time = time() - train_begin

        if math.isnan(loss):
            break

        if epoch_count % args.verbose == 0:

            eval_begin = time()
            res = []
            for user in range(dataset.num_users):
                user_input, item_input = eval_feed_dicts[user]
                predictions = ranker.rank(user_input, item_input)

                neg_predict, pos_predict = predictions[:-1], predictions[-1]
                position = (neg_predict >= pos_predict).sum()

                # calculate from HR@1 to HR@100, and from NDCG@1 to NDCG@100, AUC
                hr, ndcg, auc = [], [], []
                K = 100 if args.eval_mode == "all" else 10
                for k in range(1, K + 1):
                    hr.append(position < k)
                    ndcg.append(math.log(2) / math.log(position + 2) if position < k else 0)
                    auc.append(1 - (
                            position / len(
                        neg_predict)))  # formula: [#(Xui>Xuj) / #(Items)] = [1 - #(Xui<=Xuj) / #(Items)]
                res.append((hr, ndcg, auc))
                # break



            res = np.array(res)
            hr, ndcg, auc = (res.mean(axis=0)).tolist()
            cur_res = (hr, ndcg, auc)
            hr, ndcg, auc = np.swapaxes((hr, ndcg, auc), 0, 1)[-1]

            eval_time = time() - eval_begin

            output = "Epoch %d [%.1fs + %.1fs]: HR = %.4f, NDCG = %.4f ACC = %.4f ACC_adv = %.4f [%.1fs], |P|=%.2f, |Q|=%.2f" % \
                     (epoch_count, train_time, 0, hr, ndcg, loss,
                      loss, eval_time, 0, 0)

            write2file(args.path + "out/" + args.opath, runName + ".out", output)

        # print and log the best result
        if max_ndcg < ndcg:
            max_ndcg = ndcg
            best_res['result'] = cur_res
            best_res['epoch'] = epoch_count

            _hrs = res[:, 0, -1]
            _ndcgs = res[:, 1, -1]
            prediction2file(args.path + "out/" + args.opath, runName + ".hr", _hrs)
            prediction2file(args.path + "out/" + args.opath, runName + ".ndcg", _ndcgs)

    return max_ndcg, best_res