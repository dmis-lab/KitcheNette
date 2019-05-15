import sys
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import logging
import csv
import os

from datetime import datetime
from torch.autograd import Variable

from sklearn.metrics import roc_auc_score, precision_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import r2_score
from scipy.stats.stats import pearsonr
from math import sqrt


from utils import *
from tasks.plot import *


LOGGER = logging.getLogger(__name__)

def element(d):
    return [d[k] for k in range(0,len(d))]

# Run Regression Prediction
def run_reg(model, loader, dataset, args, train=False):
    total_step = 0.0
    stats = {'loss':[]}
    tar_set = []
    pred_set = []
    start_time = datetime.now()

    for d_idx, d in enumerate(loader):
        d1, d1_r, d1_c, d1_l, d2, d2_r, d2_c, d2_l, score = element(d)

        # Grad zero + mode change
        model.optimizer.zero_grad()
        if train: model.train(train)
        else: model.eval()

        # Get outputs from Model forward()
        outputs = model(d1_r.cuda(), d1_c.cuda(), d1_l, d2_r.cuda(), d2_c.cuda(), d2_l)

        # Get loss
        loss = model.get_loss(outputs, score.cuda())

        #
        stats['loss'] += [loss.data.item()]
        total_step += 1.0

        # Metrics for regression
        tmp_tar = score.data.cpu().numpy()
        tmp_pred = outputs[2].data.cpu().numpy()

        # Accumulate for final evaluation
        tar_set += list(tmp_tar[:])
        pred_set += list(tmp_pred[:])

        # Optimize model for train
        if train and not args.save_embed:
            loss.backward()
            nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()),
                    args.grad_max_norm)
            model.optimizer.step()

        # Train process for debug
        if train:
            ##################################################################
            # Calculate correlation for each batch
            ##################################################################
            corr = np.corrcoef(list(tmp_tar[:]), list(tmp_pred[:]))[0][1]

            # Print for print step or at last
            if d_idx % args.print_step == 0 or d_idx == (len(loader) - 1):
                    et = int((datetime.now() - start_time).total_seconds())
                    _progress = (
                        'Batch: {}/{} | Loss: {:.3f} | Total Correlation: {:.3f} | '.format(
                        d_idx + 1, len(loader), loss.data.item(), corr) +
                        '{:2d}:{:2d}:{:2d}'.format(
                        et//3600, et%3600//60, et%60))
                    LOGGER.debug(_progress)

    ##################################################################
    # Calculate correlation for "end of train" or "valid" or "test"
    ##################################################################


    total_loss = sum(stats['loss'])/len(stats['loss'])
    mse = mean_squared_error(tar_set, pred_set)
    mae = mean_absolute_error(tar_set, pred_set)
    mae2 = median_absolute_error(tar_set, pred_set)
    corr = np.corrcoef(tar_set, pred_set)[0][1]
    ev = explained_variance_score(tar_set, pred_set)
    r2 = r2_score(tar_set, pred_set)

    ##################################################################
    # End of an epoch
    ##################################################################

    LOGGER.info('Loss: {:.4f}\t'.format(
        total_loss))
    LOGGER.info('Loss\tMSE\tMAE\tMAE2\tCorr\tEV\t\tR2')
    LOGGER.info('{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}'.format(
        total_loss, mse, mae, mae2, corr, ev, r2))


    ##################################################################
    # Calculate Top only case
    ##################################################################
    if args.top_only:
        ths = [0.1, 0.2, 0.3, 0.4, 0.5]
        #ths = [0.1]
        LOGGER.info('th\tMSE_total\tMSE@1%\tMSE@2%\tMSE@5%\tAUROC\tPrecision@1%\tPrecision@2%\tPrecision@5%\tPrecision@10%')
        for th in ths:
            corr, msetotal, mse1, mse2, mse5, auroc, precision1, precision2, precision5, precision10 = evaluation(pred_set, tar_set, th)
            LOGGER.info('{}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}'.format(
                th, msetotal, mse1, mse2, mse5, auroc, precision1, precision2, precision5, precision10))

    et = int((datetime.now() - start_time).total_seconds())
    LOGGER.info('Elapsed Time: {:2d}:{:2d}:{:2d}'.format(et//3600, et%3600//60, et%60))

    # optimize with return value
    return total_loss

def precision_at_k(y_pred, y_true, k, th):
    #print(y_true[:20])
    #print(y_pred[:20])

    list_of_tuple = [(x, y) for x, y in zip(y_pred, y_true)]
    #print(list_of_tuple[:20])
    sorted_list_of_tuple = sorted(list_of_tuple, key=lambda tup: tup[0], reverse=True)
    #print(sorted_list_of_tuple[:20])

    topk = sorted_list_of_tuple[:int(len(sorted_list_of_tuple) * k)]
    topk_true = [x[1] for x in topk]
    topk_pred = [x[0] for x in topk]
    precisionk = precision_score([1 if x > th else 0 for x in topk_true],
                                 [1 if x > th else 0 for x in topk_pred], labels=[0,1], pos_label=1)
    # print([1 if x > 90.0 else 0 for x in topk_true])
    # print([1 if x > 90.0 else 0 for x in topk_pred])
    # print(precisionk)
    return precisionk

def mse_at_k(y_pred, y_true, k):
    list_of_tuple = [(x, y) for x, y in zip(y_pred, y_true)]
    sorted_list_of_tuple = sorted(list_of_tuple, key=lambda tup: tup[0], reverse=True)
    topk = sorted_list_of_tuple[:int(len(sorted_list_of_tuple) * k)]
    topk_true = [x[1] for x in topk]
    topk_pred = [x[0] for x in topk]

    msek = np.square(np.subtract(topk_pred, topk_true)).mean()
    return msek

def evaluation(y_pred, y_true, th):
    # print(y_pred)
    # print(y_true)
    # print(pearsonr(np.ravel(y_pred), y_true))
    corr = pearsonr(np.ravel(y_pred), y_true)[0]
    # mse = np.square(np.subtract(y_pred, y_true)).mean()
    msetotal = mse_at_k(y_pred, y_true, 1.0)
    mse1 = mse_at_k(y_pred, y_true, 0.01)
    mse2 = mse_at_k(y_pred, y_true, 0.02)
    mse5 = mse_at_k(y_pred, y_true, 0.05)

    auroc = float('nan')
    if len([x for x in y_true if x > th]) > 0:
        auroc = roc_auc_score([1 if x > th else 0 for x in y_true], y_pred)
    precision1 = precision_at_k(y_pred, y_true, 0.01, th)
    precision2 = precision_at_k(y_pred, y_true, 0.02, th)
    precision5 = precision_at_k(y_pred, y_true, 0.05, th)
    precision10 = precision_at_k(y_pred, y_true, 0.1, th)
    #print(auroc, precision1, precision2, precision5)
    return (corr, msetotal, mse1, mse2, mse5, auroc, precision1, precision2, precision5, precision10)

# Outputs response embeddings for a given dictionary
def save_embed(model, dataset, args, file):
    model.eval()
    ingr2vec = {}

    # Iterate dataset
    for idx, ingr in enumerate(dataset.ingredients):
        rep = dataset.ingredients[ingr]
        if args.embed_d == 1:
            d1_r = rep[args.rep_idx]
            d1_k = ingr in dataset.known
            d1_l = len(d1_r)
        else:
            d1_r = rep[0]
            d1_k = rep[1]
            d1_l = len(d1_r)

        d1_r = Variable(torch.FloatTensor(d1_r)).cuda()
        d1_l = torch.LongTensor(np.array([d1_l]))
        d1_r = d1_r.unsqueeze(0)
        d1_l = d1_l.unsqueeze(0)

        # Run model amd save embed
        _, embed1, embed2 = model(d1_r, d1_l, d1_r, d1_l)
        assert embed1.data.tolist() == embed2.data.tolist()

        # Save ingredient name and its leanred vectors
        ingr2vec[ingr] = [embed1.squeeze().data.tolist(), d1_k]

        # Print progress
        if idx % args.print_step == 0 or idx == len(dataset.ingredients) - 1:
            _progress = '{}/{} saving embeddings..'.format(
                idx + 1, len(dataset.ingredients))
            LOGGER.info(_progress)

    # plot
    save_plot(ingr2vec, args)

    # Save embed as pickle
    pickle.dump(ingr2vec, open('{}embed/{}.{}.pkl'.format(
                args.checkpoint_dir, file.split("/")[-1], args.model_name), 'wb'),
                protocol=3)
    LOGGER.info('{} number of known drugs.'.format(len(ingr2vec)))

# Outputs pred vs label scores given a dataloader
def save_prediction(model, loader, dataset, args):
    model.eval()
    csv_writer = csv.writer(open(args.checkpoint_dir + 'prediction_' +
                                 args.model_name + '.csv', 'w'))
    csv_writer.writerow(['ingr1', 'ingr1_cate', 'ingr2', 'ingr2_cate', 'prediction', 'target'])

    tar_set = []
    pred_set = []

    ingr2category = pickle.load(open(args.ingr2category_dir, 'rb'))

    for d_idx, (d1, d1_r, d1_c, d1_l, d2, d2_r, d2_c, d2_l, score) in enumerate(loader):
        # Run model for getting predictions
        outputs = model(d1_r.cuda(), d1_c.cuda(), d1_l, d2_r.cuda(), d2_c.cuda(), d2_l)
        predictions = outputs[2].data.cpu().numpy()
        targets = score.data.tolist()

        tar_set += list(targets)
        pred_set += list(predictions)

        for a1, a2, a3, a4 in zip(d1, d2, predictions, targets):
            csv_writer.writerow([a1, ingr2category[a1], a2, ingr2category[a2], a3, a4])

        # Print progress
        if d_idx % args.print_step == 0 or d_idx == len(loader) - 1:
            _progress = '{}/{} saving unknwon predictions..'.format(
                d_idx + 1, len(loader))
            LOGGER.info(_progress)

    mse = mean_squared_error(tar_set, pred_set)
    rmse = sqrt(mse)
    mae = mean_absolute_error(tar_set, pred_set)
    mae2 = median_absolute_error(tar_set, pred_set)
    corr = np.corrcoef(tar_set, pred_set)[0][1]
    ev = explained_variance_score(tar_set, pred_set)
    r2 = r2_score(tar_set, pred_set)

    LOGGER.info('Loss\tMSE\tMAE\tMAE2\tCorr\tEV\t\tR2')
    LOGGER.info('{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}'.format(
        rmse, mse, mae, mae2, corr, ev, r2))


# Outputs pred scores for new pair dataset
def save_prediction_unknowns(model, loader, dataset, args):
    model.eval()

    LOGGER.info('processing {}..'.format(args.pair_dir))
    embeddings = pickle.load(open(args.embed_dir, 'rb'))

    ingr2category = pickle.load(open(args.ingr2category_dir, 'rb'))
    category2rep = pickle.load(open(args.category2rep_dir, 'rb'))

    csv_writer = csv.writer(open(args.checkpoint_dir + 'prediction_unknowns_' +
                                 args.model_name + '.csv', 'w'))
    csv_writer.writerow(['ingr1', 'ingr1_cate', 'ingr2', 'ingr2_cate', 'prediction'])

    #for d_idx, (d1, d1_r, d1_l, d2, d2_r, d2_l, score) in enumerate(loader):
    #    print(d1, d1_r, d1_l, d2, d2_r, d2_l, score)

    df = pd.read_csv(args.pair_dir, sep=",")
    df = df.reset_index()
    print(len(df))

    batch = []
    for row_idx, row in df.iterrows():
        ingr1 = row['ingr1']
        ingr2 = row['ingr2']
        ingr1_r = embeddings[ingr1]
        ingr2_r = embeddings[ingr2]
        ingr1_c = category2rep[ingr2category[ingr1]]
        ingr2_c = category2rep[ingr2category[ingr2]]

        ingr1_cate = ingr2category[ingr1]
        ingr2_cate = ingr2category[ingr2]

        example = [ingr1, ingr1_r, ingr1_c, len(ingr1_r),
                   ingr2, ingr2_r, ingr2_c, len(ingr2_r), 0, ingr1_cate, ingr2_cate]
        batch.append(example)

        if len(batch) == 256:
            inputs = dataset.collate_fn(batch)
            outputs = model(inputs[1].cuda(), inputs[2].cuda(), inputs[3],
                                    inputs[5].cuda(), inputs[6].cuda(), inputs[7])
            predictions = outputs[2].data.cpu().numpy()



            for example, pred in zip(batch, predictions):
                csv_writer.writerow([example[0], example[9], example[4], example[10], pred])

            batch = []

        # Print progress
        if row_idx % 5000 == 0 or row_idx == len(df) - 1:
            _progress = '{}/{} saving unknwon predictions..'.format(
                row_idx + 1, len(df))
            LOGGER.info(_progress)

    if len(batch) > 0:
        inputs = dataset.collate_fn(batch)
        outputs, _, _ = model(inputs[1].cuda(), inputs[2].cuda(), inputs[3],
                                inputs[5].cuda(), inputs[6].cuda(), inputs[7])
        predictions = outputs[2].data.cpu().numpy()

        for example, pred in zip(batch, predictions):
            csv_writer.writerow([example[0], example[9], example[4], example[10], pred])

def save_plot(ingr2vec, args):
    # Plot embed as html
    with open("./tasks/data/ingr2category.pkl", "rb") as pickle_file:
        ingr2cate = pickle.load(pickle_file)

    # TSNE of ingr2vec
    ingr2vec_tsne = load_TSNE(ingr2vec, dim=2)
    save_path = '{}plot/{}.{}.pkl'.format(
                args.checkpoint_dir, file.split("/")[-1], args.model_name)
    #plot_clustering(ingr2vec, ingr2vec_tsne, save_path)
    plot_category(ingr2vec, ingr2vec_tsne, save_path, ingr2cate, True)
