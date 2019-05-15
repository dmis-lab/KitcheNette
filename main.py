import os
import sys
import logging
import pickle
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from datetime import datetime
from functools import partial
from torch.autograd import Variable

from tasks.create_dataset import IngredientDataset
from tasks.run import *
from tasks.model import Model
from utils import *


LOGGER = logging.getLogger()
DATA_PATH = './tasks/data/dataset/P7_im2recipe_dataset.pkl'  # For training (Pair scores)
PAIR_DIR = './tasks/data/pairings/P9_unknown_pairings_3000.csv'  # New pair data for scoring
EMBED_DIR = "./tasks/data/embeddings/EM1_D3_im2recipe-vocab-vectors.pkl"
INGR2CATEGORY_DIR = "./tasks/data/ingr2category.pkl"
CATEGORY2REP_DIR = "./tasks/data/category2rep.pkl"

CKPT_DIR = './results/'

MODEL_NAME = 'E999.mdl'

def str2bool(v):
    return v.lower() in ('yes', 'true', 't', '1', 'y')


argparser = argparse.ArgumentParser()
argparser.register('type', 'bool', str2bool)

# directories
argparser.add_argument('--data-path', type=str, default=DATA_PATH,
                       help='Dataset path')
argparser.add_argument('--pair-dir', type=str, default=PAIR_DIR,
                       help='Input new pairs')
argparser.add_argument('--embed-dir', type=str, default=EMBED_DIR,
                       help='Input new pairs')
argparser.add_argument('--checkpoint-dir', type=str, default=CKPT_DIR,
                       help='Directory for model checkpoint')
argparser.add_argument('--ingr2category-dir', type=str, default=INGR2CATEGORY_DIR,
                       help='Input new pairs')
argparser.add_argument('--category2rep-dir', type=str, default=CATEGORY2REP_DIR,
                       help='Input new pairs')

# Run settings
argparser.add_argument('--model-name', type=str, default=MODEL_NAME,
                       help='Model name for saving/loading')
argparser.add_argument('--print-step', type=float, default=300,
                       help='Display steps')
argparser.add_argument('--validation-step', type=float, default=1,
                       help='Number of random search validation')
argparser.add_argument('--train', type='bool', default=True,
                       help='Enable training')
argparser.add_argument('--pretrain', type='bool', default=False,
                       help='Enable training')
argparser.add_argument('--valid', type='bool', default=True,
                       help='Enable validation')
argparser.add_argument('--test', type='bool', default=True,
                       help='Enable testing')
argparser.add_argument('--resume', type='bool', default=False,
                       help='Resume saved model')
argparser.add_argument('--debug', type='bool', default=False,
                       help='Run as debug mode')
argparser.add_argument('--top-only', type='bool', default=False,
                       help='Return top/bottom 10% results only')

# Save outputs
argparser.add_argument('--save-embed', type='bool', default=False,
                       help='Save embeddings with loaded model')
argparser.add_argument('--save-prediction', type='bool', default=False,
                       help='Save predictions with loaded model')
argparser.add_argument('--save-prediction-unknowns', type='bool', default=False,
                       help='Save pair scores with loaded model')
argparser.add_argument('--embed-d', type=int, default=1,
                       help='0:val task data, 1:v0.n data')

# Train config
argparser.add_argument('--batch-size', type=int, default=64)
argparser.add_argument('--epoch', type=int, default=200)
argparser.add_argument('--learning-rate', type=float, default=5e-4)
argparser.add_argument('--weight-decay', type=float, default=0)
argparser.add_argument('--grad-max-norm', type=int, default=10)
argparser.add_argument('--grad-clip', type=int, default=10)

# Model config
argparser.add_argument('--binary', type='bool', default=False)
argparser.add_argument('--hidden-dim', type=int, default=512)
argparser.add_argument('--embed-dim', type=int, default=256)
argparser.add_argument('--linear-dr', type=float, default=0.2)
argparser.add_argument('--s-idx', type=int, default=0)
argparser.add_argument('--rep-idx', type=int, default=0)

argparser.add_argument('--category-emb', action='store_true', default=False)
argparser.add_argument('--category-dim', type=int, default=10)


argparser.add_argument('--dist-fn', type=str, default='concat')
argparser.add_argument('--seed', type=int, default=3)

args = argparser.parse_args()


def run_experiment(model, dataset, run_fn, args):
    print("\n\nrun_experiment")
    print("Current Representaion Index:", dataset.get_rep)
    print("Current Input Embedding Dimension:", dataset.input_dim)

    # Get dataloaders
    train_loader, valid_loader, test_loader = dataset.get_dataloader(
    batch_size=args.batch_size, s_idx=args.s_idx)

    # Save embeddings and exit
    if args.save_embed:
        model.load_checkpoint(args.checkpoint_dir, args.model_name)
        # run_fn(model, test_loader, dataset, args, metric, train=False)
        save_embed(model, dataset, args, args.data_path)
        sys.exit()

    # Save pair scores on pretrained model
    if args.save_prediction_unknowns:
        model.load_checkpoint(args.checkpoint_dir, args.model_name)
        # run_fn(model, test_loader, dataset, args, metric, train=False)
        save_prediction_unknowns(model, test_loader, dataset, args)
        sys.exit()

    # Save predictions on test dataset and exit
    if args.save_prediction:
        model.load_checkpoint(args.checkpoint_dir, args.model_name)
        # run_fn(model, test_loader, dataset, args, metric, train=False)
        save_prediction(model, test_loader, dataset, args)
        sys.exit()

    # Save and load model during experiments
    if args.train:
        if args.resume:
            model.load_checkpoint(args.checkpoint_dir, args.model_name)

        best = 999
        best_ep = 0
        converge_cnt = 0
        adaptive_cnt = 0
        #lr_decay = 0
        train_list = []
        break_switch = False

        for ep in range(args.epoch):
            print("\n===================================================================")
            LOGGER.info('Training Epoch %d' % (ep+1))
            train_loss = run_fn(model, train_loader, dataset, args, train=True)
            train_list.append(train_loss)

            if args.valid:
                print("\n")
                LOGGER.info('Validation')
                curr = run_fn(model, valid_loader, dataset, args, train=False)

                if not args.resume and curr < best:
                    best = curr
                    best_ep = ep+1
                    LOGGER.info('Best Validation Saved with {:.4f} at epoch{}'.format(
                                best, best_ep))
                    model.save_checkpoint({
                        'state_dict': model.state_dict(),
                        'optimizer': model.optimizer.state_dict()},
                        args.checkpoint_dir, args.model_name)
                    converge_cnt = 0
                    #lr_dacay = 0
                else:
                    converge_cnt += 1
                   # lr_decay += 1

                if ep >= 50:
                    if curr - np.mean(train_list[int(round(len(train_list)/2)):]) <= 0.01:
                        break_switch = True

                if break_switch:
                    break

                # if converge_cnt >= 3:
                #     for param_group in model.optimizer.param_groups:
                #         param_group['lr'] *= 0.5
                #         tmp_lr = param_group['lr']
                #     converge_cnt = 0
                #     adaptive_cnt += 1
                #     LOGGER.info('Adaptive {}: learning rate {:.4f}'.format(
                #                 adaptive_cnt, model.optimizer.param_groups[0]['lr']))

                # if adaptive_cnt > 5:
                #     LOGGER.info('Early stopping applied')
                #     break

    if args.test:
        print("\n===================================================================")
        LOGGER.info('Performance Test on Valid & Test Set')
        if args.train or args.resume:
            model.load_checkpoint(args.checkpoint_dir, args.model_name)
        if args.train:
            LOGGER.info('Best Validation at Epoch {}'.format(
                        best_ep))
        LOGGER.info('Validation')
        run_fn(model, valid_loader, dataset, args, train=False)
        LOGGER.info('Test')
        run_fn(model, test_loader, dataset, args, train=False)
        print("===================================================================")


def get_dataset(path):
    return pickle.load(open(path, 'rb'))


def get_run_fn(args):
    return run_reg


def get_model(args, dataset):
    dataset.set_rep(args.rep_idx)
    print("get_model")
    print("Current Representaion Index:", dataset.get_rep)
    print("Current Input Embedding Dimension:", dataset.input_dim)
    model = Model(input_dim=dataset.input_dim,
                      category_emb=args.category_emb,
                      category_dim=args.category_dim,
                      hidden_dim=args.hidden_dim,
                      embed_dim=args.embed_dim,
                      output_dim=1,
                      linear_dropout=args.linear_dr,
                      dist_fn=args.dist_fn,
                      learning_rate=args.learning_rate,
                      weight_decay=args.weight_decay).cuda()
    return model


def init_logging(args):
    LOGGER.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]',
                            '%m/%d/%Y %I:%M:%S %p')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    LOGGER.addHandler(console)

    # For logfile writing
    logfile = logging.FileHandler(
        args.checkpoint_dir + 'logs/' + args.model_name + '.txt', 'w')
    logfile.setFormatter(fmt)
    LOGGER.addHandler(logfile)


def init_seed(seed=None):
    if seed is None:
        seed = int(round(time.time() * 1000)) % 10000

    LOGGER.info("Using seed={}, pid={}".format(seed, os.getpid()))
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

def main():
    # Initialize logging and prepare seed
    init_logging(args)
    LOGGER.info('COMMAND: {}'.format(' '.join(sys.argv)))

    # Get datset, run function, model
    dataset = get_dataset(args.data_path)
    run_fn = get_run_fn(args)
    model_name = args.model_name

    # Random search validation
    for model_idx in range(args.validation_step):
        start_time = datetime.now()
        LOGGER.info('Validation step {}'.format(model_idx+1))
        init_seed(args.seed)

        # Get model
        model = get_model(args, dataset)

        # Run experiment
        run_experiment(model, dataset, run_fn, args)

        et = int((datetime.now() - start_time).total_seconds())
        LOGGER.info('TOTAL Elapsed Time: {:2d}:{:2d}:{:2d}'.format(et//3600, et%3600//60, et%60))

if __name__ == '__main__':
    main()
