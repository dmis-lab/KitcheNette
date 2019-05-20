import numpy as np
import sys
import copy
import pickle
import string
import os
import random
import csv
import torch
import argparse
import scipy.sparse as sp

from os.path import expanduser
from torch.autograd import Variable
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler

LOAD_PATH = '../data/kitchenette_dataset.pkl'  # Dataset

class IngredientDataset(object):
    def __init__(self, ingredient_pair_path, ingredient_embedding_path, ingredient_vocab_path):

        self.initial_setting()

        # 1. Save ingredient pair scores
        self.ingredients, self.categories, self.pairs = self.process_pairs(ingredient_pair_path, ingredient_vocab_path)

        # 2. Add Embedding Vectors of Ingredients
        self.append_ingredient_vectors(ingredient_embedding_path, self.ingredients)

        # 3. Split Dataset
        self.dataset = self.split_dataset(self.pairs, unk_test=False)

    def initial_setting(self):
        # Dataset split into train/valid/test
        self.ingredients = {}
        self.categories = {}
        self.pairs = []
        self.dataset = {'train': [], 'valid': [], 'test': []}
        self.SR = [0.7, 0.1, 0.2] # split ratio
        self.input_maxlen = 0

        # Vector Dimensions
        self.sub_lens = []

    def get_dataloader(self, batch_size=32, shuffle=True, num_workers=10, s_idx=0):

        train_dataset = Representation(self.dataset['train'], self.ingredients, self.categories,
                                     self._rep_idx, s_idx=s_idx)
        train_sampler = SortedBatchSampler(train_dataset.lengths(),
                                           batch_size,
                                           shuffle=True)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True,
        )

        valid_dataset = Representation(self.dataset['valid'], self.ingredients, self.categories,
                                      self._rep_idx, s_idx=s_idx)
        valid_sampler = SortedBatchSampler(valid_dataset.lengths(),
                                           batch_size,
                                           shuffle=False)
        valid_loader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=batch_size,
            sampler=valid_sampler,
            num_workers=num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True,
            shuffle=False,
        )

        test_dataset = Representation(self.dataset['test'], self.ingredients, self.categories,
                                       self._rep_idx, s_idx=s_idx)
        test_sampler = SortedBatchSampler(test_dataset.lengths(),
                                           batch_size,
                                           shuffle=False)
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            sampler=test_sampler,
            num_workers=num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True,
            shuffle=False,
        )

        return train_loader, valid_loader, test_loader

    def collate_fn(self, batch):
        # pair1[0], pair1_rep[1], pair1_rep_cate[2], pair1_len[3], pair2[4], pair2_rep[5], pair2_rep_cate[6], pair2_len[7], score[8]
        pair1_raws = [ex[0] for ex in batch]
        pair1_lens = torch.LongTensor([ex[3] for ex in batch])
        pair2_raws = [ex[4] for ex in batch]
        pair2_lens = torch.LongTensor([ex[7] for ex in batch])

        pair1_maxlen = max([len(ex[1]) for ex in batch])
        pair1_reps = torch.FloatTensor(len(batch), pair1_maxlen).zero_()

        pair1_rep_cates_maxlen = max([len(ex[2]) for ex in batch])
        pair1_rep_cates = torch.FloatTensor(len(batch), pair1_rep_cates_maxlen).zero_()

        pair2_maxlen = max([len(ex[5]) for ex in batch])
        pair2_reps = torch.FloatTensor(len(batch), pair2_maxlen).zero_()

        pair2_rep_cates_maxlen = max([len(ex[6]) for ex in batch])
        pair2_rep_cates = torch.FloatTensor(len(batch), pair2_rep_cates_maxlen).zero_()

        scores = torch.FloatTensor(len(batch)).zero_()

        for idx, ex in enumerate(batch):
            pair1_rep = ex[1]
            pair1_rep = torch.FloatTensor(pair1_rep)
            pair1_reps[idx, :pair1_rep.size(0)].copy_(pair1_rep)

            pair1_rep_cate = ex[2]
            pair1_rep_cate = torch.FloatTensor(pair1_rep_cate)
            pair1_rep_cates[idx, :pair1_rep_cate.size(0)].copy_(pair1_rep_cate)

            pair2_rep = ex[5]
            pair2_rep = torch.FloatTensor(pair2_rep)
            pair2_reps[idx, :pair2_rep.size(0)].copy_(pair2_rep)

            pair2_rep_cate = ex[6]
            pair2_rep_cate = torch.FloatTensor(pair2_rep_cate)
            pair2_rep_cates[idx, :pair2_rep_cate.size(0)].copy_(pair2_rep_cate)

            scores[idx] = ex[8]

        # Set as Variables
        pair1_reps = Variable(pair1_reps)
        pair2_reps = Variable(pair2_reps)
        pair1_rep_cates = Variable(pair1_rep_cates)
        pair2_rep_cates = Variable(pair2_rep_cates)
        scores = Variable(scores)

        return (pair1_raws, pair1_reps, pair1_rep_cates, pair1_lens,
                pair2_raws, pair2_reps, pair2_rep_cates, pair2_lens,
                scores)

    def decode_data(self, d1, d1_r, d1_c, d1_l, d2, d2_r, d2_c, d2_l, score):
        d1_r = d1_r.data.tolist()
        d2_r = d2_r.data.tolist()
        print('Pair1: {}, length: {}'.format(d1, d1_l))
        print(d1_r[:10])
        print(d1_c)
        print('Pair2: {}, length: {}'.format(d2, d2_l))
        print(d2_r[:10])
        print(d2_c)
        print('Score: {}\n'.format(score.data.item()))

    # rep_idx [0, 1, 2, 3]
    def set_rep(self, rep_idx):
        self._rep_idx = rep_idx

    @property
    def get_rep(self):
        return self._rep_idx

    @property
    def input_dim(self):
        #random
        if self._rep_idx == 0:
            return self.sub_lens[0]
        #glove embedding vectors
        elif self._rep_idx == 1:
            return self.sub_lens[1]
        #im2recipe embedding vectors
        elif self._rep_idx == 2:
            return self.sub_lens[2]
        else:
            assert False, 'Wrong rep_idx {}'.format(rep_idx)

    def loadGloveModel(self, gloveFile):
        print("Loading Glove Model...")
        f = open(gloveFile,'r')
        model = {}
        for line in f:
            splitLine = line.split()
            word = splitLine[0].lower().replace('-',' ').replace(' ','_')
            embedding = np.array([float(val) for val in splitLine[1:]])
            model[word] = embedding
        print("Done.",len(model)," words loaded!")
        return model


#Representation(self.dataset['tr'], self.ingredients, self.categories, self._rep_idx, s_idx=s_idx)
class Representation(Dataset):
    def __init__(self, examples, ingredients, categories, rep_idx, s_idx):
        self.examples = examples
        self.ingredients = ingredients
        self.categories = categories
        self.rep_idx = rep_idx
        self.s_idx = s_idx

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        example = self.examples[index]
        next_idx = index
        while (self.ingredients[example[0]][self.rep_idx].any() is None or
               self.ingredients[example[1]][self.rep_idx].any() is None):
            next_idx = (next_idx + 1) % len(self.examples)
            example = self.examples[next_idx]
        pair1, pair2, scores = example

        # Choose pair representation
        pair1_rep = self.ingredients[pair1][self.rep_idx]
        pair1_rep_cate = self.categories[pair1]
        pair1_len = len(pair1_rep)
        pair2_rep = self.ingredients[pair2][self.rep_idx]
        pair2_rep_cate = self.categories[pair2]
        pair2_len = len(pair2_rep)

        # [npmi_score, nnpmi_score, jaccard_score, pairing_type]
        score = scores[self.s_idx]

        return pair1, pair1_rep, pair1_rep_cate, pair1_len, pair2, pair2_rep, pair2_rep_cate, pair2_len, score

    def lengths(self):
        def get_longer_length(ex):
            pair1_len = len(self.ingredients[ex[0]][self.rep_idx])
            pair2_len = len(self.ingredients[ex[1]][self.rep_idx])
            length = pair1_len if pair1_len > pair2_len else pair2_len
            return [length, pair1_len, pair2_len]
        return [get_longer_length(ex) for ex in self.examples]

class SortedBatchSampler(Sampler):
    def __init__(self, lengths, batch_size, shuffle=True):
        self.lengths = lengths
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        lengths = np.array(
            [(l1, l2, l3, np.random.random()) for l1, l2, l3 in self.lengths],
            dtype=[('l1', np.int_), ('l2', np.int_), ('l3', np.int_),
                   ('rand', np.float_)]
        )
        indices = np.argsort(lengths, order=('l1', 'rand'))
        batches = [indices[i:i + self.batch_size]
                   for i in range(0, len(indices), self.batch_size)]
        if self.shuffle:
            np.random.shuffle(batches)
        return iter([i for batch in batches for i in batch])

    def __len__(self):
        return len(self.lengths)

def init_seed(seed=None):
    if seed is None:
        seed = int(round(time.time() * 1000)) % 10000
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

if __name__ == '__main__':
    def str2bool(v):
        return v.lower() in ('yes', 'true', 't', '1', 'y')

    # Run settings
    argparser = argparse.ArgumentParser()
    argparser.register('type', 'bool', str2bool)

    # Load
    argparser.add_argument('--load-path', type=str, default=LOAD_PATH,
                           help='Dataset path to load')
    args = argparser.parse_args()

    init_seed(1004)
    print('## Load preprocess %s' % args.load_path)
    dataset = pickle.load(open(args.load_path, 'rb'))

    # Loader testing
    # rep_idx = 0 : random vectors
    # rep_idx = 1 : glove vectors
    # rep_idx = 2 : im2recipe vectors
    dataset.set_rep(rep_idx=2)
    print("Current Representaion Index:", dataset.get_rep)

    # s_idx == 0 means regression
    # s_idx == 1 means binary classification

    for idx, (d1, d1_r, d1_c, d1_l, d2, d2_r, d2_c, d2_l, score) in enumerate(
            dataset.get_dataloader(batch_size=1600, s_idx=0)[2]):
        dataset.decode_data(d1[0], d1_r[0], d1_c[0], d1_l[0], d2[0], d2_r[0], d2_c[0], d2_l[0], score[0])
        pass

    dataset.set_rep(rep_idx=2)
    print("Current Representaion Index:", dataset.get_rep)

    for idx, (d1, d1_r, d1_c, d1_l, d2, d2_r, d2_c, d2_l, score) in enumerate(
            dataset.get_dataloader(batch_size=1600, s_idx=0)[2]):
        dataset.decode_data(d1[0], d1_r[0], d1_c[0], d1_l[0], d2[0], d2_r[0], d2_c[0], d2_l[0], score[0])
        pass
