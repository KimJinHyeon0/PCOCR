"""Unified interface to all dynamic graph model experiments"""

'''
class imbalance test

adjusted amount of training graph.
random sampled majority classes with num of minority class's graph 
'''

import math
import logging
import time
import sys
import random
import argparse

from tqdm import tqdm
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

from module import TGAN
from graph import NeighborFinder

from collections import Counter

class LR(torch.nn.Module):
    def __init__(self, dim, drop=0.3, agg='mean'):
        super().__init__()

        self.fc_1 = torch.nn.Linear(dim, 80)
        self.fc_2 = torch.nn.Linear(80, 10)
        self.fc_3 = torch.nn.Linear(10, 1)
        self.act = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=drop, inplace=True)


        # if agg == 'lstm':
        #     self.logger.info('Aggregation uses LSTM model')
        # elif agg == 'mean':
        #     self.logger.info('Aggregation uses constant mean model')
        # else:
        #     raise ValueError('invalid agg_method value, use attn or lstm')


    def forward(self, x):
        x = x.mean(dim=0)
        x = self.act(self.fc_1(x))
        x = self.dropout(x)
        x = self.act(self.fc_2(x))
        x = self.dropout(x)
        return self.fc_3(x)


random.seed(222)
np.random.seed(222)
torch.manual_seed(222)

### Argument and global variables
parser = argparse.ArgumentParser('Interface for TGAT experiments on node classification')
parser.add_argument('-d', '--data', type=str, help='data sources to use, try wikipedia or reddit', default='iama')
parser.add_argument('--bs', type=int, default=30, help='batch_size')
parser.add_argument('--prefix', type=str, default='')
parser.add_argument('--n_degree', type=int, default=50, help='number of neighbors to sample')
parser.add_argument('--n_neg', type=int, default=1)
parser.add_argument('--n_head', type=int, default=2)
parser.add_argument('--n_epoch', type=int, default=15, help='number of epochs')
parser.add_argument('--n_layer', type=int, default=2)
parser.add_argument('--lr', type=float, default=3e-4)
parser.add_argument('--tune', action='store_true', help='parameters tunning mode, use train-test split on training data only.')
parser.add_argument('--drop_out', type=float, default=0.1, help='dropout probability')
parser.add_argument('--gpu', type=int, default=0, help='idx for the gpu to use')
parser.add_argument('--node_dim', type=int, default=None, help='Dimentions of the node embedding')
parser.add_argument('--time_dim', type=int, default=None, help='Dimentions of the time embedding')
parser.add_argument('--agg_method', type=str, choices=['attn', 'lstm', 'mean'], help='local aggregation method', default='attn')
parser.add_argument('--attn_mode', type=str, choices=['prod', 'map'], default='prod')
parser.add_argument('--time', type=str, choices=['time', 'pos', 'empty'], help='how to use time information', default='time')

parser.add_argument('--new_node', action='store_true', help='model new node')
parser.add_argument('--uniform', action='store_true', help='take uniform sampling from temporal neighbors')

try:
    args = parser.parse_args()
except:
    parser.print_help()
    sys.exit(0)

BATCH_SIZE = args.bs
NUM_NEIGHBORS = args.n_degree
NUM_NEG = 1
NUM_EPOCH = args.n_epoch
NUM_HEADS = args.n_head
DROP_OUT = args.drop_out
GPU = args.gpu
UNIFORM = args.uniform
NEW_NODE = args.new_node
USE_TIME = args.time
AGG_METHOD = args.agg_method
ATTN_MODE = args.attn_mode
SEQ_LEN = NUM_NEIGHBORS
DATA = args.data
NUM_LAYER = args.n_layer
LEARNING_RATE = args.lr
NODE_LAYER = 1
NODE_DIM = args.node_dim
TIME_DIM = args.time_dim

### set up logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler('log/{}.log'.format(str(time.time())))
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.WARN)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)
logger.info(args)

### Load data and train val test split
g_df = pd.read_csv('./processed/{}_structure.csv'.format(DATA))
e_feat = np.load('./processed/{}_edge_feat.npy'.format(DATA))
n_feat = np.load('./processed/{}_node_feat.npy'.format(DATA))

train_time = 3888000

time_cut = 86400

g_num = g_df.g_num.values
g_ts = g_df.g_ts.values
src_l = g_df.u.values
dst_l = g_df.i.values
e_idx_l = g_df.idx.values
label_l = g_df.label.values
ts_l = g_df.ts.values



max_src_index = src_l.max()
max_idx = max(src_l.max(), dst_l.max())

train_flag = (g_ts < train_time)
test_flag = (g_ts >= train_time)

logger.info('Training use all train data')
train_g_num_l = g_num[train_flag]
train_src_l = src_l[train_flag]
train_dst_l = dst_l[train_flag]
train_ts_l = ts_l[train_flag]
train_e_idx_l = e_idx_l[train_flag]
train_label_l = label_l[train_flag]

# use the true test dataset
test_g_num_l = g_num[test_flag]
test_src_l = src_l[test_flag]
test_dst_l = dst_l[test_flag]
test_ts_l = ts_l[test_flag]
test_e_idx_l = e_idx_l[test_flag]
test_label_l = label_l[test_flag]



### Initialize the data structure for graph and edge sampling
adj_list = [[] for _ in range(max_idx + 1)]
for src, dst, eidx, ts in zip(train_src_l, train_dst_l, train_e_idx_l, train_ts_l):
    adj_list[src].append((dst, eidx, ts))
    adj_list[dst].append((src, eidx, ts))
train_ngh_finder = NeighborFinder(adj_list, uniform=UNIFORM)

# full graph with all the data for the test and validation purpose
full_adj_list = [[] for _ in range(max_idx + 1)]
for src, dst, eidx, ts in zip(src_l, dst_l, e_idx_l, ts_l):
    full_adj_list[src].append((dst, eidx, ts))
    full_adj_list[dst].append((src, eidx, ts))
full_ngh_finder = NeighborFinder(full_adj_list, uniform=UNIFORM)

### Model initialize
device = torch.device('cuda:{}'.format(GPU))
tgan = TGAN(train_ngh_finder, n_feat, e_feat,
            num_layers=NUM_LAYER, use_time=USE_TIME, agg_method=AGG_METHOD, attn_mode=ATTN_MODE,
            seq_len=SEQ_LEN, n_head=NUM_HEADS, drop_out=DROP_OUT, node_dim=NODE_DIM, time_dim=TIME_DIM)
optimizer = torch.optim.Adam(tgan.parameters(), lr=LEARNING_RATE)
criterion = torch.nn.BCELoss()
tgan = tgan.to(device)

num_instance = len(train_src_l)
logger.debug('num of training instances: {}'.format(num_instance))
logger.debug('num of graphs per epoch: {}'.format(len(set(train_g_num_l))))

logger.info('loading saved TGAN model')
model_path = f'./saved_models/{args.prefix}-{args.agg_method}-{args.attn_mode}-{DATA}.pth'
tgan.load_state_dict(torch.load(model_path))
tgan.eval()
logger.info('TGAN models loaded')
logger.info('Start training node classification task')

lr_model = LR(n_feat.shape[1])
lr_model = lr_model.to(device)
lr_optimizer = torch.optim.Adam(lr_model.parameters(), lr=args.lr)
tgan.ngh_finder = full_ngh_finder
lr_criterion = torch.nn.BCELoss()
lr_criterion_eval = torch.nn.BCELoss()

def eval_epoch(src_l, ts_l, g_num_l, lr_model, tgan, data_type, num_layer=NODE_LAYER):

    ground_truth = np.zeros(len(set(g_num_l)))
    pred_prob = np.zeros(len(set(g_num_l)))

    loss = 0
    num_instance = len(set(g_num_l))

    with torch.no_grad():
        lr_model.eval()
        tgan.eval()

        g_num_list = []
        raw_list = []
        splitted_list = []
        label_list = []
        pred_list = []
        prob_list = []

        for i, k in enumerate(set(g_num_l)):
            temp_src_cut = src_l[g_num_l == k]
            temp_ts_cut = ts_l[g_num_l == k]

            valid_flag = (temp_ts_cut < time_cut)

            src_l_cut = temp_src_cut[valid_flag]
            ts_l_cut = temp_ts_cut[valid_flag]

            label = 0
            if False in valid_flag:
                label = 1

            if len(src_l_cut) < 2:
                continue

            src_embed = tgan.tem_conv(src_l_cut, ts_l_cut, num_layer)
            src_label = torch.tensor([label]).float().to(device)

            lr_prob = lr_model(src_embed).sigmoid()
            loss += lr_criterion_eval(lr_prob, src_label).item()

            ground_truth[i] = label
            pred_prob[i] = lr_prob.cpu().numpy()

            g_num_list.append(k)
            raw_list.append(len(temp_src_cut))
            splitted_list.append(len(src_l_cut))
            label_list.append(label)
            prob_list.append(round(lr_prob.item(), 4))
            pred_list.append(1 if lr_prob >= 0.5 else 0)

    auc_roc = roc_auc_score(ground_truth, pred_prob)

    if data_type == 'test':
        df = pd.DataFrame({'g_num' : g_num_list,
                           'raw_edge_num' : raw_list,
                           'sliced_edge_num' : splitted_list,
                           'label' : label_list,
                           'prob' : prob_list,
                           'pred' : pred_list})

        # df.to_csv('./saved_data/{}-{}-{}-{}.csv'.format(DATA, time_cut, 'mean', round(auc_roc, 4)))
        print('df saved')

    return auc_roc, loss / num_instance

for epoch in tqdm(range(args.n_epoch)):
    tgan = tgan.eval()
    lr_model = lr_model.train()

    zero_g = []
    one_g = []

    for k in set(train_g_num_l):
        temp_src_cut = train_src_l[train_g_num_l == k]
        temp_ts_cut = train_ts_l[train_g_num_l == k]

        valid_flag = (temp_ts_cut < time_cut)
        src_l_cut = temp_src_cut[valid_flag]

        if len(src_l_cut) < 2:
            continue

        if False in valid_flag:
            one_g.append(k)
        else:
            zero_g.append(k)

    if len(one_g) > len(zero_g):
        one_g = random.sample(one_g, len(zero_g))
    else:
        zero_g = random.sample(zero_g, len(one_g))

    g_l = one_g + zero_g

    for k in g_l:
        temp_src_cut = train_src_l[train_g_num_l == k]
        temp_ts_cut = train_ts_l[train_g_num_l == k]

        valid_flag = (temp_ts_cut < time_cut)

        src_l_cut = temp_src_cut[valid_flag]
        ts_l_cut = temp_ts_cut[valid_flag]

        label = 0
        if False in valid_flag:
            label = 1

        if len(src_l_cut) < 2:
            continue

        lr_optimizer.zero_grad()
        with torch.no_grad():
            src_embed = tgan.tem_conv(src_l_cut, ts_l_cut, NODE_LAYER)

        src_label = torch.tensor([label]).float().to(device)

        lr_prob = lr_model(src_embed).sigmoid()
        lr_loss = lr_criterion(lr_prob, src_label)
        lr_loss.backward()
        lr_optimizer.step()

    train_auc, train_loss = eval_epoch(train_src_l, train_ts_l, train_g_num_l, lr_model, tgan, 'train')
    test_auc, test_loss = eval_epoch(test_src_l, test_ts_l, test_g_num_l, lr_model, tgan, 'train')
    #torch.save(lr_model.state_dict(), './saved_models/edge_{}_wkiki_node_class.pth'.format(DATA))
    logger.info(f'train auc: {train_auc}, test auc: {test_auc}')

test_auc, test_loss = eval_epoch(test_src_l, test_ts_l, test_g_num_l, lr_model, tgan, 'test')
# torch.save(lr_model.state_dict(), './saved_models/{}-{}-{}-{}.pth'.format(DATA, time_cut, 'mean', round(test_auc, 4)))
logger.info(f'test auc: {test_auc}')