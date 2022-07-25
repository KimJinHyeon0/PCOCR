import os
import math
import logging
import time
import random
import sys
import argparse

import torch
import pandas as pd
import numpy as np

from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, recall_score

from module import TGAN
from graph import NeighborFinder
from utils import EarlyStopMonitor, RandEdgeSampler

### Argument and global variables
parser = argparse.ArgumentParser('Interface for TGAT experiments on link predictions')
parser.add_argument('-d', '--data', type=str, help='data sources to use, try wikipedia or reddit', default='iama')
parser.add_argument('--bs', type=int, default=200, help='batch_size')
parser.add_argument('--prefix', type=str, default='', help='prefix to name the checkpoints')
parser.add_argument('--n_degree', type=int, default=20, help='number of neighbors to sample')
parser.add_argument('--n_head', type=int, default=2, help='number of heads used in attention layer')
parser.add_argument('--n_epoch', type=int, default=50, help='number of epochs')
parser.add_argument('--n_layer', type=int, default=2, help='number of network layers')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('--drop_out', type=float, default=0.1, help='dropout probability')
parser.add_argument('--gpu', type=int, default=0, help='idx for the gpu to use')
parser.add_argument('--node_dim', type=int, default=100, help='Dimentions of the node embedding')
parser.add_argument('--time_dim', type=int, default=100, help='Dimentions of the time embedding')
parser.add_argument('--agg_method', type=str, choices=['attn', 'lstm', 'mean'], help='local aggregation method',
                    default='attn')
parser.add_argument('--attn_mode', type=str, choices=['prod', 'map'], default='prod',
                    help='use dot product attention or mapping based')
parser.add_argument('--time', type=str, choices=['time', 'pos', 'empty'], help='how to use time information',
                    default='time')
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
# NEW_NODE = args.new_node
USE_TIME = args.time
AGG_METHOD = args.agg_method
ATTN_MODE = args.attn_mode
SEQ_LEN = NUM_NEIGHBORS
NUM_LAYER = args.n_layer
LEARNING_RATE = args.lr
NODE_DIM = args.node_dim
TIME_DIM = args.time_dim

'''
=====CONFIGS=====

        SUBREDDIT : 'iama'
                    'showerthoughts'
        
        TRAINING_METHOD = 'SELECTIVE'
                          'FULL'
                          
        WORD_EMBEDDING : 'bert-base-uncased'  
                         'roberta-base'
                         'deberta-base'
                         'fasttext'
                         'glove' 
                         'tf-idf'

        TIME_CUT : int

        max_round : int

=====CONFIGS=====
'''
SUBREDDIT = 'iama'
TRAINING_METHOD = 'SELECTIVE'
WORD_EMBEDDING = 'bert-base-uncased'
TIME_CUT = 309000
max_round = 10

MODEL_SAVE_PATH = f'./saved_models/TGAT-{SUBREDDIT}-{WORD_EMBEDDING}-{TRAINING_METHOD}.pth'
get_checkpoint_path = lambda \
    epoch: f'./saved_checkpoints/TGAT-{SUBREDDIT}-{WORD_EMBEDDING}-{TRAINING_METHOD}-{epoch}.pth'

### set up logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler(f'log/{str(time.time())}.log')
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.WARN)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)
logger.info(args)

def random_shuffle(num_instances, src, dst, ts, label):
    indices = np.arange(num_instances)
    np.random.shuffle(indices)
    return src[indices], dst[indices], ts[indices], label[indices]

def eval_one_epoch(hint, tgan, sampler, src, dst, ts, label):
    acc, ap, f1, auc = [], [], [], []
    with torch.no_grad():
        tgan = tgan.eval()
        num_test_instance = len(src)
        num_test_batch = math.ceil(num_test_instance / BATCH_SIZE)

        src, dst, ts, label = random_shuffle(num_test_instance, src, dst, ts, label)

        for k in range(num_test_batch):
            # percent = 100 * k / num_test_batch
            # if k % int(0.2 * num_test_batch) == 0:
            #     logger.info('{0} progress: {1:10.4f}'.format(hint, percent))
            s_idx = k * BATCH_SIZE
            e_idx = min(num_test_instance - 1, s_idx + BATCH_SIZE)
            src_l_cut = src[s_idx:e_idx]
            dst_l_cut = dst[s_idx:e_idx]
            ts_l_cut = ts[s_idx:e_idx]
            label_l_cut = label[s_idx:e_idx]

            pred_prob = tgan.contrast(src_l_cut, dst_l_cut, ts_l_cut, NUM_NEIGHBORS).cpu().numpy()
            pred_label = pred_prob > 0.5
            true_label = label_l_cut

            acc.append((pred_label == true_label).mean())
            ap.append(average_precision_score(true_label, pred_prob))
            f1.append(f1_score(true_label, pred_label))
            auc.append(roc_auc_score(true_label, pred_prob))

    return np.mean(acc), np.mean(ap), np.mean(f1), np.mean(auc)


### Load data and train val test split
g_df = pd.read_csv(f'./processed/{SUBREDDIT}_structure.csv', index_col=0)
n_feat = np.load(f'./processed/{SUBREDDIT}_node_feat_{WORD_EMBEDDING}.npy')
e_feat = np.load(f'./processed/{SUBREDDIT}_edge_feat_{n_feat.shape[1]}.npy')

g_num = g_df.g_num.values
g_ts = g_df.g_ts.values
src_l = g_df.u.values
dst_l = g_df.i.values
ts_l = g_df.ts.values
label_l = g_df.label.values
e_idx_l = g_df.idx.values

train_time = 3888000  # '2018-02-15 00:00:00' - '2018-01-01 00:00:00'
test_time = np.quantile(g_ts[(g_ts > train_time)], 0.5)

train_flag = (g_ts < train_time)
test_flag = (g_ts >= train_time) & (g_ts < test_time)
valid_flag = (g_ts >= test_time)

train_g_num = g_num[train_flag]
train_src_l = src_l[train_flag]
train_dst_l = dst_l[train_flag]
train_ts_l = ts_l[train_flag]
train_label_l = label_l[train_flag]
train_e_idx_l = e_idx_l[train_flag]

val_src_l = src_l[valid_flag]
val_dst_l = dst_l[valid_flag]
val_ts_l = ts_l[valid_flag]
val_label_l = label_l[valid_flag]
val_e_idx_l = e_idx_l[valid_flag]

test_src_l = src_l[test_flag]
test_dst_l = dst_l[test_flag]
test_ts_l = ts_l[test_flag]
test_label_l = label_l[test_flag]
test_e_idx_l = e_idx_l[test_flag]

np.random.seed(2020)
logger.info(f'SUBREDDIT : {SUBREDDIT}')
logger.info(f'TRAINING METHOD : {TRAINING_METHOD}')
logger.info(f'TIME_CUT : {TIME_CUT}')
logger.info(f'WORD_EMBEDDING : {WORD_EMBEDDING}')

if TRAINING_METHOD == 'SELECTIVE':
    time_cut_flag = (train_ts_l < TIME_CUT)

    train_g_num = train_g_num[time_cut_flag]
    train_src_l = train_src_l[time_cut_flag]
    train_dst_l = train_dst_l[time_cut_flag]
    train_ts_l = train_ts_l[time_cut_flag]
    train_label_l = train_label_l[time_cut_flag]
    train_e_idx_l = train_e_idx_l[time_cut_flag]

    train_index_l = np.arange(len(train_g_num))
    selective_flag = np.zeros((len(train_g_num)), dtype=bool)

    for k in set(train_g_num):
        temp_flag = (train_g_num == k)

        temp_index = train_index_l[temp_flag]
        temp_label = train_label_l[temp_flag]

        if temp_label.mean() >= 0.5:
            major, minor = temp_index[temp_label == 1], temp_index[temp_label == 0]

        else:
            major, minor = temp_index[temp_label == 0], temp_index[temp_label == 1]

        balanced = np.random.choice(major, len(minor), replace=False)
        selective_flag[balanced] = True
        selective_flag[minor] = True

    train_src_l = train_src_l[selective_flag]
    train_dst_l = train_dst_l[selective_flag]
    train_ts_l = train_ts_l[selective_flag]
    train_label_l = train_label_l[selective_flag]
    train_e_idx_l = train_e_idx_l[selective_flag]

total_src_l = np.hstack((train_src_l, test_src_l, val_src_l))
total_dst_l = np.hstack((train_dst_l, test_dst_l, val_dst_l))
total_ts_l = np.hstack((train_ts_l, test_ts_l, val_ts_l))
total_e_idx_l = np.hstack((train_e_idx_l, test_e_idx_l, val_e_idx_l))

### Initialize the data structure for graph and edge sampling
# build the graph for fast query
# graph only contains the training data (with 10% nodes removal)

max_idx = max(src_l.max(), dst_l.max())

adj_list = [[] for _ in range(max_idx + 1)]
for src, dst, eidx, ts in zip(train_src_l, train_dst_l, train_e_idx_l, train_ts_l):
    adj_list[src].append((dst, eidx, ts))
    adj_list[dst].append((src, eidx, ts))
train_ngh_finder = NeighborFinder(adj_list, uniform=UNIFORM)

# full graph with all the data for the test and validation purpose
full_adj_list = [[] for _ in range(max_idx + 1)]
for src, dst, eidx, ts in zip(total_src_l, total_dst_l, total_e_idx_l, total_ts_l):
    full_adj_list[src].append((dst, eidx, ts))
    full_adj_list[dst].append((src, eidx, ts))
full_ngh_finder = NeighborFinder(full_adj_list, uniform=UNIFORM)

train_rand_sampler = RandEdgeSampler(train_src_l, train_dst_l)
val_rand_sampler = RandEdgeSampler(val_src_l, val_dst_l)
# nn_val_rand_sampler = RandEdgeSampler(nn_val_src_l, nn_val_dst_l)
test_rand_sampler = RandEdgeSampler(test_src_l, test_dst_l)
# nn_test_rand_sampler = RandEdgeSampler(nn_test_src_l, nn_test_dst_l)

### Model initialize
device = torch.device(f'cuda:{GPU}')
tgan = TGAN(train_ngh_finder, n_feat, e_feat,
            num_layers=NUM_LAYER, use_time=USE_TIME, agg_method=AGG_METHOD, attn_mode=ATTN_MODE,
            seq_len=SEQ_LEN, n_head=NUM_HEADS, drop_out=DROP_OUT, node_dim=NODE_DIM, time_dim=TIME_DIM)

optimizer = torch.optim.Adam(tgan.parameters(), lr=LEARNING_RATE)
criterion = torch.nn.BCELoss()
tgan = tgan.to(device)

num_instance = len(train_src_l)
num_batch = math.ceil(num_instance / BATCH_SIZE)

logger.info(f'num of training instances: {num_instance}')
logger.info(f'num of batches per epoch: {num_batch}')
early_stopper = EarlyStopMonitor(max_round)

for epoch in range(NUM_EPOCH):  # NUM_EPOCH = 50
    # Training
    # training use only training graph
    tgan.ngh_finder = train_ngh_finder
    acc, ap, f1, auc, m_loss = [], [], [], [], []
    train_src_l, train_dst_l, train_ts_l, train_label_l = \
        random_shuffle(num_instance, train_src_l, train_dst_l, train_ts_l, train_label_l)
    logger.info(f'start {epoch} epoch')
    for k in range(num_batch):  # num_batch = case_num / BATCH_SIZE(200)
        # percent = 100 * k / num_batch
        # if k % int(0.2 * num_batch) == 0:
        #     logger.info('progress: {0:10.4f}'.format(percent))
        s_idx = k * BATCH_SIZE  # BATCH_SIZE = 200
        e_idx = min(num_instance - 1, s_idx + BATCH_SIZE)
        src_l_cut, dst_l_cut = train_src_l[s_idx:e_idx], train_dst_l[s_idx:e_idx]
        ts_l_cut = train_ts_l[s_idx:e_idx]
        label_l_cut = train_label_l[s_idx:e_idx]

        optimizer.zero_grad()
        tgan = tgan.train()
        pred_prob = tgan.contrast(src_l_cut, dst_l_cut, ts_l_cut, NUM_NEIGHBORS)
        loss = criterion(pred_prob, torch.from_numpy(label_l_cut).float().to(device))

        loss.backward()
        optimizer.step()

        # get training results
        with torch.no_grad():
            tgan = tgan.eval()
            pred_prob = pred_prob.cpu().detach().numpy()
            pred_label = pred_prob > 0.5
            true_label = label_l_cut
            acc.append((pred_label == true_label).mean())
            ap.append(average_precision_score(true_label, pred_prob))
            f1.append(f1_score(true_label, pred_label))
            m_loss.append(loss.item())
            auc.append(roc_auc_score(true_label, pred_prob))

    # validation phase use all information
    tgan.ngh_finder = full_ngh_finder
    val_acc, val_ap, val_f1, val_auc = eval_one_epoch('val for old nodes', tgan, val_rand_sampler,
                                                      val_src_l, val_dst_l, val_ts_l, val_label_l)

    # nn_val_acc, nn_val_ap, nn_val_f1, nn_val_auc = eval_one_epoch('val for new nodes', tgan, val_rand_sampler,
    #                                                               nn_val_src_l,
    #                                                               nn_val_dst_l, nn_val_ts_l, nn_val_label_l)

    logger.info(f'epoch: {epoch}:')
    logger.info(f'Epoch mean loss: {np.mean(m_loss)}')
    logger.info(f'train acc: {np.mean(acc)}, val acc: {val_acc}')
    logger.info(f'train auc: {np.mean(auc)}, val auc: {val_auc}')
    logger.info(f'train ap: {np.mean(ap)}, val ap: {val_ap}')
    logger.info(f'train f1: {np.mean(f1)}, val f1: {val_f1}')

    if early_stopper.early_stop_check(val_auc):
        logger.info(f'No improvment over {early_stopper.max_round} epochs, stop training')
        best_epoch = early_stopper.best_epoch
        logger.info(f'Loading the best model at epoch {best_epoch}')
        best_model_path = get_checkpoint_path(best_epoch)
        tgan.load_state_dict(torch.load(best_model_path))
        logger.info(f'Loaded the best model at epoch {best_epoch} for inference')
        tgan.eval()
        os.remove(best_model_path)
        break
    else:
        if early_stopper.is_best:
            torch.save(tgan.state_dict(), get_checkpoint_path(epoch))
            logger.info(f'Saved TGAT-{SUBREDDIT}-{WORD_EMBEDDING}-{TRAINING_METHOD}-{early_stopper.best_epoch}.pth')
            for i in range(epoch):
                try:
                    os.remove(get_checkpoint_path(i))
                    logger.info(f'Deleted TGAT-{SUBREDDIT}-{WORD_EMBEDDING}-{TRAINING_METHOD}-{i}.pth')
                except:
                    continue

# testing phase use all information
tgan.ngh_finder = full_ngh_finder
test_acc, test_ap, test_f1, test_auc = eval_one_epoch('test for old nodes', tgan, test_rand_sampler, test_src_l,
                                                      test_dst_l, test_ts_l, test_label_l)

# nn_test_acc, nn_test_ap, nn_test_f1, nn_test_auc = eval_one_epoch('test for new nodes', tgan, nn_test_rand_sampler, nn_test_src_l, nn_test_dst_l, nn_test_ts_l, nn_test_label_l)

logger.info(f'Test statistics: Old nodes -- acc: {test_acc}, auc: {test_auc}, ap: {test_ap}, f1: {test_ap}')
# logger.info('Test statistics: New nodes -- acc: {}, auc: {}, ap: {}, f1: {}'.format(nn_test_acc, nn_test_auc, nn_test_ap, nn_test_f1))

logger.info('Saving TGAN model')
torch.save(tgan.state_dict(), MODEL_SAVE_PATH)
logger.info('TGAN models saved')