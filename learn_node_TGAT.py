import os
import logging
import time
import sys
import random
import argparse

from tqdm import tqdm
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, recall_score
from collections import Counter

from module import TGAN
from graph import NeighborFinder
from utils import EarlyStopMonitor

class MEAN(torch.nn.Module):
    def __init__(self, input_dim, post_concat, NUM_FC, fc_dim, drop=0.3):
        super().__init__()

        self.input_dim = input_dim
        self.post_concat = post_concat
        self.NUM_FC = NUM_FC
        self.fc_dim = fc_dim

        if self.NUM_FC == 2:
            self.fc_1 = torch.nn.Linear(self.input_dim + self.post_concat * self.input_dim, self.fc_dim)
            self.fc_2 = torch.nn.Linear(self.fc_dim, 1)

        elif self.NUM_FC == 1:
            self.fc_1 = torch.nn.Linear(self.input_dim + self.post_concat * self.input_dim, 1)

        self.act = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=drop, inplace=True)

    def forward(self, x, k):
        if self.post_concat:
            k = k.astype(np.int64)
            post_k = torch.from_numpy(n_feat[k]).to(device)
            x = x.mean(dim=0)
            x = torch.cat((x, post_k), axis=0)
            x = self.dropout(x)

        else:
            x = x.mean(dim=0)
            x = self.dropout(x)

        if self.NUM_FC == 2:
            x = self.act(self.fc_1(x))
            x = self.dropout(x)
            x = self.fc_2(x)

        elif self.NUM_FC == 1:
            x = self.fc_1(x)

        return x


class LSTM(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, lstm_layer, bidirectional,
                 fc_dim, seq_len, sampling_method, NUM_FC, attention, post_concat, drop=0.3):
        super().__init__()

        self.post_concat = post_concat
        self.output_dim = output_dim
        self.num_layers = int(lstm_layer)
        self.input_size = input_dim
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        self.fc_dim = fc_dim
        self.seq_len = seq_len
        self.sampling_method = sampling_method
        self.NUM_FC = NUM_FC
        self.ATTENTION = attention

        self.lstm = torch.nn.LSTM(input_size=self.input_size,
                                  hidden_size=self.hidden_dim,
                                  num_layers=self.num_layers,
                                  bidirectional=self.bidirectional)
        if NUM_FC == 2:
            if bidirectional:
                self.fc_1 = torch.nn.Linear(self.hidden_dim * 2 + self.post_concat * self.input_size, self.fc_dim)

            else:
                self.fc_1 = torch.nn.Linear(self.hidden_dim + self.post_concat * self.input_size, self.fc_dim)

            self.fc_2 = torch.nn.Linear(fc_dim, self.output_dim)


        elif NUM_FC == 1:
            if bidirectional:
                self.fc_1 = torch.nn.Linear(self.hidden_dim * 2 + self.post_concat * self.input_size, self.output_dim)

            else:
                self.fc_1 = torch.nn.Linear(self.hidden_dim + self.post_concat * self.input_size, self.output_dim)

        self.act = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(drop)
        self.softmax = torch.nn.Softmax(dim=0)

    def forward(self, emb, k):

        if emb.shape[0] > self.seq_len:
            if sampling_method == 'NEWEST':
                embedded = emb[-self.seq_len:, :]

            elif sampling_method == 'OLDEST':
                embedded = emb[:self.seq_len, :]

            elif sampling_method == 'RANDOM':
                sampled_idx = np.sort(np.random.choice(len(emb), self.seq_len, replace=False))
                embedded = torch.vstack([emb[sampled_idx]])

        else:
            p2d = (0, 0, 0, self.seq_len - emb.shape[0])
            embedded = torch.nn.functional.pad(emb, p2d)
        embedded = torch.unsqueeze(embedded, 1)

        output, (h_out, c_out) = self.lstm(embedded)

        if self.bidirectional:
            h_out = self.dropout(torch.cat((h_out[-2, :, :], h_out[-1, :, :]), dim=1))
        else:
            h_out = self.dropout(h_out[-1, :, :])

        if self.ATTENTION:
            output = output[:, -1, :]
            context_v = h_out.view([-1, 1])

            attn_score = self.softmax(torch.mm(output, context_v)).view([1, -1])
            h_out = self.dropout(torch.matmul(attn_score, output))

        if self.post_concat:
            k = k.astype(np.int64)
            post_k = torch.from_numpy(n_feat[k]).to(device)
            h_out = torch.unsqueeze(torch.cat((h_out[0], post_k), axis=0), 0)

        if self.NUM_FC == 2:
            h_out = self.act(self.fc_1(h_out))
            h_out = self.dropout(h_out)
            result = self.fc_2(h_out)

        elif self.NUM_FC == 1:
            result = self.fc_1(h_out)

        return result.squeeze(1)


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
parser.add_argument('--n_epoch', type=int, default=50, help='number of epochs')
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
NUM_LAYER = args.n_layer
LEARNING_RATE = args.lr
NODE_LAYER = 1
NODE_DIM = args.node_dim
TIME_DIM = args.time_dim
max_round = 10

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

# GPU configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load hyperparameters of latest implementation
test_list = pd.read_csv('test_list.csv', index_col=0)

MODEL_PERFORMANCE_PATH = f'./saved_models/model_perfomance_eval.csv'
try:
    saved_model = pd.read_csv(MODEL_PERFORMANCE_PATH, index_col=0)
    MODEL_NUM = saved_model.index[-1] + 1
except:
    MODEL_NUM = 0

spec = test_list.loc[MODEL_NUM]

SUBREDDIT, WORD_EMBEDDING, EMBEDDING_METHOD, tgat_time_cut, time_cut, PRED_METHOD, sampling_method, post_concat, bidirectional, \
lstm_layer, NUM_FC, fc_dim, hidden_dim, SEQ_SLICING, TRAINING_METHOD, CLASS_BALANCING = spec[:]
bidirectional = bool(bidirectional)
post_concat = bool(post_concat)
hidden_dim = int(hidden_dim)
fc_dim = int(fc_dim)
output_dim = 1

if PRED_METHOD == 'MEAN':
    n_layer = None
    bidirectional = False
    sampling_method = None

MODEL_NUM = str(MODEL_NUM).zfill(3)
logger.info(f'MODEL_NUM : {MODEL_NUM}')

MODEL_SAVE_PATH = f'./saved_models/predict_models/{MODEL_NUM}-PREDICT.pth'
get_checkpoint_path = lambda \
        epoch: f'./saved_checkpoints/{MODEL_NUM}-PREDICT-{epoch}.pth'

if EMBEDDING_METHOD != 'TGAT':
    logger.info('Only for Pretrained TGAT Model')
    logger.info('Exit Program')
    exit()

# Load data and train val test split
g_df = pd.read_csv(f'./processed/{SUBREDDIT}_structure.csv', index_col=0)
n_feat = np.load(f'./processed/{SUBREDDIT}_node_feat_{WORD_EMBEDDING}.npy')
e_feat = np.load(f'./processed/{SUBREDDIT}_edge_feat_{n_feat.shape[1]}.npy')

g_num = g_df.g_num.values
g_ts = g_df.g_ts.values
src_l = g_df.u.values
dst_l = g_df.i.values
e_idx_l = g_df.idx.values
label_l = g_df.label.values
ts_l = g_df.ts.values

train_time = 3888000
test_time = np.quantile(np.unique(g_ts[(g_ts > train_time)]), 0.5)

max_src_index = src_l.max()
max_idx = max(src_l.max(), dst_l.max())

train_flag = (g_ts < train_time)
test_flag = (g_ts >= test_time)
val_flag = (g_ts >= train_time) & (g_ts < test_time)

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

val_g_num_l = g_num[val_flag]
val_src_l = src_l[val_flag]
val_dst_l = dst_l[val_flag]
val_ts_l = ts_l[val_flag]
val_label_l = label_l[val_flag]
val_e_idx_l = e_idx_l[val_flag]

total_src_l = np.hstack((train_src_l, test_src_l, val_src_l))
total_dst_l = np.hstack((train_dst_l, test_dst_l, val_dst_l))
total_ts_l = np.hstack((train_ts_l, test_ts_l, val_ts_l))
total_e_idx_l = np.hstack((train_e_idx_l, test_e_idx_l, val_e_idx_l))

cntr = Counter(g_num)
MAX_SEQ = cntr.most_common(1)[0][1]

if 0 < SEQ_SLICING:
    MAX_SEQ = round(np.quantile(list(cntr.values()), SEQ_SLICING))
elif SEQ_SLICING < 0:
    MAX_SEQ = int(-SEQ_SLICING)

# Graph labeling
graph_label_map = {}
for k in set(g_num):
    temp_ts_l = ts_l[g_num == k]
    valid_flag = (temp_ts_l < time_cut)

    if sum(valid_flag) < 2:
        continue

    if False in valid_flag:
        graph_label_map[k] = 1
    else:
        graph_label_map[k] = 0

# Class balancing in train data
if CLASS_BALANCING == 'BALANCED':
    neg_k = np.array([])
    pos_k = np.array([])

    for k in set(train_g_num_l):
        if k in graph_label_map:
            if graph_label_map[k]:
                pos_k = np.append(pos_k, k)
            else:
                neg_k = np.append(neg_k, k)

    if len(pos_k) > len(neg_k):
        major, minor = pos_k, neg_k
    else:
        major, minor = neg_k, pos_k

# Initialize the data structure for graph and edge sampling
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


tgan = TGAN(train_ngh_finder, n_feat, e_feat,
            num_layers=NUM_LAYER, use_time=USE_TIME, agg_method=AGG_METHOD, attn_mode=ATTN_MODE,
            seq_len=SEQ_LEN, n_head=NUM_HEADS, drop_out=DROP_OUT, node_dim=NODE_DIM, time_dim=TIME_DIM)
optimizer = torch.optim.Adam(tgan.parameters(), lr=LEARNING_RATE)
criterion = torch.nn.BCELoss()
tgan = tgan.to(device)

num_instance = len(train_src_l)
logger.debug(f'num of training instances: {num_instance}')
logger.debug(f'num of graphs per epoch: {len(set(train_g_num_l))}')

logger.info('loading saved TGAN model')

#select pre-trained model
pretrained_model = f'TGAT-{SUBREDDIT}-{WORD_EMBEDDING}-{TRAINING_METHOD}.pth'
logger.info(f'TGAN Model : {pretrained_model}')
logger.info(spec[:])

model_path = f'./saved_models/pretrained_models/{pretrained_model}'
tgan.load_state_dict(torch.load(model_path))
tgan.eval()
logger.info(f'TGAN models loaded : {pretrained_model}')
logger.info('Start training Graph classification task')

if PRED_METHOD == 'LSTM':
    lr_model = LSTM(n_feat.shape[1], hidden_dim, output_dim, lstm_layer, bidirectional,
                    fc_dim, MAX_SEQ, sampling_method, NUM_FC, 0, post_concat)
elif PRED_METHOD == 'ATTENTION':
    lr_model = LSTM(n_feat.shape[1], hidden_dim, output_dim, lstm_layer, bidirectional,
                    fc_dim, MAX_SEQ, sampling_method, NUM_FC, 1, post_concat)
elif PRED_METHOD == 'MEAN':
    lr_model = MEAN(n_feat.shape[1], post_concat, NUM_FC, fc_dim)

lr_model = lr_model.to(device)
lr_optimizer = torch.optim.Adam(lr_model.parameters(), lr=args.lr)
tgan.ngh_finder = full_ngh_finder
lr_criterion = torch.nn.BCELoss()
lr_criterion_eval = torch.nn.BCELoss()

early_stopper = EarlyStopMonitor(max_round)

def eval_epoch(src_l, ts_l, g_num_l, lr_model, tgan, data_type, num_layer=NODE_LAYER):

    graph_num = np.array([])
    raw_edge_len = np.array([])
    sliced_edge_len = np.array([])
    true_label = np.array([])
    pred_prob = np.array([])
    pred_label = np.array([])

    loss = 0
    g_l = np.unique(g_num_l)
    np.random.shuffle(g_l)
    with torch.no_grad():
        lr_model.eval()
        tgan.eval()

        for i, k in enumerate(g_l):
            if k not in graph_label_map:
                continue
            temp_src_cut = src_l[g_num_l == k]
            temp_ts_cut = ts_l[g_num_l == k]

            valid_flag = (temp_ts_cut < time_cut)

            src_l_cut = temp_src_cut[valid_flag]
            ts_l_cut = temp_ts_cut[valid_flag]
            label = graph_label_map[k]

            src_embed = tgan.tem_conv(src_l_cut, ts_l_cut, num_layer)
            src_label = torch.cuda.FloatTensor([label])

            lr_prob = lr_model(src_embed, k).sigmoid()
            loss += lr_criterion_eval(lr_prob, src_label).item()

            graph_num = np.append(graph_num, k)
            raw_edge_len = np.append(raw_edge_len, len(temp_src_cut))
            sliced_edge_len = np.append(sliced_edge_len, len(src_l_cut))
            true_label = np.append(true_label, label)
            pred_prob = np.append(pred_prob, lr_prob.cpu().numpy())
            pred_label = np.append(pred_label, np.round(lr_prob.cpu().numpy()))

    label_ratio = sum(true_label)/len(true_label)
    acc = (true_label == pred_label).mean()
    auc_roc = roc_auc_score(true_label, pred_prob)
    AP = average_precision_score(true_label, pred_prob)
    recall = recall_score(true_label, pred_label)
    F1 = f1_score(true_label, pred_label)

    # save performance
    if data_type:
        new = [{'SUBREDDIT' : SUBREDDIT,
                'WORD_EMBEDDING' : WORD_EMBEDDING,
                'EMBEDDING_METHOD': EMBEDDING_METHOD,
                'TGAT_TIME_CUT': tgat_time_cut,
                'PRED_TIME_CUT': time_cut,
                'PRED_METHOD': PRED_METHOD,
                'SAMPLING_METHOD': sampling_method,
                'POST_CONCAT': post_concat,
                'BIDIRECTIONAL': bidirectional,
                'NUM_LAYER': lstm_layer,
                'NUM_FC': NUM_FC,
                'FC_DIM': fc_dim,
                'HIDDEN_DIM': hidden_dim,
                'MAX_SEQ_QUANTILE': SEQ_SLICING,
                'TRAINING_METHOD': TRAINING_METHOD,
                'CLASS_BALANCING': CLASS_BALANCING,
                'NUM_GRAPH': len(graph_num),
                'RAW_EDGE_NUM': sum(raw_edge_len),
                'USED_EDGE_NUM': sum(sliced_edge_len),
                'LABEL_RATIO': label_ratio,
                'ACCURACY': acc,
                'AUC_ROC_SCORE': auc_roc,
                'AP_SCORE': AP,
                'RECALL_SCORE': recall,
                'F1_SCORE': F1}]
        new_model = pd.DataFrame.from_dict(new)
        try:
            saved_model = pd.read_csv(MODEL_PERFORMANCE_PATH, index_col=0)
            updated_model = saved_model.append(new_model)
            updated_model = updated_model.reset_index(drop=True)
            updated_model.to_csv(MODEL_PERFORMANCE_PATH)
        except:
            new_model.to_csv(MODEL_PERFORMANCE_PATH)

        #save model details
        df = pd.DataFrame({'g_num': graph_num,
                           'raw_edge_num': raw_edge_len,
                           'sliced_edge_num': sliced_edge_len,
                           'true_label': true_label,
                           'pred_prob': pred_prob,
                           'pred_label': pred_label})

        df.to_csv(f'./saved_data/{MODEL_NUM}.csv')

    return acc, auc_roc, AP, recall, F1, loss / num_instance

for epoch in tqdm(range(NUM_EPOCH)):
    tgan = tgan.eval()
    lr_model = lr_model.train()

    # random choice
    if CLASS_BALANCING == 'BALANCED':
        major = np.random.choice(major, len(minor), replace=False)
        g_l = np.hstack((major, minor))
    else:
        g_l = np.unique(train_g_num_l)

    np.random.shuffle(g_l)

    for k in g_l:
        src_l_cut = train_src_l[train_g_num_l == k]
        ts_l_cut = train_ts_l[train_g_num_l == k]
        label = graph_label_map[k]

        lr_optimizer.zero_grad()
        with torch.no_grad():
            src_embed = tgan.tem_conv(src_l_cut, ts_l_cut, NODE_LAYER)

        src_label = torch.cuda.FloatTensor([label])

        lr_prob = lr_model(src_embed, k).sigmoid()
        lr_loss = lr_criterion(lr_prob, src_label)
        lr_loss.backward()
        lr_optimizer.step()

    val_acc, val_auc, val_AP, val_recall, val_F1, val_loss = eval_epoch(val_src_l, val_ts_l, val_g_num_l, lr_model, tgan, 0)
    logger.info(f'epoch: {epoch}:')
    logger.info(f'val loss: {val_loss}')
    logger.info(f'val acc: {val_acc}')
    logger.info(f'val auc: {val_auc}')
    logger.info(f'val ap: {val_AP}')
    logger.info(f'val f1: {val_F1}')

    if early_stopper.early_stop_check(val_auc):
        logger.info(f'No improvment over {early_stopper.max_round} epochs, stop training')
        best_epoch = early_stopper.best_epoch
        logger.info(f'Loading the best model at epoch {best_epoch}')
        best_model_path = get_checkpoint_path(best_epoch)
        lr_model.load_state_dict(torch.load(best_model_path))
        logger.info(f'Loaded the best model at epoch {best_epoch} for inference')
        os.remove(best_model_path)
        break
    else:
        if early_stopper.is_best:
            torch.save(lr_model.state_dict(), get_checkpoint_path(epoch))
            logger.info(f'Saved {MODEL_NUM}-PREDICT-{early_stopper.best_epoch}.pth')
            for i in range(epoch):
                try:
                    os.remove(get_checkpoint_path(i))
                    logger.info(f'Deleted {MODEL_NUM}-PREDICT-{i}.pth')
                except:
                    continue

test_acc, test_auc, test_AP, test_recall, test_F1, test_loss = eval_epoch(test_src_l, test_ts_l, test_g_num_l, lr_model, tgan, 1)
logger.info(f'test acc: {test_acc}, test auc: {test_auc}, test AP: {test_AP}, test Recall_Score: {test_recall}, test F1: {test_F1}')
torch.save(lr_model.state_dict(), MODEL_SAVE_PATH)
