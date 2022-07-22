import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
import torch.nn.functional as F
import pandas as pd
import numpy as np
import os
import random
from sklearn.metrics import f1_score, roc_auc_score
from collections import defaultdict
import math
from utils import EarlyStopMonitor

"""
Simple supervised GraphSAGE model as well as examples running the model
on the Cora and Pubmed datasets.
"""


class MeanAggregator(nn.Module):
    """
    Aggregates a node's embeddings using mean of neighbors' embeddings
    """

    def __init__(self, features, cuda=True, gcn=False):
        """
        Initializes the aggregator for a specific graph.
        features -- function mapping LongTensor of node ids to FloatTensor of feature values.
        cuda -- whether to use GPU
        gcn --- whether to perform concatenation GraphSAGE-style, or add self-loops GCN-style
        """

        super(MeanAggregator, self).__init__()

        self.features = features
        self.cuda = cuda
        self.gcn = gcn

    def forward(self, nodes, to_neighs, num_sample=10):
        """
        nodes --- list of nodes in a batch
        to_neighs --- list of sets, each set is the set of neighbors for node in batch
        num_sample --- number of neighbors to sample. No sampling if None.
        """
        # Local pointers to functions (speed hack)
        _set = set
        if not num_sample is None:
            _sample = random.sample
            samp_neighs = [_set(_sample(to_neigh,
                                        num_sample,
                                        )) if len(to_neigh) >= num_sample else to_neigh for to_neigh in to_neighs]
        else:
            samp_neighs = to_neighs

        if self.gcn:
            samp_neighs = [samp_neigh + set([nodes[i]]) for i, samp_neigh in enumerate(samp_neighs)]
        unique_nodes_list = list(set.union(*samp_neighs))
        #  print ("\n unl's size=",len(unique_nodes_list))
        unique_nodes = {n: i for i, n in enumerate(unique_nodes_list)}
        mask = Variable(torch.zeros(len(samp_neighs), len(unique_nodes)))
        column_indices = [unique_nodes[n] for samp_neigh in samp_neighs for n in samp_neigh]
        row_indices = [i for i in range(len(samp_neighs)) for j in range(len(samp_neighs[i]))]
        mask[row_indices, column_indices] = 1
        if self.cuda:
            mask = mask.cuda()
        num_neigh = mask.sum(1, keepdim=True)

        mask = mask.div(num_neigh)
        if self.cuda:
            embed_matrix = self.features(torch.cuda.LongTensor(unique_nodes_list))
        else:
            embed_matrix = self.features(torch.LongTensor(unique_nodes_list))
        to_feats = mask.mm(embed_matrix)
        return to_feats


class Encoder(nn.Module):
    def __init__(self, features, feature_dim,
                 embed_dim, adj_lists, aggregator,
                 num_sample=10,
                 base_model=None, gcn=False, cuda=True,
                 feature_transform=False):
        super(Encoder, self).__init__()

        self.features = features
        self.feat_dim = feature_dim
        self.adj_lists = adj_lists
        self.aggregator = aggregator
        self.num_sample = num_sample
        if base_model != None:
            self.base_model = base_model

        self.gcn = gcn
        self.embed_dim = embed_dim
        self.cuda = cuda
        self.aggregator.cuda = cuda
        self.weight = nn.Parameter(
            torch.FloatTensor(embed_dim, self.feat_dim if self.gcn else 2 * self.feat_dim))
        init.xavier_uniform_(self.weight)
        self.dropout = torch.nn.Dropout(p=0.1, inplace=True)

    def forward(self, nodes):
        """
        Generates embeddings for a batch of nodes.
        nodes     -- list of nodes
        """
        neigh_feats = self.aggregator.forward(nodes, [self.adj_lists[int(node)] for node in nodes],
                                              self.num_sample)
        if not self.gcn:
            if self.cuda:
                self_feats = self.features(torch.cuda.LongTensor(nodes))

            else:
                self_feats = self.features(torch.LongTensor(nodes))
            combined = torch.cat([self_feats, neigh_feats], dim=1)

        else:
            combined = neigh_feats
        combined = F.relu(self.weight.mm(combined.t()))
        return self.dropout(combined)


class SupervisedGraphSage(nn.Module):

    def __init__(self, num_classes, enc):
        super(SupervisedGraphSage, self).__init__()
        self.enc = enc
        self.xent = nn.CrossEntropyLoss()
        self.embeds = 0
        self.weight = nn.Parameter(torch.cuda.FloatTensor(num_classes, enc.embed_dim))
        init.xavier_uniform_(self.weight)

    def forward(self, nodes):
        embeds = self.enc(nodes)
        self.embeds = embeds.t()
        scores = self.weight.mm(embeds)

        return scores.t()

    def loss(self, nodes, labels):
        scores = self.forward(nodes)

        return self.xent(scores, labels.squeeze()).to(device)


def load_data():

    print('Loading Data')
    g_df = pd.read_csv(f'./processed/{SUBREDDIT}_structure.csv', index_col=0)
    n_feat = np.load(f'./processed/{SUBREDDIT}_node_feat_{WORD_EMBEDDING}.npy')

    g_num = g_df.g_num.values
    g_ts = g_df.g_ts.values
    src_l = g_df.u.values
    dst_l = g_df.i.values
    ts_l = g_df.ts.values
    label_l = g_df.label.values

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

    val_src_l = src_l[valid_flag]
    val_dst_l = dst_l[valid_flag]
    val_label_l = label_l[valid_flag]

    test_src_l = src_l[test_flag]
    test_dst_l = dst_l[test_flag]
    test_label_l = label_l[test_flag]

    if TRAINING_METHOD == 'SELECTIVE':
        time_cut_flag = (train_ts_l < time_cut)

        train_g_num = train_g_num[time_cut_flag]
        train_src_l = train_src_l[time_cut_flag]
        train_dst_l = train_dst_l[time_cut_flag]
        train_label_l = train_label_l[time_cut_flag]

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
        train_label_l = train_label_l[selective_flag]

    total_src_l = np.hstack((train_src_l, test_src_l, val_src_l))
    total_dst_l = np.hstack((train_dst_l, test_dst_l, val_dst_l))

    adj_lists = defaultdict(set)
    for src, dst in zip(total_src_l, total_dst_l):
        adj_lists[src].add(dst)
        adj_lists[dst].add(src)

    return n_feat, adj_lists, train_src_l, train_label_l, test_src_l, test_label_l, val_src_l, val_label_l

def random_shuffle(num_instance, src, label):
    indices = np.arange(num_instance)
    np.random.shuffle(indices)
    return src[indices], label[indices]

def eval_one_epoch(graphsage, src_l, label_l):
    acc, ap, f1, auc = [], [], [], []
    with torch.no_grad():
        graphsage = graphsage.eval()
        num_instance = len(src_l)
        num_batch = math.ceil(num_instance / BATCH_SIZE)
        #random shuffle
        src_l, label_l = random_shuffle(num_instance, src_l, label_l)
        for k in range(num_batch):
            s_idx = k * BATCH_SIZE
            e_idx = min(num_instance - 1, s_idx + BATCH_SIZE)
            src_l_cut = src_l[s_idx:e_idx]
            label_l_cut = label_l[s_idx:e_idx]

            output = graphsage.forward(src_l_cut).data.cpu()
            pred_prob = output[np.arange(src_l_cut.shape[0]), label_l_cut.astype(np.int).squeeze()]
            pred_label = output.argmax(axis=1)
            auc.append(roc_auc_score(label_l_cut, pred_prob))
            f1.append(f1_score(label_l_cut, pred_label))

    return np.mean(auc), np.mean(f1)

np.random.seed(222)
random.seed(222)
def run():
    NUM_EPOCH = 1000

    feat_data, adj_lists, \
    train_src_l, train_label_l, \
    test_src_l, test_label_l, \
    val_src_l, val_label_l = load_data()
    print('Data Loaded')
    features = nn.Embedding(feat_data.shape[0], feat_data.shape[1])
    features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=False)
    features.cuda()

    agg1 = MeanAggregator(features, cuda=True)
    enc1 = Encoder(features, feat_data.shape[1], feat_data.shape[1], adj_lists, agg1, gcn=False, cuda=True)
    agg2 = MeanAggregator(lambda nodes: enc1(nodes).t(), cuda=True)
    enc2 = Encoder(lambda nodes: enc1(nodes).t(), enc1.embed_dim, feat_data.shape[1], adj_lists, agg2,
                   base_model=enc1, gcn=False, cuda=True)
    enc1.num_samples = 5
    enc2.num_samples = 5

    graphsage = SupervisedGraphSage(2, enc2)
    graphsage.to(device)
    early_stopper = EarlyStopMonitor(max_round)

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, graphsage.parameters()), lr=0.0005)

    num_instance = len(train_src_l)
    print(f'num_instance : {num_instance}')
    num_batch = math.ceil(num_instance / BATCH_SIZE)

    for epoch in range(NUM_EPOCH):
        m_loss = []

        #random shuffle
        train_src_l, train_label_l = random_shuffle(num_instance, train_src_l, train_label_l)
        print(f'Start {epoch} epoch')
        for k in range(num_batch):
            s_idx = k * BATCH_SIZE
            e_idx = min(num_instance - 1, s_idx + BATCH_SIZE)
            src_l_cut = train_src_l[s_idx:e_idx]
            label_l_cut = train_label_l[s_idx:e_idx]

            optimizer.zero_grad()

            loss = graphsage.loss(src_l_cut, Variable(torch.cuda.LongTensor(label_l_cut)))
            m_loss.append(loss.item())

            loss.backward()
            optimizer.step()
        print(f'Epoch mean Loss : {np.mean(m_loss)}')
        val_auc, val_f1 = eval_one_epoch(graphsage, val_src_l, val_label_l)
        print("Validation AUC:", val_auc)
        print("Validation F1:", val_f1)

        if early_stopper.early_stop_check(val_auc):
            print(f'No improvment over {early_stopper.max_round} epochs, stop training')
            best_epoch = early_stopper.best_epoch
            print(f'Loading the best model at epoch {best_epoch}')
            best_model_path = get_checkpoint_path(best_epoch)
            graphsage.load_state_dict(torch.load(best_model_path))
            print(f'Loaded the best model at epoch {best_epoch} for inference')
            os.remove(best_model_path)
            break
        else:
            if early_stopper.is_best:
                torch.save(graphsage.state_dict(), get_checkpoint_path(epoch))
                print(f'Saved SAGE-{SUBREDDIT}-{WORD_EMBEDDING}-{TRAINING_METHOD}-{early_stopper.best_epoch}.pth')
                for i in range(epoch):
                    try:
                        os.remove(get_checkpoint_path(i))
                        print(f'Deleted SAGE-{SUBREDDIT}-{WORD_EMBEDDING}-{TRAINING_METHOD}-{i}.pth')
                    except:
                        continue

    test_auc, test_f1 = eval_one_epoch(graphsage, test_src_l, test_label_l)
    print("TEST AUC:", test_auc)
    print("TEST F1:", test_f1)
    print('Saving SAGE_MEAN model')
    torch.save(graphsage.state_dict(), MODEL_SAVE_PATH)
    print('SAGE_MEAN models saved')


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
                         'tf-idf' - not supported yet

        TIME_CUT : int

        max_round : int

=====CONFIGS=====
'''

SUBREDDIT = 'iama'
TRAINING_METHOD = 'SELECTIVE'
WORD_EMBEDDING = 'bert-base-uncased'
time_cut = 309000
max_round = 10
BATCH_SIZE = 200

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_SAVE_PATH = f'./saved_models/SAGE-{SUBREDDIT}-{WORD_EMBEDDING}-{TRAINING_METHOD}.pth'
get_checkpoint_path = lambda \
    epoch: f'./saved_checkpoints/SAGE-{SUBREDDIT}-{WORD_EMBEDDING}-{TRAINING_METHOD}-{epoch}.pth'

if __name__ == "__main__":
    print(f'SUBREDDIT : {SUBREDDIT}')
    print(f'TIME_CUT :{time_cut}')
    print(f'TRAINING_METHOD : {TRAINING_METHOD}')
    print(f'WORD_EMBEDDING : {WORD_EMBEDDING}')
    run()

