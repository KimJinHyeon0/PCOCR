import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
import torch.nn.functional as F
import pandas as pd
import numpy as np
import time
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
    """
    Encodes a node's using 'convolutional' GraphSage approach
    """

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

        self.weight = nn.Parameter(torch.cuda.FloatTensor(num_classes, enc.embed_dim))
        init.xavier_uniform_(self.weight)

    def forward(self, nodes):
        embeds = self.enc(nodes)
        scores = self.weight.mm(embeds)

        return scores.t()

    def loss(self, nodes, labels):
        scores = self.forward(nodes)

        return self.xent(scores, labels.squeeze()).to(device)


def load_data(DATA, time_cut, train_time, TRAINING_METHOD):
    print('Loading Data')
    g_df = pd.read_csv('./processed/{}_structure.csv'.format(DATA))
    n_feat = np.load('./processed/{}_node_feat.npy'.format(DATA))
    g_ts = g_df.g_ts.values
    ts_l = g_df.ts.values

    if TRAINING_METHOD == 'SELECTIVE':
        train_flag = (g_ts < train_time)
        time_cut_flag = (ts_l < time_cut)

        train_df = g_df[train_flag & time_cut_flag]
        temp_df = g_df[~train_flag]

        train_g_num = train_df.g_num.values
        train_label_l = train_df.label.values
        train_index_l = np.arange(len(train_df))

        selective_flag = np.zeros((len(train_df)), dtype=bool)

        for k in set(train_g_num):
            temp_flag = (train_g_num == k)

            if sum(temp_flag) < 2:
                continue

            temp_index = train_index_l[temp_flag]
            temp_label = train_label_l[temp_flag]

            if temp_label.mean() >= 0.5:
                major, minor = temp_index[temp_label == 1], temp_index[temp_label == 0]

            else:
                major, minor = temp_index[temp_label == 0], temp_index[temp_label == 1]

            balanced = np.random.choice(major, len(minor), replace=False)
            selective_flag[balanced] = True
            selective_flag[minor] = True

        train_df = train_df[selective_flag]
        g_df = pd.concat([train_df, temp_df])

    g_num = g_df.g_num.values
    g_ts = g_df.g_ts.values
    src = g_df.u.values
    dst = g_df.i.values
    label_l = g_df.label.values

    test_time = np.quantile(g_ts[(g_ts > train_time)], 0.5)
    train_flag = (g_ts < train_time)
    test_flag = (g_ts >= train_time) & (g_ts < test_time)
    val_flag = (g_ts >= test_time)

    total_node_set = np.sort(np.unique(np.concatenate((src, dst))))
    train_node_set = np.unique(np.concatenate((src[train_flag], dst[train_flag])))
    test_node_set = np.unique(np.concatenate((src[test_flag], dst[test_flag])))
    val_node_set = np.unique(np.concatenate((src[val_flag], dst[val_flag])))

    test_idx = np.random.choice(total_node_set, 1)[0]
    before_feat = n_feat[test_idx]

    feat_data = n_feat[total_node_set]

    assert feat_data.shape[0] == total_node_set.shape[0]
    assert len(total_node_set) == len(train_node_set) + len(test_node_set) + len(val_node_set)

    labels = np.empty((total_node_set.shape[0], 1), dtype=np.int64)
    node_map = {}

    for i, node in enumerate(total_node_set):
        node_map[node] = i
        if node in g_num:
            labels[i] = 1
        elif node in src:
            labels[i] = label_l[(g_df.idx.values == node)]
        else:
            labels[i] = 0

    adj_lists = defaultdict(set)
    for s, d in zip(src, dst):
        n_1 = node_map[s]
        n_2 = node_map[d]
        adj_lists[n_1].add(n_2)
        adj_lists[n_2].add(n_1)

    test_2_idx = node_map[test_idx]
    after_feat = feat_data[test_2_idx]

    train = np.array(list(map(lambda x: node_map[x], train_node_set)))
    test = np.array(list(map(lambda x: node_map[x], test_node_set)))
    val = np.array(list(map(lambda x: node_map[x], val_node_set)))

    assert np.array_equal(before_feat, after_feat)
    assert feat_data.shape[0] == len(total_node_set)
    assert np.array_equal(np.sort(np.hstack((train, test, val))), np.array(list(map(lambda x: node_map[x], total_node_set))))

    return feat_data, labels, adj_lists, train, test, val, node_map, g_df

def eval_one_epoch(graphsage, data, labels):
    val_acc, val_ap, val_f1, val_auc = [], [], [], []
    with torch.no_grad():
        graphsage = graphsage.eval()
        num_instance = len(data)
        num_batch = math.ceil(num_instance / BATCH_SIZE)

        for k in range(num_batch):
            s_idx = k * BATCH_SIZE
            e_idx = min(num_instance - 1, s_idx + BATCH_SIZE)
            batch_nodes = data[s_idx:e_idx]
            batch_labels = labels[batch_nodes]

            batch_output = graphsage.forward(batch_nodes)
            pred_score = batch_output.data.cpu().numpy()

            val_auc.append(roc_auc_score(batch_labels, pred_score[np.arange(batch_labels.shape[0]), batch_labels.astype(np.int).squeeze()]))
            val_f1.append(f1_score(batch_labels, pred_score.argmax(axis=1), average="micro"))

    return np.mean(val_auc), np.mean(val_f1)


def run():
    NUM_EPOCH = 1000
    max_round = 10
    np.random.seed(1)
    random.seed(1)
    feat_data, labels, adj_lists, train, test, val, node_map, g_df = load_data(DATA, time_cut, train_time, TRAINING_METHOD)
    print('Data Loaded')
    features = nn.Embedding(feat_data.shape[0], feat_data.shape[1])
    features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=False)
    features.cuda()

    agg1 = MeanAggregator(features, cuda=True)
    enc1 = Encoder(features, 768, 768, adj_lists, agg1, gcn=False, cuda=True)
    agg2 = MeanAggregator(lambda nodes: enc1(nodes).t(), cuda=True)
    enc2 = Encoder(lambda nodes: enc1(nodes).t(), enc1.embed_dim, 768, adj_lists, agg2,
                   base_model=enc1, gcn=False, cuda=True)
    enc1.num_samples = 5
    enc2.num_samples = 5

    graphsage = SupervisedGraphSage(2, enc2)
    graphsage.to(device)
    early_stopper = EarlyStopMonitor(max_round)

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, graphsage.parameters()), lr=0.0001)
    times = []

    num_instance = len(train)
    print('num_instance :', num_instance)
    num_batch = math.ceil(num_instance / BATCH_SIZE)

    for epoch in range(NUM_EPOCH):
        np.random.shuffle(train)
        print('Start {} epoch'.format(epoch))
        for k in range(num_batch):
            s_idx = k * BATCH_SIZE
            e_idx = min(num_instance - 1, s_idx + BATCH_SIZE)
            batch_nodes = train[s_idx:e_idx]
            start_time = time.time()
            optimizer.zero_grad()
            loss = graphsage.loss(batch_nodes,
                                  Variable(torch.cuda.LongTensor(labels[batch_nodes])))
            loss.backward()
            optimizer.step()
            end_time = time.time()
            times.append(end_time - start_time)
            # print(batch_nodes, batch_nodes.shape)

        val_auc, val_f1 = eval_one_epoch(graphsage, val, labels)
        print("Validation AUC:", val_auc)
        print("Validation F1:", val_f1)
        print("Average batch time:", np.mean(times))

        if early_stopper.early_stop_check(val_auc):
            print('No improvment over {} epochs, stop training'.format(early_stopper.max_round))
            best_epoch = early_stopper.best_epoch
            print(f'Loading the best model at epoch {best_epoch}')
            best_model_path = get_checkpoint_path(best_epoch)
            graphsage.load_state_dict(torch.load(best_model_path))
            print(f'Loaded the best model at epoch {best_epoch} for inference')
            graphsage.eval()
            os.remove(best_model_path)
            print('Deleted {}-SAGE_MEAN-{}.pth'.format(MODEL_NUM, best_epoch))
            break
        else:
            if early_stopper.is_best:
                torch.save(graphsage.state_dict(), get_checkpoint_path(epoch))
                print('Saved {}-SAGE_MEAN-{}.pth'.format(MODEL_NUM, early_stopper.best_epoch))
                for i in range(epoch):
                    try:
                        os.remove(get_checkpoint_path(i))
                        print('Deleted {}-SAGE_MEAN-{}.pth'.format(MODEL_NUM, i))
                    except:
                        continue

    test_f1, test_auc = eval_one_epoch(graphsage, test, labels)
    print("TEST AUC:", test_auc)
    print("TEST F1:", test_f1)
    print('Saving SAGE_MEAN model')
    torch.save(graphsage.state_dict(), MODEL_SAVE_PATH)
    print('SAGE_MEAN models saved')

DATA = 'iama'
train_time = 3888000
time_cut = 309000
TRAINING_METHOD = 'SELECTIVE'
MODEL_NUM = '000'
BATCH_SIZE = 200
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
get_checkpoint_path = lambda \
    epoch: f'./saved_checkpoints/{MODEL_NUM}-SAGE_MEAN-{epoch}.pth'
MODEL_SAVE_PATH = f'./saved_models/{MODEL_NUM}-SAGE_MEAN.pth'
if __name__ == "__main__":
    print('DATA :', DATA)
    print('time_cut :', time_cut)
    print('Training Method :', TRAINING_METHOD)
    print('MODEL_NUM : ', MODEL_NUM)
    run()

