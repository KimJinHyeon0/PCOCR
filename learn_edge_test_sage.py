import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
import torch.nn.functional as F
import pandas as pd
import numpy as np
import time
import random
from sklearn.metrics import f1_score, roc_auc_score
from collections import defaultdict

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
            embed_matrix = self.features(torch.LongTensor(unique_nodes_list).cuda())
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
                 base_model=None, gcn=False, cuda=False,
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
        return combined


class SupervisedGraphSage(nn.Module):

    def __init__(self, num_classes, enc):
        super(SupervisedGraphSage, self).__init__()
        self.enc = enc
        self.xent = nn.CrossEntropyLoss()

        self.weight = nn.Parameter(torch.FloatTensor(num_classes, enc.embed_dim).cuda())
        init.xavier_uniform_(self.weight)

    def forward(self, nodes):
        embeds = self.enc(nodes)
        scores = self.weight.mm(embeds)
        #print(scores, scores.shape)
        return scores.t()

    def loss(self, nodes, labels):
        scores = self.forward(nodes)
        # print(scores, scores.shape)
        return self.xent(scores, labels.squeeze())


def load_cora():
    DATA = 'iama'
    g_df = pd.read_csv('./processed/{}_structure.csv'.format(DATA))
    n_feat = np.load('./processed/{}_node_feat.npy'.format(DATA))
    g_num = g_df.g_num.values
    g_ts = g_df.g_ts.values
    src_l = g_df.u.values
    dst_l = g_df.i.values
    ts_l = g_df.ts.values
    label_l = g_df.label.values
    e_idx_l = g_df.idx.values

    train_time = 3888000
    test_time = np.quantile(g_df[g_df['g_ts'] > train_time].g_ts, 0.5)

    train_flag = (g_ts < train_time)
    test_flag = (g_ts >= train_time) & (g_ts < test_time)
    val_flag = (g_ts > test_time)

    train = g_df[train_flag].index.values
    test = g_df[test_flag].index.values
    val = g_df[val_flag].index.values

    feat_data = n_feat
    labels = label_l

    adj_lists = defaultdict(set)
    for src, dst, eidx, ts in zip(src_l, dst_l, e_idx_l, ts_l):
        adj_lists[src].add(dst)
        adj_lists[dst].add(src)

    total_node_set = set(np.unique(np.hstack([g_df.u.values, g_df.i.values])))
    num_nodes = len(total_node_set)

    return feat_data, labels, adj_lists, num_nodes, train, test, val, src_l


def run():
    NUM_EPOCH = 30
    np.random.seed(1)
    random.seed(1)
    feat_data, labels, adj_lists, num_nodes, train, test, val, src_l = load_cora()
    features = nn.Embedding(num_nodes, 768)
    features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=True)
    features.cuda()

    agg1 = MeanAggregator(features, cuda=True)
    enc1 = Encoder(features, 768, 128, adj_lists, agg1, gcn=False, cuda=True)
    agg2 = MeanAggregator(lambda nodes: enc1(nodes).t(), cuda=True)
    enc2 = Encoder(lambda nodes: enc1(nodes).t(), enc1.embed_dim, 128, adj_lists, agg2,
                   base_model=enc1, gcn=False, cuda=True)
    enc1.num_samples = 5
    enc2.num_samples = 5

    graphsage = SupervisedGraphSage(2, enc2)
    graphsage.cuda()
    # rand_indices = np.random.permutation(num_nodes)


    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, graphsage.parameters()), lr=0.07)
    times = []
    batch_num = len(train)//256
    for epoch in range(NUM_EPOCH):
        random.shuffle(train)
        print('Start {} epoch'.format(epoch))
        for batch in range(1, batch_num):

            batch_indices = train[(batch-1)*256:batch*256]
            batch_nodes = src_l[batch_indices]
            start_time = time.time()
            optimizer.zero_grad()
            loss = graphsage.loss(batch_nodes,
                                  Variable(torch.cuda.LongTensor(labels[batch_indices])))
            loss.backward()
            optimizer.step()
            end_time = time.time()
            times.append(end_time - start_time)
            # print(batch_nodes, batch_nodes.shape)

        val_output = graphsage.forward(src_l[val])
        print("Validation AUC:", roc_auc_score(labels[val], val_output.data.cpu().numpy().argmax(axis=1), average="micro"))
        print("Average batch time:", np.mean(times))
    torch.save(features.weight.data, './processed/{}_n_feat_SAGE.pt'.format(DATA))
    #torch.save(graphsage.state_dict(), './saved_models/GraphSage')

DATA = 'iama'
if __name__ == "__main__":
    run()
