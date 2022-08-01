import os
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import pandas as pd
import math
import dgl
from sklearn.metrics import f1_score, roc_auc_score

from utils import EarlyStopMonitor

class GATLayer(nn.Module):
    def __init__(self, g, in_dim, out_dim):
        super(GATLayer, self).__init__()
        self.g = g
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_fc.weight, gain=gain)

    def edge_attention(self, edges):
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
        a = self.attn_fc(z2)
        return {'e': F.leaky_relu(a)}

    def message_func(self, edges):
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'h': h}

    def forward(self, h):
        z = self.fc(h)
        self.g.ndata['z'] = z
        self.g.apply_edges(self.edge_attention)
        self.g.update_all(self.message_func, self.reduce_func)
        return self.g.ndata.pop('h')

def edge_attention(self, edges):
    z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
    a = self.attn_fc(z2)
    return {'e' : F.leaky_relu(a)}

def reduce_func(self, nodes):
    alpha = F.softmax(nodes.mailbox['e'], dim=1)
    h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
    return {'h' : h}

class MultiHeadGATLayer(nn.Module):
    def __init__(self, g, in_dim, out_dim, num_heads, merge='cat'):
        super(MultiHeadGATLayer, self).__init__()
        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(GATLayer(g, in_dim, out_dim))
        self.merge = merge

    def forward(self, h):
        head_outs = [attn_head(h) for attn_head in self.heads]
        if self.merge == 'cat':
            # concat on the output feature dimension (dim=1)
            return torch.cat(head_outs, dim=1)
        else:
            # merge using average
            return torch.mean(torch.stack(head_outs))

class GAT(nn.Module):
    def __init__(self, g, in_dim, hidden_dim, out_dim, num_heads):
        super(GAT, self).__init__()
        self.layer1 = MultiHeadGATLayer(g, in_dim, hidden_dim, num_heads)
        self.layer2 = MultiHeadGATLayer(g, hidden_dim * num_heads, out_dim, 1)
        self.dropout = nn.Dropout(p=0.1, inplace=True)
        self.fc = nn.Linear(out_dim, 1)

        self.feature = 0

    def forward(self, h):
        h = self.layer1(h)
        h = F.elu(h)
        h = self.dropout(h)
        h = self.layer2(h)
        self.feature = h
        h = self.dropout(h)
        return self.fc(h)

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

    train_time = 3283200  # '2018-02-08 00:00:00' - '2018-01-01 00:00:00'
    test_time = 3888000   # '2018-02-15 00:00:00' - '2018-01-01 00:00:00'

    train_flag = (g_ts < train_time)
    valid_flag = (g_ts >= train_time) & (g_ts < test_time)
    test_flag = (g_ts >= test_time)

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
        time_cut_flag = (train_ts_l < TIME_CUT)

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

    total_src_l = np.hstack((train_src_l, test_src_l, val_src_l))
    total_dst_l = np.hstack((train_dst_l, test_dst_l, val_dst_l))

    features = torch.cuda.FloatTensor(n_feat)
    graph = dgl.graph((total_src_l, total_dst_l), num_nodes=features.shape[0]).to(device)

    labels = torch.zeros((features.shape[0], 1), dtype=torch.float, device=device)
    for index, label in zip(src_l, label_l):
        if label:
            labels[index] = label

    train_mask = torch.zeros(features.shape[0], dtype=torch.bool, device=device)
    val_mask = torch.zeros(features.shape[0], dtype=torch.bool, device=device)
    test_mask = torch.zeros(features.shape[0], dtype=torch.bool, device=device)

    train_mask[train_src_l] = True
    val_mask[val_src_l] = True
    test_mask[test_src_l] = True

    return graph, features, labels, train_mask, val_mask, test_mask

def eval_one_epoch(gat, features, labels, mask):
    f1, auc = [], []
    with torch.no_grad():
        gat.eval()
        num_instance = sum(mask).cpu()
        num_batch = math.ceil(num_instance / BATCH_SIZE)

        indices = np.arange(num_instance)
        np.random.shuffle(indices)
        for k in range(num_batch):
            s_idx = k * BATCH_SIZE
            e_idx = min(num_instance - 1, s_idx + BATCH_SIZE)

            batch_indices = indices[s_idx:e_idx]

            prob = gat(features).sigmoid()

            prob_cut = prob[mask][batch_indices].cpu().numpy()
            label_cut = labels[mask][batch_indices].cpu().numpy()

            pred_label = np.round(prob_cut)

            auc.append(roc_auc_score(label_cut, prob_cut))
            f1.append(f1_score(label_cut, pred_label))

    return np.mean(auc), np.mean(f1)

torch.manual_seed(222)
np.random.seed(222)

def run():
    NUM_EPOCH = 1000

    g, features, labels, train_mask, val_mask, test_mask = load_data()
    print('Data Loaded')
    # create the model, 2 heads, each head has hidden size 8
    gat = GAT(g,
              in_dim=features.shape[1],
              hidden_dim=features.shape[1],
              out_dim=features.shape[1],
              num_heads=2)

    gat.to(device)
    lr_criterion = nn.BCELoss()
    early_stopper = EarlyStopMonitor(max_round)
    num_instance = sum(train_mask).cpu()
    print(f'num_instance : {num_instance}')
    indices = np.arange(num_instance)
    num_batch = math.ceil(num_instance / BATCH_SIZE)
    # create optimizer
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, gat.parameters()), lr=LEARNING_RATE)

    # main loop
    gat.train()
    for epoch in range(NUM_EPOCH):
        m_loss = []
        print(f'Start {epoch} epoch')
        # random shuffle
        np.random.shuffle(indices)

        for k in range(num_batch):
            s_idx = k * BATCH_SIZE
            e_idx = min(num_instance - 1, s_idx + BATCH_SIZE)
            batch_indices = indices[s_idx:e_idx]

            optimizer.zero_grad()
            prob = gat(features).sigmoid()
            loss = lr_criterion(prob[train_mask][batch_indices], labels[train_mask][batch_indices])
            m_loss.append(loss.item())
            loss.backward()
            optimizer.step()
        print(f'Epoch mean Loss : {np.mean(m_loss)}')
        val_auc, val_f1 = eval_one_epoch(gat, features, labels, val_mask)
        print("Validation AUC:", val_auc)
        print("Validation F1:", val_f1)

        if early_stopper.early_stop_check(val_auc):
            print(f'No improvement over {early_stopper.max_round} epochs, stop training')
            best_epoch = early_stopper.best_epoch
            print(f'Loading the best model at epoch {best_epoch}')
            best_model_path = get_checkpoint_path(best_epoch)
            gat.load_state_dict(torch.load(best_model_path))
            print(f'Loaded the best model at epoch {best_epoch} for inference')
            os.remove(best_model_path)
            break
        else:
            if early_stopper.is_best:
                torch.save(gat.state_dict(), get_checkpoint_path(epoch))
                print(f'Saved {MODEL_NAME}-{early_stopper.best_epoch}.pth')
                for i in range(epoch):
                    try:
                        os.remove(get_checkpoint_path(i))
                        print(f'Deleted {MODEL_NAME}-{i}.pth')
                    except:
                        continue

    test_auc, test_f1 = eval_one_epoch(gat, features, labels, test_mask)
    print("TEST AUC:", test_auc)
    print("TEST F1:", test_f1)
    print('Saving GAT model')
    torch.save(gat.state_dict(), MODEL_SAVE_PATH)
    print('GAT models saved')


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

        BATCH_SIZE :" (200, 128, 64)

        LEARNING_RATE : (3e-4, 1e-5, 1e-4)

=====CONFIGS=====
'''

SUBREDDIT = 'iama'
TRAINING_METHOD = 'SELECTIVE'
WORD_EMBEDDING = 'bert-base-uncased'
TIME_CUT = 309000
max_round = 10
BATCH_SIZE = 200
LEARNING_RATE = 3e-4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = f'GAT-{SUBREDDIT}-{WORD_EMBEDDING}-{BATCH_SIZE}-{LEARNING_RATE}'
MODEL_SAVE_PATH = f'./saved_models/pretrained_models/{MODEL_NAME}.pth'
get_checkpoint_path = lambda \
    epoch: f'./saved_checkpoints/{MODEL_NAME}-{epoch}.pth'

if __name__ == "__main__":
    print(f'SUBREDDIT : {SUBREDDIT}')
    print(f'TIME_CUT :{TIME_CUT}')
    print(f'TRAINING_METHOD : {TRAINING_METHOD}')
    print(f'WORD_EMBEDDING : {WORD_EMBEDDING}')
    run()