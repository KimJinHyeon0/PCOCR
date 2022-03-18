import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import time
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score


import dgl

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(222)

class GATLayer(nn.Module):
    def __init__(self, g, in_dim, out_dim):
        super(GATLayer, self).__init__()
        self.g = g
        # equation (1)
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        # equation (2)
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_fc.weight, gain=gain)

    def edge_attention(self, edges):
        # edge UDF for equation (2)
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
        a = self.attn_fc(z2)
        return {'e': F.leaky_relu(a)}

    def message_func(self, edges):
        # message UDF for equation (3) & (4)
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        # reduce UDF for equation (3) & (4)
        # equation (3)
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        # equation (4)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'h': h}

    def forward(self, h):
        # equation (1)
        z = self.fc(h)
        self.g.ndata['z'] = z
        # equation (2)
        self.g.apply_edges(self.edge_attention)
        # equation (3) & (4)
        self.g.update_all(self.message_func, self.reduce_func)
        return self.g.ndata.pop('h')



def edge_attention(self, edges):
    # edge UDF for equation (2)
    z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
    a = self.attn_fc(z2)
    return {'e' : F.leaky_relu(a)}


def reduce_func(self, nodes):
    # reduce UDF for equation (3) & (4)
    # equation (3)
    alpha = F.softmax(nodes.mailbox['e'], dim=1)
    # equation (4)
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
        # Be aware that the input dimension is hidden_dim*num_heads since
        # multiple head outputs are concatenated together. Also, only
        # one attention head in the output layer.
        self.layer2 = MultiHeadGATLayer(g, hidden_dim * num_heads, out_dim, 1)

        self.dropout = nn.Dropout(p=0.1, inplace=True)
        self.fc_1 = nn.Linear(out_dim, 8)
        self.fc_2 = nn.Linear(8, 1)

        self.embeds = 0
    def forward(self, h):
        h = self.layer1(h)
        h = self.dropout(h)
        h = F.elu(h)
        h = self.layer2(h)
        self.embeds = h
        h = self.dropout(h)
        h = self.fc_1(h)
        h = self.dropout(h)

        return self.fc_2(h)



def load_cora_data(DATA, time_cut, train_time, TRAINING_METHOD):
    g_df = pd.read_csv('./processed/{}_structure.csv'.format(DATA))
    n_feat = np.load('./processed/{}_node_feat.npy'.format(DATA))
    g_num = g_df.g_num.values
    g_ts = g_df.g_ts.values
    src_l = g_df.u.values
    dst_l = g_df.i.values
    ts_l = g_df.ts.values
    label_l = g_df.label.values
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
    total_label_l = np.hstack((train_label_l, test_label_l, val_label_l))

    features = torch.cuda.FloatTensor(n_feat)
    labels = np.zeros(features.shape[0])
    labels[total_src_l] = total_label_l
    labels = torch.cuda.FloatTensor(labels)
    labels = torch.unsqueeze(labels, dim=1)
    mask = np.zeros(features.shape[0])
    mask[total_src_l] = 1
    mask = torch.cuda.BoolTensor(mask)
    g = dgl.graph((total_src_l, total_dst_l), num_nodes=features.shape[0])
    return g, features, labels, mask




DATA = 'iama'
time_cut = 309000
train_time = 3888000
TRAINING_METHOD = 'SELECTIVE'
OUT_NODE_FEAT = './processed/{}_node_feat_GAT.npy'.format(DATA)

g, features, labels, mask = load_cora_data(DATA, time_cut, train_time, TRAINING_METHOD)
g = g.to(device)
# create the model, 2 heads, each head has hidden size 8
net = GAT(g,
          in_dim=features.size()[1],
          hidden_dim=768,
          out_dim=768,
          num_heads=2)
net.to(device)
# create optimizer
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
criterion = torch.nn.BCELoss()
# main loop
dur = []
for epoch in range(30):
    if epoch >= 3:
        t0 = time.time()

    optimizer.zero_grad()
    prob = net(features).sigmoid()
    loss = criterion(prob[mask], labels[mask])
    loss.backward()
    optimizer.step()

    pred_prob = prob[mask].detach().cpu().numpy()
    pred_label = np.round(pred_prob)
    true_label = torch.squeeze(labels[mask].detach()).cpu().numpy()

    acc = (true_label == pred_label).mean()
    auc_roc = roc_auc_score(true_label, pred_prob)
    F1 = f1_score(true_label, pred_label)

    if epoch >= 3:
        dur.append(time.time() - t0)

    print("Epoch {:05d} | Loss {:.4f} | ACC {:.4f} | AUC {:.4f} | F1 {:.4f} | Time(s) {:.4f}".format(
        epoch, loss.item(), acc, auc_roc, F1, np.mean(dur)))

output = net.embeds.detach().cpu()
print(output, type(output))
np.save(OUT_NODE_FEAT, output)