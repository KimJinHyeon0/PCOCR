import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv, SAGEConv, GATConv


class GCN(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(GCN, self).__init__()
        self.layer_1 = GraphConv(in_feats, h_feats)
        self.fc = nn.Linear(h_feats, num_classes)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, block, in_feat):
        h = self.layer_1(block, in_feat)
        h = F.relu(h)
        h = self.fc(h)

        return self.sigmoid(h)

    def out(self, block, in_feat):
        return self.layer_1(block, in_feat)


class SAGE(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes, agg='mean'):
        super(SAGE, self).__init__()
        self.layer_1 = SAGEConv(in_feats, h_feats, agg)
        self.fc = nn.Linear(h_feats, num_classes)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, block, in_feat):
        h = self.layer_1(block, in_feat)
        h = F.relu(h)
        h = self.fc(h)

        return self.sigmoid(h)

    def out(self, block, in_feat):
        return self.layer_1(block, in_feat)


class GAT(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes, num_heads=1):
        super(GAT, self).__init__()
        self.layer_1 = GATConv(in_feats, h_feats, num_heads)
        self.fc = nn.Linear(h_feats, num_classes)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, block, in_feat):
        h = self.layer_1(block, in_feat).squeeze(dim=1)
        h = F.relu(h)
        h = self.fc(h)

        return self.sigmoid(h)

    def out(self, block, in_feat):
        return self.layer_1(block, in_feat).squeeze(dim=1)

