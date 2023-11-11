import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import SparseGraphLearn
from torch_geometric.nn import GCNConv, GATConv


class SGLCN(nn.Module):
    def __init__(self, nfeat, hidd_gl, nhid, nclass, dropout):
        super(SGLCN, self).__init__()
        self.gl = SparseGraphLearn(nfeat, hidd_gl)
        self.gc1 = GCNConv(nfeat, nhid)
        self.gc2 = GCNConv(nhid, nclass)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, edge_index):
        h, edge_weight = self.gl(self.dropout(inputs), edge_index)
        x = F.relu(self.gc1(self.dropout(inputs), edge_index, edge_weight))
        N = inputs.shape[0]
        x = self.gc2(self.dropout(x), edge_index=edge_index, edge_weight=edge_weight)
        s = torch.sparse_coo_tensor(edge_index, edge_weight, torch.Size([N, N]))
        s = self.dropout(s.to_dense())
        return h, s, x


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()
        self.gc1 = GCNConv(nfeat, nhid)
        self.gc2 = GCNConv(nhid, nclass)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, edge_index):
        x = self.gc1(self.dropout(inputs), edge_index)
        x = F.relu(x)
        x = self.gc2(x, edge_index)
        return x


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, nheads, dropout):
        super(GAT, self).__init__()
        self.gat1 = GATConv(nfeat, nhid, nheads, dropout)

        self.gat2 = GATConv(nhid * nheads, nclass, dropout=dropout, concat=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, edge_index):
        x = self.dropout(inputs)
        x = self.gat1(x, edge_index)
        x = self.dropout(x)
        x = self.gat2(x, edge_index)
        x = F.elu(x)
        return x
