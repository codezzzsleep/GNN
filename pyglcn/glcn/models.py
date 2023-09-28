import torch.nn as nn
import torch.nn.functional as F
from layers import SparseGraphLearn
from torch_geometric.nn import GCNConv


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

        x = self.gc2(self.dropout(x), edge_index, edge_weight)

        return h, edge_weight, x


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()
        self.gc1 = GCNConv(nfeat, nhid)
        self.gc2 = GCNConv(nhid, nclass)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, edge_index):
        x = F.relu(self.gc1(self.dropout(inputs), edge_index))
        x = self.gc2(x, edge_index)
        return x
