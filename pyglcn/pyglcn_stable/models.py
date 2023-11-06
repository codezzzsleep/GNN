from layers import GCNConv, SparseGraphLearn
import torch.nn as nn
import torch.nn.functional as F


class GLCN(nn.Module):
    def __init__(self, in_dim, hidden_gcn, hidden_gl, out_dim, dropout, alpha=0.5):
        super(GLCN, self).__init__()
        self.gl = SparseGraphLearn(in_dim, hidden_gl, alpha=alpha)
        self.gcn1 = GCNConv(in_dim, hidden_gcn)
        self.gcn2 = GCNConv(hidden_gcn, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, edge):
        h, adj = self.gl(self.dropout(inputs), edge)
        x = self.gcn1(self.dropout(inputs), adj)
        x = F.relu(x)
        x = self.gcn2(self.dropout(x), adj)
        return h, adj, x


class GCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout):
        super(GCN, self).__init__()
        self.gcn1 = GCNConv(in_dim, hidden_dim, bias=False)
        self.gcn2 = GCNConv(hidden_dim, out_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, adj):
        x = self.gcn1(self.dropout(inputs), adj)
        x = F.relu(x)
        x = self.gcn2(self.dropout(x), adj)
        return x
