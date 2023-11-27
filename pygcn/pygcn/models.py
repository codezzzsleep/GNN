from layers import GCNConv
import torch.nn as nn
import torch.nn.functional as F


class GCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout):
        super(GCN, self).__init__()
        self.gcn1 = GCNConv(in_dim, hidden_dim, bias=True)
        self.gcn2 = GCNConv(in_dim, hidden_dim, bias=True)
        self.gcn3 = GCNConv(in_dim, hidden_dim, bias=True)
        self.gcn4 = GCNConv(in_dim, hidden_dim, bias=True)
        self.gcn5 = GCNConv(in_dim, hidden_dim, bias=True)
        self.gcn6 = GCNConv(in_dim, hidden_dim, bias=True)
        self.gcn7 = GCNConv(in_dim, hidden_dim, bias=True)
        self.gcn8 = GCNConv(hidden_dim, out_dim, bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, adj):
        x1 = self.gcn1(self.dropout(inputs), adj)
        x2 = self.gcn2(self.dropout(inputs), adj)
        x3 = self.gcn3(self.dropout(inputs), adj)
        x4 = self.gcn4(self.dropout(inputs), adj)
        x5 = self.gcn5(self.dropout(inputs), adj)
        x6 = self.gcn6(self.dropout(inputs), adj)
        x7 = self.gcn7(self.dropout(inputs), adj)
        x = x1 + x2 + x3 + x4 + x5 + x6 + x7
        x = F.relu(x)
        x = self.gcn8(self.dropout(x), adj)
        return x
