import torch
import torch.nn as nn
import torch.nn.functional as F


class GCNConv(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True):
        super(GCNConv, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_dim, out_dim))
        nn.init.xavier_uniform_(self.weight)
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_dim))
            nn.init.zeros_(self.bias)
        else:
            self.bias = None

    def forward(self, inputs, adj):
        support = torch.mm(inputs, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            output = output + self.bias
        return output
