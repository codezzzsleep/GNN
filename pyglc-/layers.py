import math

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F


class SparseGraphLearn(Module):
    def __init__(self, in_features, out_features, bias=True):
        super(SparseGraphLearn, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.a = Parameter(torch.FloatTensor(out_features, 1))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.bias = None
        self.reset_parameter()

    def reset_parameter(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.xavier_uniform_(self.a)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, inputs, edge):
        # inputs [2708, 1433]
        # self.weight [1433,70]
        h = torch.mm(inputs, self.weight)
        edge_weight = torch.abs(h[edge[0]] - h[edge[1]])
        # edge_weight [10556, 70]
        # self.a [70, 1]
        # --> edge_weight [10556, 1]
        edge_weight = torch.mm(edge_weight, self.a)
        edge_weight = F.relu(edge_weight)
        edge_weight = torch.squeeze(edge_weight)
        return h, edge_weight
