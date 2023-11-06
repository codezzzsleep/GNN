import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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

    def forward(self, input, adj):
        suport = torch.mm(input, self.weight)
        output = torch.mm(adj, suport)
        if self.bias is not None:
            output = output + self.bias
        return output


class SparseGraphLearn(nn.Module):
    def __init__(self, input_dim, output_dim, alpha=0.5, bias=False):
        super(SparseGraphLearn, self).__init__()
        self.weights = nn.Parameter(torch.Tensor(input_dim, output_dim))
        self.a = nn.Parameter(torch.Tensor(output_dim, 1))
        self.alpha = alpha
        if bias:
            self.bias = nn.Parameter(torch.Tensor(output_dim))
        else:
            self.bias = None

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weights)
        nn.init.xavier_uniform_(self.a)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, inputs, edge):
        h = torch.matmul(inputs, self.weights)

        edge_v = torch.abs(h[edge[0]] - h[edge[1]])
        edge_v = torch.squeeze(F.relu(torch.matmul(edge_v, self.a)))
        N = inputs.size(0)
        sgraph = torch.sparse_coo_tensor(edge, edge_v, torch.Size([N, N]))
        sgraph = F.softmax(sgraph.to_dense(), dim=-1)

        # 使用残差神经网络
        _v = torch.ones(edge_v.shape).to(device)
        edge = torch.sparse_coo_tensor(edge, _v, torch.Size([N, N])).to(device)
        if torch.allclose(sgraph, torch.zeros_like(sgraph)):
            print("GL矩阵全为0")
        else:
            print("GL矩阵不全为0")
        sgraph = sgraph + edge.to_dense() * self.alpha

        return h, sgraph
