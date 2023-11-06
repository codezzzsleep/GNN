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
    def __init__(self, input_dim, output_dim, bias=False):
        super(SparseGraphLearn, self).__init__()
        self.weights = nn.Parameter(torch.Tensor(input_dim, output_dim))
        self.a = nn.Parameter(torch.Tensor(output_dim, 1))
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

    # def __call__(self, inputs):
    #        x = inputs
    #        # dropout
    #        if self.sparse_inputs:
    #            x = sparse_dropout(x, 1-self.dropout, self.num_features_nonzero)
    #        else:
    #            x = tf.nn.dropout(x, 1-self.dropout)
    #
    #        # graph learning
    #        h = dot(x, self.vars['weights'], sparse=self.sparse_inputs)
    #        N = self.num_nodes
    #        edge_v = tf.abs(tf.gather(h,self.edge[0]) - tf.gather(h,self.edge[1]))
    #        edge_v = tf.squeeze(self.act(dot(edge_v, self.vars['a'])))
    #        sgraph = tf.SparseTensor(indices=tf.transpose(self.edge), values=edge_v, dense_shape=[N, N])
    #        sgraph = tf.sparse_softmax(sgraph)
    #        return h, sgraph
    def forward(self, inputs, edge):
        h = torch.matmul(inputs, self.weights).to(device)

        edge_v = torch.abs(h[edge[0]] - h[edge[1]]).to(device)
        edge_v = torch.squeeze(F.relu(torch.matmul(edge_v, self.a))).to(device)
        N = inputs.size(0)
        sgraph = torch.sparse_coo_tensor(edge, edge_v, torch.Size([N, N])).to(device)
        sgraph = F.softmax(sgraph.to_dense(), dim=-1).to(device)

        # 使用残差神经网络
        _v = torch.ones(edge_v.shape).to(device)
        edge = torch.sparse_coo_tensor(edge, _v, torch.Size([N, N])).to(device)
        if torch.allclose(sgraph, torch.zeros_like(sgraph)):
            print("GL矩阵全为0")
        else:
            print("GL矩阵不全为0")
        sgraph = sgraph + edge.to_dense()

        return h, sgraph
