import torch
from utils import load_data

adj, edge, features, labels, idx_train, idx_val, idx_test = load_data("cora")


# X 是节点的特征矩阵
# S 是图的邻接矩阵
# D是度矩阵
def gl_loss(X, S, gamma=0, bate=0):
    D = compute_degree_matrix(S)
    loss = torch.trace(torch.matmul(torch.matmul(X.t(), (D - S)), X))
    loss1 = torch.norm(S, p=2)
    loss = loss1 + loss
    return loss


def compute_degree_matrix(S):
    # 对邻接矩阵的每一行求和，得到对应节点的度
    degrees = torch.sum(S, dim=1)
    # 构建度矩阵
    D = torch.diag(degrees)
    return D


def get_similarity(x):
    x_dis = x @ x.T
    mask = torch.eye(x_dis.shape[0])
    x_sum = torch.sum(x ** 2, 1).reshape(-1, 1)
    x_sum = torch.sqrt(x_sum).reshape(-1, 1)
    x_sum = x_sum @ x_sum.T
    x_dis = x_dis * (x_sum ** (-1))
    x_dis = (1 - mask) * x_dis
    return x_dis


def Ncontrast(x, adj_label, tau=1):
    x_dis = get_similarity(x)
    x_dis = torch.exp(tau * x_dis)
    x_dis_sum = torch.sum(adj_label, 1)
    x_dis_sum_pos = torch.sum(x_dis * adj_label, 1)
    loss = -torch.log(x_dis_sum_pos * (x_dis_sum ** (-1)) + 1e-8).mean()
    return loss
