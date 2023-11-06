import torch


# X 是节点的特征矩阵
# S 是图的邻接矩阵
# D是度矩阵
def gl_loss(X, S, losslr1, losslr2):
    D = compute_degree_matrix(S)
    return 2 * torch.trace(torch.matmul(torch.matmul(X.t(), (D - S)), X))


def compute_degree_matrix(S):
    # 对邻接矩阵的每一行求和，得到对应节点的度
    degrees = torch.sum(S, dim=1)
    # 构建度矩阵
    D = torch.diag(degrees)
    return D
