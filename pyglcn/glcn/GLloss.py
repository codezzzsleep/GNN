import torch


def gl_loss(x, edge_index, edge_weight, losslr1, losslr2):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = x.to(device)
    N = x.shape[0]
    adj = torch.sparse_coo_tensor(edge_index, edge_weight, torch.Size((N, N))).to_dense()
    adj = adj.to(device)

    D = torch.diag(torch.ones(x.shape[0])) * -1  # 计算度矩阵
    D = D.to(device)

    D = (D + adj) * -1
    D = torch.mm(x.t(), D)

    loss1 = torch.trace(torch.mm(D, x)) * losslr1
    loss2 = torch.trace(torch.mm(adj.t(), adj)) * losslr2

    return loss1 - loss2
