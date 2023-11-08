import numpy as np
import torch


def get_similarity(x):
    x_dis = x @ x.T
    mask = torch.eye(x_dis.shape[0])
    x_sum = torch.sum(x ** 2, 1).reshape(-1, 1)
    x_sum = torch.sqrt(x_sum).reshape(-1, 1)
    x_sum = x_sum @ x_sum.T
    x_dis = x_dis * (x_sum ** (-1))
    x_dis = (1 - mask) * x_dis
    return x_dis


if __name__ == "__main__":
    a = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
    a = torch.from_numpy(a)
    print(a)
    print(get_similarity(a))
