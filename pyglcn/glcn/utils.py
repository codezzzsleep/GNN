import torch_geometric.datasets as geo_datasets
import torch
import numpy as np


def load_data():
    cora_dataset = geo_datasets.Planetoid(root='../data', name='Cora')
    cora_data = cora_dataset[0]
    x = cora_data.x
    edge_index = cora_data.edge_index
    labels = cora_data.y
    train_mask = cora_data.train_mask
    val_mask = cora_data.val_mask
    test_mask = cora_data.test_mask
    return x, edge_index, labels, train_mask, val_mask, test_mask


def load_dataset():
    cora_dataset = geo_datasets.Planetoid(root='../data', name='Cora')

    return cora_dataset


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def same_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    a = load_data()
    print(a)
