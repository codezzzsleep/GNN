import numpy as np
import torch
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader


def get_cora():
    dataset = Planetoid(root='/data/cora', name='Cora', transform=T.NormalizeFeatures())
    data = dataset[0]
    x = data.x
    edge = data.edge_index
    num_features = dataset.num_features
    num_classes = dataset.num_classes
    return x, edge, num_features, num_classes, data


def get_citeseer():
    dataset = Planetoid(root='/data/citeseer', name='Citeseer', transform=T.NormalizeFeatures())
    data = dataset[0]
    x = data.x
    edge = data.edge_index
    num_features = dataset.num_features
    num_classes = dataset.num_classes
    return x, edge, num_features, num_classes, data


def get_pubmed():
    dataset = Planetoid(root='/data/pubmed', name='PubMed', transform=T.NormalizeFeatures())
    data = dataset[0]
    x = data.x
    edge = data.edge_index
    num_features = dataset.num_features
    num_classes = dataset.num_classes
    return x, edge, num_features, num_classes, data


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
