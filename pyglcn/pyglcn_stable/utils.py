import numpy as np
import scipy.sparse as sp
import torch
import scipy.io as sio


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool_)


def load_data(dataset_str, path="../data/"):
    path = path + dataset_str + "/"
    if dataset_str == "cora":
        features = sio.loadmat(path + "feature")
        features = features['matrix']
        adj = sio.loadmat(path + "adj")
        adj = adj['matrix']
        labels = sio.loadmat(path + "label")
        labels = labels['matrix']
        idx_train = range(140)
        idx_val = range(200, 500)
        idx_test = range(500, 1500)
    elif dataset_str == "citeseer":
        features = sio.loadmat(path + "feature")
        features = features['matrix']
        adj = sio.loadmat(path + "adj")
        adj = adj['matrix']
        labels = sio.loadmat(path + "label")
        labels = labels['matrix']
        idx_test = sio.loadmat(path + "test.mat")
        idx_test = idx_test['array'].flatten()
        idx_train = range(120)
        idx_val = range(120, 620)
    else:
        features = sio.loadmat(path + "feature")
        features = features['matrix']
        adj = sio.loadmat(path + "adj")
        adj = adj['matrix']
        labels = sio.loadmat(path + "label")
        labels = labels['matrix']
        idx_test = sio.loadmat(path + "test.mat")
        idx_test = idx_test['matrix']
        idx_train = range(60)
        idx_val = range(200, 500)

    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    features = tuple_to_sparse(preprocess_features(features)).to_dense()
    adj, edge = preprocess_adj(adj)
    adj = tuple_to_sparse(adj)
    edge = torch.LongTensor(edge)
    labels = torch.LongTensor(np.where(labels)[1])
    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, edge, features, labels, idx_train, idx_val, idx_test


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""

    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def tuple_to_sparse(tuple_data):
    coords, values, shape = tuple_data
    indices = torch.LongTensor(coords.T)
    values = torch.FloatTensor(values)
    shape = torch.Size(shape)
    sparse_tensor = torch.sparse_coo_tensor(indices, values, shape)
    return sparse_tensor


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return sparse_to_tuple(features)


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    edge = np.array(np.nonzero(adj_normalized.todense()))
    return sparse_to_tuple(adj_normalized), edge


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


def construct_feed_dict(features, adj, labels, labels_mask, epoch, placeholders):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['labels_mask']: labels_mask})
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['adj']: adj})
    feed_dict.update({placeholders['step']: epoch})
    feed_dict.update({placeholders['num_nodes']: features[2][0]})
    feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})
    return feed_dict


if __name__ == "__main__":
    adj, edge, features, labels, idx_train, idx_val, idx_test = load_data("cora")
    print(edge.dtype)
    print(type(edge))
    print(edge.shape)
