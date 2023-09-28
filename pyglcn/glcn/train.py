from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from utils import load_data, accuracy
from models import SGLCN, GCN
from GLloss import gl_loss
import torch.nn as nn

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=600,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=30,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.6,
                    help='Dropout rate (1 - keep probability).')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

    # Load data
features, edge_index, labels, idx_train, idx_val, idx_test = load_data()
loss_fn = nn.CrossEntropyLoss()
# Model and optimizer
# model = GCN(nfeat=features.shape[1],
#             # hidd_gl=70,
#             nhid=args.hidden,
#             nclass=labels.max().item() + 1,
#             dropout=args.dropout)
model = SGLCN(nfeat=features.shape[1],
              hidd_gl=70,
              nhid=args.hidden,
              nclass=labels.max().item() + 1,
              dropout=args.dropout)
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)
# params = list(model.parameters())
#
# # 打印参数信息
# for i, param in enumerate(params):
#     print(f"Parameter {i}: {param.shape}")
if args.cuda:
    model.cuda()
    features = features.cuda()
    edge_index = edge_index.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()


def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    h, edge_weight, output = model(features, edge_index)
    loss_train = loss_fn(output[idx_train], labels[idx_train])
    loss2 = gl_loss(x=h, edge_index=edge_index, edge_weight=edge_weight, losslr1=0.01, losslr2=1e-4)
    loss_train = loss2 + loss_train
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    model.eval()
    h, edge_weight, output = model(features, edge_index)

    loss_val = loss_fn(output[idx_val], labels[idx_val])
    loss2 = gl_loss(x=h, edge_index=edge_index, edge_weight=edge_weight, losslr1=0.01, losslr2=1e-4)
    loss_val = loss_val + loss2
    acc_val = accuracy(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch + 1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))
    test()

def test():
    model.eval()
    h, edge_weight, output = model(features, edge_index)
    loss_test = loss_fn(output[idx_test], labels[idx_test])
    loss2 = gl_loss(x=h, edge_index=edge_index, edge_weight=edge_weight, losslr1=0.01, losslr2=1e-4)
    loss_test = loss2 + loss_test
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))


# Train model
t_total = time.time()
for epoch in range(args.epochs):
    train(epoch)
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Testing
test()