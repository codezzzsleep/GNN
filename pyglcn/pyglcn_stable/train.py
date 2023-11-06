import time
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

from utils import load_data, accuracy, same_seed
from models import GLCN, GCN
import matplotlib.pyplot as plt
from GLoss import gl_loss
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("../logs/run2")
loss_list = []
test_acc_list = []
# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=3407, help='Random seed.')
parser.add_argument('--epochs', type=int, default=1000,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--alpha', type=float, default=0.3,
                    help='residual ratio.')
parser.add_argument('--weight_decay', type=float, default=1e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.6,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--losslr1', type=float, default=0.1,
                    help='')
parser.add_argument('--losslr2', type=float, default=0.01,
                    help='')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

same_seed(args.seed)

# Load data
adj, edge, features, labels, idx_train, idx_val, idx_test = load_data("cora")

# Model and optimizer

model = GLCN(in_dim=features.shape[1],
             hidden_gl=70,
             hidden_gcn=30,
             out_dim=labels.max().item() + 1,
             dropout=args.dropout)
# print(model.gl.weights)
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)
scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
model_ = GCN(in_dim=features.shape[1],
             hidden_dim=30,
             out_dim=labels.max().item() + 1,
             dropout=args.dropout)
optimizer_ = optim.Adam(model_.parameters(),
                        lr=args.lr, weight_decay=args.weight_decay)
loss_fn = nn.CrossEntropyLoss()
if args.cuda:
    model.cuda()
    model_.cuda()
    features = features.cuda()
    adj = adj.cuda()
    edge = edge.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()


def train(epoch):
    t = time.time()
    model.train()
    model_.train()
    optimizer.zero_grad()
    optimizer_.zero_grad()
    x, adj_new, output = model(features, edge)
    output_ = model_(features, adj)
    L2_loss1 = torch.norm(model.gl.weights, p=2)
    L2_loss2 = torch.norm(model.gcn1.weight, p=2)
    loss_train = loss_fn(output[idx_train], labels[idx_train])
    loss_train_ = loss_fn(output_[idx_train], labels[idx_train])

    writer.add_scalars(main_tag="CrossEntropyLoss",
                       tag_scalar_dict={
                           "glcn": loss_train,
                           "gcn": loss_train_
                       },
                       global_step=epoch + 1)

    gl_loss_train = gl_loss(x, adj_new, args.losslr1,
                            args.losslr2) + + L2_loss1 * args.weight_decay + L2_loss2 * args.weight_decay
    writer.add_scalar("GLoss", gl_loss_train, epoch + 1)
    loss_train = gl_loss_train + loss_train
    writer.add_scalars(main_tag="TotalLoss",
                       tag_scalar_dict={
                           "glcn": loss_train,
                           "gcn": loss_train_
                       },
                       global_step=epoch + 1)
    loss_list.append(loss_train.item())
    acc_train = accuracy(output[idx_train], labels[idx_train])
    acc_train_ = accuracy(output_[idx_train], labels[idx_train])
    writer.add_scalars(main_tag="Train-Acc",
                       tag_scalar_dict={
                           "glcn": acc_train,
                           "gcn": acc_train_
                       },
                       global_step=epoch + 1)
    loss_train.backward()
    loss_train_.backward()
    optimizer.step()
    optimizer_.step()

    model.eval()
    x, adj_new, output = model(features, edge)
    output_ = model_(features, adj)

    loss_val = loss_fn(output[idx_val], labels[idx_val])
    loss_val_ = loss_fn(output_[idx_val], labels[idx_val])

    gl_loss_val = gl_loss(x, adj_new, args.losslr1, args.losslr2)
    loss_val += gl_loss_val
    acc_val = accuracy(output[idx_val], labels[idx_val])
    acc_val_ = accuracy(output_[idx_val], labels[idx_val])

    writer.add_scalars(main_tag="Val-Acc",
                       tag_scalar_dict={
                           "glcn": acc_val,
                           "gcn": acc_val_
                       },
                       global_step=epoch + 1)
    print('Epoch: {:04d}'.format(epoch + 1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))
    test(epoch)


# def train_(epoch):
#     t = time.time()
#     model_.train()
#     optimizer_.zero_grad()
#     output = model_(features, adj)
#     loss_train = loss_fn(output[idx_train], labels[idx_train])
#     acc_train = accuracy(output[idx_train], labels[idx_train])
#     loss_train.backward()
#     optimizer_.step()
#
#     model_.eval()
#     output = model_(features, adj)
#
#     loss_val = loss_fn(output[idx_val], labels[idx_val])
#     acc_val = accuracy(output[idx_val], labels[idx_val])
#     print('Epoch: {:04d}'.format(epoch + 1),
#           'loss_train: {:.4f}'.format(loss_train.item()),
#           'acc_train: {:.4f}'.format(acc_train.item()),
#           'loss_val: {:.4f}'.format(loss_val.item()),
#           'acc_val: {:.4f}'.format(acc_val.item()),
#           'time: {:.4f}s'.format(time.time() - t))
#     test_()
#
#
# def test_():
#     model_.eval()
#     output = model_(features, adj)
#     loss_test = loss_fn(output[idx_test], labels[idx_test])
#     acc_test = accuracy(output[idx_test], labels[idx_test])
#     print("Test set results:",
#           "loss= {:.4f}".format(loss_test.item()),
#           "accuracy= {:.4f}".format(acc_test.item()))


def test(epoch):
    model.eval()
    model_.eval()
    x, adj_new, output = model(features, edge)
    output_ = model_(features, adj)

    loss_test = loss_fn(output[idx_test], labels[idx_test])
    loss_test_ = loss_fn(output_[idx_test], labels[idx_test])

    gl_loss_test = gl_loss(x, adj_new, args.losslr1, args.losslr2)
    loss_test += gl_loss_test
    acc_test = accuracy(output[idx_test], labels[idx_test])
    acc_test_ = accuracy(output_[idx_test], labels[idx_test])
    test_acc_list.append(acc_test)
    writer.add_scalars(main_tag="Test-Acc",
                       tag_scalar_dict={
                           "glcn": acc_test,
                           "gcn": acc_test_
                       },
                       global_step=epoch + 1)
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))


# Train model
t_total = time.time()
for epoch in range(args.epochs):
    train(epoch)
    # train_(epoch)
writer.close()
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# # Testing
# test()
# Create a line plot
x = [i for i in range(1, args.epochs + 1)]

plt.plot(x, loss_list)

# Add labels and a title
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('0.1Loss-Epoch Plot')

# Show the plot
plt.show()
plt.plot(x, test_acc_list)
plt.xlabel('Epoch')
plt.ylabel('Acc on test')
plt.title('0.1alpha Acc-Epoch Plot')
plt.show()
