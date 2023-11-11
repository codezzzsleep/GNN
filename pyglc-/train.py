from models import *
from utils import *
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

x, edge, num_features, num_classes, data = get_pubmed()

writer = SummaryWriter("logs/run14-pubmed")
dropout = 0.6
epochs = 400
same_seed(3047)
nheads = 8
gcn_best_acc = 0
gat_best_acc = 0
model1 = GCN(num_features, 16, num_classes, dropout)
optimizer1 = optim.Adam(model1.parameters(), lr=0.01, weight_decay=1e-4)
loss_fn = nn.CrossEntropyLoss()
model2 = GAT(num_features, 16, num_classes, nheads, dropout)
optimizer2 = optim.Adam(model2.parameters(), lr=0.01, weight_decay=1e-4)


def train(epoch, gcn_best_acc=None, gat_best_acc=None):
    model1.train()
    model2.train()
    optimizer1.zero_grad()
    optimizer2.zero_grad()
    output1 = model1(x, edge)
    output2 = model2(x, edge)
    loss1 = loss_fn(output1[data.train_mask], data.y[data.train_mask])
    loss2 = loss_fn(output2[data.train_mask], data.y[data.train_mask])
    writer.add_scalars(main_tag="CrossEntropyLoss-train",
                       tag_scalar_dict={
                           "gcn": loss1,
                           "gat": loss2,
                       },
                       global_step=epoch + 1)
    train_acc1 = accuracy(output1[data.train_mask], data.y[data.train_mask])
    train_acc2 = accuracy(output2[data.train_mask], data.y[data.train_mask])

    writer.add_scalars(main_tag="Train-Acc",
                       tag_scalar_dict={
                           "gcn": train_acc1,
                           "gat": train_acc2,
                       },
                       global_step=epoch + 1)
    loss1.backward()
    loss2.backward()
    optimizer1.step()
    optimizer2.step()

    model1.eval()
    model2.eval()
    output1 = model1(x, edge)
    output2 = model2(x, edge)

    loss1 = loss_fn(output1[data.val_mask], data.y[data.val_mask])
    loss2 = loss_fn(output2[data.val_mask], data.y[data.val_mask])

    writer.add_scalars(main_tag="CrossEntropyLoss-val",
                       tag_scalar_dict={
                           "gcn": loss1,
                           "gat": loss2,
                       },
                       global_step=epoch + 1)
    val_acc1 = accuracy(output1[data.val_mask], data.y[data.val_mask])
    val_acc2 = accuracy(output2[data.val_mask], data.y[data.val_mask])

    writer.add_scalars(main_tag="Val-Acc",
                       tag_scalar_dict={
                           "gcn": val_acc1,
                           "gat": val_acc2,
                       },
                       global_step=epoch + 1)
    loss1 = loss_fn(output1[data.test_mask], data.y[data.test_mask])
    loss2 = loss_fn(output2[data.test_mask], data.y[data.test_mask])

    writer.add_scalars(main_tag="CrossEntropyLoss-test",
                       tag_scalar_dict={
                           "gcn": loss1,
                           "gat": loss2,
                       },
                       global_step=epoch + 1)
    test_acc1 = accuracy(output1[data.test_mask], data.y[data.test_mask])
    test_acc2 = accuracy(output2[data.test_mask], data.y[data.test_mask])
    if test_acc1 > gcn_best_acc:
        gcn_best_acc = test_acc1
    if test_acc2 > gat_best_acc:
        gat_best_acc = test_acc2
    writer.add_scalars(main_tag="Test-Acc",
                       tag_scalar_dict={
                           "gcn": test_acc1,
                           "gat": test_acc2,
                       },
                       global_step=epoch + 1)
    return gcn_best_acc, gat_best_acc


if __name__ == "__main__":
    for i in range(epochs):
        gcn_best_acc, gat_best_acc = train(i, gcn_best_acc, gat_best_acc)
        print(f"epoch {i} done!")
    print(f"gcn_best_acc {gat_best_acc}")
    print(f"gat_best_acc {gat_best_acc}")
writer.close()
