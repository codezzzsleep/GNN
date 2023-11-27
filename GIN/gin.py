import torch
import torch.nn as nn
import torch_geometric as pyg
import torch_geometric.data as pyg_data

from torch_geometric.datasets import Planetoid
dataset = Planetoid(root='/tmp/Cora',name='Cora')
class GINConv(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, eps=0.5):
        super(GINConv, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        self.eps = eps

    def forward(self, x, edge_index):
        edge_attr = self.mlp(x[edge_index[0]] + x[edge_index[1]])
        return self.eps * edge_attr

class GINModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes):
        super(GINModel, self).__init__()
        self.conv1 = GINConv(in_channels, hidden_channels, hidden_channels)
        self.conv2 = GINConv(hidden_channels, hidden_channels, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = x.to_dense()
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GINModel(dataset[0].num_node_features, 32, dataset[0].num_classes).to(device)
data = dataset[0].to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = nn.CrossEntropyLoss()

def train(model, data, optimizer, epoch):
    model.train()
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

def test(model, data):
    model.eval()
    output = model(data)
    pred = output.argmax(dim=1)
    acc = (pred[data.test_mask] == data.y[data.test_mask]).sum().item() / data.test_mask.sum().item()
    return acc

for epoch in range(200):
    loss = train(model, data, optimizer, epoch)
    if epoch % 10 == 0:
        print(f'Epoch: {epoch:03d}, Loss: {loss:.3f}')
    acc = test(model, data)
    print(f'Test Acc: {acc:.3f}')

print(f'Final Test Acc: {test(model, data):.3f}')