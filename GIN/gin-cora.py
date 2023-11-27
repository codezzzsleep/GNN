import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GINConv
import torch.nn.functional as F
# 设置随机种子以保证结果可重复
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True

# 加载 Cora 数据集
dataset = Planetoid(root='data/cora', name='cora')

# 数据预处理
data = dataset[0]
data.x = torch.FloatTensor(data.x)

# 定义 GIN 图神经网络模型
class GINClassifier(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes):
        super(GINClassifier, self).__init__()
        self.conv1 = GINConv([in_channels, hidden_channels * 2, hidden_channels])
        self.conv2 = GINConv([hidden_channels, hidden_channels * 2, hidden_channels])
        self.fc = nn.Linear(hidden_channels, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = x.unsqueeze(0)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = x.view(-1, self.conv2.out_channels)
        logits = self.fc(x)
        return logits

# 定义超参数和模型
in_channels = dataset.num_features
hidden_channels = 16
num_classes = dataset.num_classes
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GINClassifier(in_channels, hidden_channels, num_classes).to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 训练模型
def train():
    model.train()
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

# 评估模型
def evaluate():
    model.eval()
    output = model(data)
    pred = output.argmax(dim=1)
    acc = float((pred[data.test_mask] == data.y[data.test_mask]).sum()) / data.test_mask.sum()
    return acc

# 训练和评估
num_epochs = 200
for epoch in range(num_epochs):
    train_loss = train()
    acc = evaluate()
    print(f'Epoch: {epoch + 1}, Loss: {train_loss:.4f}, Accuracy: {acc:.4f}')

print(f'Final Accuracy: {evaluate():.4f}')