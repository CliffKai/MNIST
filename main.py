import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import numpy as np

########################################################
# 1. 数据预处理和增强
########################################################

# 数据增强和标准化
train_transform = transforms.Compose([
    transforms.RandomRotation(10),  # 随机旋转 ±10 度
    transforms.RandomAffine(0, translate=(0.1, 0.1)),  # 随机平移
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # 标准化 (MNIST的均值和标准差)
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])


########################################################
# 2. 数据集加载与划分
########################################################

def get_datasets(data_dir='./data'):
    # 下载并加载训练集和测试集
    full_train_dataset = datasets.MNIST(root=data_dir, train=True, transform=train_transform, download=True)
    test_dataset = datasets.MNIST(root=data_dir, train=False, transform=test_transform, download=True)

    # 划分出验证集
    train_size = int(0.9 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

    return train_dataset, val_dataset, test_dataset


########################################################
# 3. 用全连接网络实现MNIST数据集分类的模型训练
########################################################

class FCModel(nn.Module):
    def __init__(self):
        super(FCModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        return self.fc(x)


########################################################
# 4. 用CNN网络实现MNIST数据集分类的模型训练
########################################################

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # 28x28 -> 28x28
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 28x28 -> 28x28
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 28x28 -> 14x14
            nn.Dropout(0.25),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # 14x14 -> 14x14
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 14x14 -> 7x7
            nn.Dropout(0.25)
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * 7 * 7, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


########################################################
# 5. 训练函数通用化
########################################################

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, epochs=20,
                model_path='models/model.pth'):
    best_acc = 0.0
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct / total
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)

        # 验证阶段
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        epoch_val_loss = val_loss / len(val_loader.dataset)
        epoch_val_acc = correct_val / total_val
        val_losses.append(epoch_val_loss)
        val_accuracies.append(epoch_val_acc)

        # 调整学习率
        scheduler.step(epoch_val_loss)

        print(f"Epoch [{epoch + 1}/{epochs}] "
              f"Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc * 100:.2f}% "
              f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc * 100:.2f}%")

        # 保存最佳模型
        if epoch_val_acc > best_acc:
            best_acc = epoch_val_acc
            torch.save(model.state_dict(), model_path)
            print(f"--> 保存最佳模型到 {model_path}")

    # 绘制训练和验证损失曲线
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), train_losses, label='Train Loss')
    plt.plot(range(1, epochs + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()

    # 绘制训练和验证准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs + 1), train_accuracies, label='Train Accuracy')
    plt.plot(range(1, epochs + 1), val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Curve')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.show()

    print(f"最佳验证准确率: {best_acc * 100:.2f}%")


########################################################
# 6. 主函数
########################################################

def main():
    # 创建模型目录
    if not os.path.exists('models'):
        os.makedirs('models')

    # 获取数据集
    train_dataset, val_dataset, test_dataset = get_datasets()

    # 数据加载器
    batch_size = 128
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 训练全连接网络
    print("开始训练全连接网络...")
    fc_model = FCModel().to(device)
    fc_criterion = nn.CrossEntropyLoss()
    fc_optimizer = optim.Adam(fc_model.parameters(), lr=0.001)
    fc_scheduler = optim.lr_scheduler.ReduceLROnPlateau(fc_optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    train_model(fc_model, train_loader, val_loader, fc_criterion, fc_optimizer, fc_scheduler, device, epochs=20,
                model_path='models/fc_model.pth')

    # 训练CNN网络
    print("\n开始训练CNN网络...")
    cnn_model = CNNModel().to(device)
    cnn_criterion = nn.CrossEntropyLoss()
    cnn_optimizer = optim.Adam(cnn_model.parameters(), lr=0.001)
    cnn_scheduler = optim.lr_scheduler.ReduceLROnPlateau(cnn_optimizer, mode='min', factor=0.5, patience=3,
                                                         verbose=True)
    train_model(cnn_model, train_loader, val_loader, cnn_criterion, cnn_optimizer, cnn_scheduler, device, epochs=20,
                model_path='models/cnn_model.pth')

    # 评估测试集
    print("\n评估全连接网络在测试集上的表现...")
    evaluate_model(fc_model, test_loader, device, model_type='FC')

    print("\n评估CNN网络在测试集上的表现...")
    evaluate_model(cnn_model, test_loader, device, model_type='CNN')


########################################################
# 7. 测试函数
########################################################

def evaluate_model(model, test_loader, device, model_type='Model'):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = correct / total
    print(f"{model_type} 在测试集上的准确率: {acc * 100:.2f}%")

    # 混淆矩阵和分类报告
    cm = confusion_matrix(all_labels, all_preds)
    print(f"{model_type} 的混淆矩阵:\n{cm}")
    print(f"{model_type} 的分类报告:\n{classification_report(all_labels, all_preds)}")

    # 可视化混淆矩阵
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'{model_type} Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(10)
    plt.xticks(tick_marks, tick_marks)
    plt.yticks(tick_marks, tick_marks)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(f'{model_type}_confusion_matrix.png')
    plt.show()


if __name__ == '__main__':
    main()
