import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import numpy as np


# 定义全连接网络模型
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


# 定义卷积神经网络模型
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
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


def load_model(model_type, model_path, device):
    """
    加载指定类型的模型。

    参数:
        model_type (str): 'fc' 或 'cnn'
        model_path (str): 模型文件路径
        device (torch.device): 设备

    返回:
        model (nn.Module): 加载后的模型
    """
    if model_type == 'fc':
        model = FCModel().to(device)
    elif model_type == 'cnn':
        model = CNNModel().to(device)
    else:
        raise ValueError("model_type 必须是 'fc' 或 'cnn'")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件未找到: {model_path}")

    # 设置 weights_only=True 以提高安全性
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    return model


def predict_image(model, image, device):
    """
    使用模型对单张图片进行预测。

    参数:
        model (nn.Module): 已加载的模型
        image (PIL.Image): 要预测的图片
        device (torch.device): 设备

    返回:
        predicted (int): 预测的类别
        probability (float): 预测的概率
    """
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # 与训练时一致的标准化
    ])
    img_tensor = transform(image).unsqueeze(0).to(device)  # 添加批次维度
    with torch.no_grad():
        output = model(img_tensor)
        probabilities = torch.softmax(output, dim=1)
        prob, predicted = torch.max(probabilities, 1)
    return predicted.item(), prob.item()


def plot_confusion_matrix(cm, classes, model_type, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    绘制混淆矩阵。

    参数:
        cm (array, shape = [n, n]): 混淆矩阵
        classes (list): 类别名称
        model_type (str): 模型类型（用于标题）
        normalize (bool): 是否归一化
        title (str): 标题
        cmap (matplotlib.colors.Colormap): 颜色映射
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(f'{model_type} {title}')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
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


def main():
    # 定义目录路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.abspath(os.path.join(script_dir, '..', 'data'))  # 官方测试集路径
    models_dir = os.path.abspath(os.path.join(script_dir, '..', 'models'))  # 模型目录
    fc_model_path = os.path.join(models_dir, 'fc_model.pth')  # 全连接模型路径
    cnn_model_path = os.path.join(models_dir, 'cnn_model.pth')  # CNN模型路径

    # 选择设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 加载模型
    try:
        fc_model = load_model('fc', fc_model_path, device)
        print(f"成功加载全连接模型: {fc_model_path}")
    except Exception as e:
        print(f"加载全连接模型时出错: {e}")
        return

    try:
        cnn_model = load_model('cnn', cnn_model_path, device)
        print(f"成功加载CNN模型: {cnn_model_path}")
    except Exception as e:
        print(f"加载CNN模型时出错: {e}")
        return

    # 加载MNIST官方测试集
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_dataset = datasets.MNIST(root=data_dir, train=False, transform=test_transform, download=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)
    print(f"加载MNIST官方测试集，共有 {len(test_dataset)} 张图片。")

    # 定义类别名称
    classes = [str(i) for i in range(10)]

    # 函数用于评估模型
    def evaluate(model, loader, device):
        model.eval()
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        acc = correct / total
        print(f"{model.__class__.__name__} 在官方测试集上的准确率: {acc * 100:.2f}%")

        # 混淆矩阵和分类报告
        cm = confusion_matrix(all_labels, all_preds)
        print(f"{model.__class__.__name__} 的混淆矩阵:\n{cm}")
        print(f"{model.__class__.__name__} 的分类报告:\n{classification_report(all_labels, all_preds)}")

        # 绘制混淆矩阵
        plot_confusion_matrix(cm, classes, model.__class__.__name__, normalize=False, title='Confusion Matrix')

    # 评估全连接模型
    print("\n评估全连接网络在官方测试集上的表现...")
    evaluate(fc_model, test_loader, device)

    # 评估CNN模型
    print("\n评估CNN网络在官方测试集上的表现...")
    evaluate(cnn_model, test_loader, device)


if __name__ == '__main__':
    main()
