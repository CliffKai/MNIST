import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import csv


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


# 定义CNN网络模型
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
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # 标准化与训练时一致
    ])
    img_tensor = transform(image).unsqueeze(0).to(device)  # 添加批次维度
    with torch.no_grad():
        output = model(img_tensor)
        probabilities = torch.softmax(output, dim=1)
        prob, predicted = torch.max(probabilities, 1)
    return predicted.item(), prob.item()


def main():
    # 定义目录路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.abspath(os.path.join(script_dir, '..', 'data', 'standardize'))  # 图片目录
    models_dir = os.path.abspath(os.path.join(script_dir, '..', 'models'))  # 模型目录
    fc_model_path = os.path.join(models_dir, 'fc_model.pth')  # 全连接模型路径
    cnn_model_path = os.path.join(models_dir, 'cnn_model.pth')  # CNN模型路径

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    # 检查图片目录是否存在
    if not os.path.exists(data_dir):
        print(f"图片目录不存在: {data_dir}")
        return

    # 创建CSV文件保存结果
    results_file = os.path.join(script_dir, 'prediction_results.csv')
    with open(results_file, mode='w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['Filename', 'FC_Prediction', 'FC_Probability', 'CNN_Prediction', 'CNN_Probability']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for filename in os.listdir(data_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(data_dir, filename)
                try:
                    with Image.open(img_path).convert('L') as img:
                        # 将文件名作为标签
                        label = filename

                        # 使用全连接模型预测
                        fc_pred, fc_prob = predict_image(fc_model, img, device)

                        # 使用CNN模型预测
                        cnn_pred, cnn_prob = predict_image(cnn_model, img, device)

                        # 输出结果
                        print(
                            f"图片标签为：{label}-模型FC预测值为：{fc_pred}-预测概率为：{fc_prob:.4f}-模型CNN预测值为：{cnn_pred}-预测概率为：{cnn_prob:.4f}")

                        # 写入CSV
                        writer.writerow({
                            'Filename': label,
                            'FC_Prediction': fc_pred,
                            'FC_Probability': f"{fc_prob:.4f}",
                            'CNN_Prediction': cnn_pred,
                            'CNN_Probability': f"{cnn_prob:.4f}"
                        })
                except Exception as e:
                    print(f"处理图片 {filename} 时出错: {e}")

    print(f"预测结果已保存到 {results_file}")


if __name__ == '__main__':
    main()
