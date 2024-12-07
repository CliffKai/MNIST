import os
import torchvision
import torchvision.transforms as transforms

def download_mnist(data_dir='../data'):
    """
    下载MNIST数据集到指定目录。
    """
    transform = transforms.ToTensor()
    train_dataset = torchvision.datasets.MNIST(
        root=data_dir, train=True, transform=transform, download=True
    )
    test_dataset = torchvision.datasets.MNIST(
        root=data_dir, train=False, transform=transform, download=True
    )
    print(f"MNIST数据集已下载到 {data_dir}")

if __name__ == '__main__':
    download_mnist()
