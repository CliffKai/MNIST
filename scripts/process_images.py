import os
from PIL import Image

def standardize_images(input_dir='../data/myData', output_dir='../data/standardize', size=(28,28)):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(input_dir, filename)
            try:
                with Image.open(img_path).convert('L') as img:  # 转为灰度
                    img = img.resize(size)  # 调整大小为28x28
                    # 保存标准化后的图片
                    save_path = os.path.join(output_dir, os.path.splitext(filename)[0] + '.png')
                    img.save(save_path)
                print(f"Processed and saved: {save_path}")
            except Exception as e:
                print(f"Failed to process {img_path}: {e}")

if __name__ == '__main__':
    standardize_images()
