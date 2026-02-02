import os
import warnings
import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage import io, color

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'KaiTi', 'FangSong']
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')


class ImageIO:
    def __init__(self, output_dir='output'):
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def read_image_and_info(self, image_path):
        print("=" * 60)
        print("模块1: 图像读取与基础信息展示")
        print("=" * 60)

        try:
            image = cv2.imread(image_path)
            if image is None:
                image = io.imread(image_path)
                if len(image.shape) == 3 and image.shape[2] == 4:
                    image = color.rgba2rgb(image)
                    image = (image * 255).astype(np.uint8)

            if image is None:
                raise ValueError(f"无法读取图像: {image_path}")

            height, width = image.shape[:2]
            channels = 1 if len(image.shape) == 2 else image.shape[2]
            dtype = str(image.dtype)
            pixel_range = f"[{image.min()}, {image.max()}]"

            if len(image.shape) == 3:
                if channels == 4:
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
                else:
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

            save_path = os.path.join(self.output_dir, 'original_rgb.jpg')
            cv2.imwrite(save_path, cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))

            print(f"图像信息:")
            print(f"  - 尺寸: {width} × {height}")
            print(f"  - 通道数: {channels}")
            print(f"  - 数据类型: {dtype}")
            print(f"  - 像素值范围: {pixel_range}")
            print(f"  - 图像已保存至: {save_path}")

            plt.figure(figsize=(15, 5))
            plt.subplot(1, 2, 1)
            if channels == 1:
                plt.imshow(image, cmap='gray')
            else:
                plt.imshow(image_rgb)
            plt.title('原始图像')
            plt.axis('off')

            plt.subplot(1, 2, 2)
            plt.text(0.1, 0.9, f"图像基本信息", fontsize=14, weight='bold')
            plt.text(0.1, 0.7, f"尺寸: {width} × {height}", fontsize=12)
            plt.text(0.1, 0.6, f"通道数: {channels}", fontsize=12)
            plt.text(0.1, 0.5, f"数据类型: {dtype}", fontsize=12)
            plt.text(0.1, 0.4, f"像素范围: {pixel_range}", fontsize=12)
            plt.text(0.1, 0.3, f"文件: {os.path.basename(image_path)}", fontsize=10)
            plt.axis('off')
            plt.title('图像信息')

            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'image_info.png'), dpi=150, bbox_inches='tight')
            plt.show()

            image_info = {
                'width': width,
                'height': height,
                'channels': channels,
                'dtype': dtype,
                'pixel_range': pixel_range
            }

            return image_rgb, image_info

        except Exception as e:
            print(f"图像读取错误: {str(e)}")
            return None, None