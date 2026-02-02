import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


class ImageTransform:
    def __init__(self, output_dir='output'):
        self.output_dir = output_dir

    def apply_transformations(self, image_rgb):
        print("\n" + "=" * 60)
        print("模块2: 图像的基本变换")
        print("=" * 60)

        try:
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            height, width = image_bgr.shape[:2]

            scale_down = cv2.resize(image_bgr, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
            scale_up = cv2.resize(image_bgr, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR)

            center = (width // 2, height // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, 45, 1.0)
            cos = np.abs(rotation_matrix[0, 0])
            sin = np.abs(rotation_matrix[0, 1])

            new_width = int((height * sin) + (width * cos))
            new_height = int((height * cos) + (width * sin))

            rotation_matrix[0, 2] += (new_width / 2) - center[0]
            rotation_matrix[1, 2] += (new_height / 2) - center[1]

            rotated = cv2.warpAffine(image_bgr, rotation_matrix, (new_width, new_height))
            translation_matrix = np.float32([[1, 0, 50], [0, 1, -30]])
            translated = cv2.warpAffine(image_bgr, translation_matrix, (width, height))
            gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
            gray_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
            flip_horizontal = cv2.flip(image_bgr, 1)
            flip_vertical = cv2.flip(image_bgr, 0)

            scale_down_rgb = cv2.cvtColor(scale_down, cv2.COLOR_BGR2RGB)
            scale_up_rgb = cv2.cvtColor(scale_up, cv2.COLOR_BGR2RGB)
            rotated_rgb = cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB)
            translated_rgb = cv2.cvtColor(translated, cv2.COLOR_BGR2RGB)
            flip_horizontal_rgb = cv2.cvtColor(flip_horizontal, cv2.COLOR_BGR2RGB)
            flip_vertical_rgb = cv2.cvtColor(flip_vertical, cv2.COLOR_BGR2RGB)

            transformations = {
                'scale_down': scale_down_rgb,
                'scale_up': scale_up_rgb,
                'rotated': rotated_rgb,
                'translated': translated_rgb,
                'gray': gray_rgb,
                'flip_horizontal': flip_horizontal_rgb,
                'flip_vertical': flip_vertical_rgb
            }

            for name, img in transformations.items():
                save_path = os.path.join(self.output_dir, f'transform_{name}.jpg')
                cv2.imwrite(save_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

            plt.figure(figsize=(15, 10))
            transformations_info = [
                (image_rgb, '原始图像'),
                (scale_down_rgb, '0.5倍缩小'),
                (scale_up_rgb, '2倍放大'),
                (rotated_rgb, '45°旋转'),
                (translated_rgb, '平移'),
                (gray_rgb, '灰度图'),
                (flip_horizontal_rgb, '水平翻转'),
                (flip_vertical_rgb, '垂直翻转')
            ]

            for i, (img, title) in enumerate(transformations_info, 1):
                plt.subplot(3, 3, i)
                if 'gray' in title:
                    plt.imshow(img[:, :, 0], cmap='gray')
                else:
                    plt.imshow(img)
                plt.title(title)
                plt.axis('off')

            plt.suptitle('图像基本变换对比', fontsize=16, weight='bold')
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'transformations.png'), dpi=150, bbox_inches='tight')
            plt.show()

            print("图像变换已完成")

        except Exception as e:
            print(f"图像变换错误: {str(e)}")