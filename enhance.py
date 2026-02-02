import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.util import random_noise


class ImageEnhancement:
    def __init__(self, output_dir='output'):
        self.output_dir = output_dir

    def enhance_and_restore(self, image_rgb):
        print("\n" + "=" * 60)
        print("模块3: 图像的增强和复原")
        print("=" * 60)

        try:
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

            hist_eq = cv2.equalizeHist(gray)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            clahe_eq = clahe.apply(gray)

            fig_enhance = plt.figure(figsize=(15, 10))
            images_enhance = [gray, hist_eq, clahe_eq]
            titles_enhance = ['原始灰度图', '全局直方图均衡化', '自适应直方图均衡化']

            for i, (img, title) in enumerate(zip(images_enhance, titles_enhance), 1):
                plt.subplot(2, 3, i)
                plt.imshow(img, cmap='gray')
                plt.title(title)
                plt.axis('off')

                plt.subplot(2, 3, i + 3)
                plt.hist(img.ravel(), 256, [0, 256], color='blue', alpha=0.7)
                plt.title(f'{title}直方图')
                plt.xlabel('像素值')
                plt.ylabel('频数')
                plt.grid(True, alpha=0.3)

            plt.suptitle('图像增强对比', fontsize=16, weight='bold')
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'enhancement.png'), dpi=150, bbox_inches='tight')
            plt.show()

            noise_sp = random_noise(gray, mode='s&p', amount=0.02)
            noise_sp = (255 * noise_sp).astype(np.uint8)

            noise_gaussian = random_noise(gray, mode='gaussian', mean=0, var=0.005)
            noise_gaussian = (255 * noise_gaussian).astype(np.uint8)

            blurred = cv2.GaussianBlur(gray, (5, 5), sigmaX=2, sigmaY=2)
            denoise_sp = cv2.medianBlur(noise_sp, 3)
            denoise_gaussian = cv2.GaussianBlur(noise_gaussian, (3, 3), sigmaX=1, sigmaY=1)
            deblurred = cv2.GaussianBlur(blurred, (5, 5), sigmaX=2, sigmaY=2)

            restoration_images = {
                'noise_sp': noise_sp,
                'noise_gaussian': noise_gaussian,
                'blurred': blurred,
                'denoise_sp': denoise_sp,
                'denoise_gaussian': denoise_gaussian,
                'deblurred': deblurred
            }

            for name, img in restoration_images.items():
                save_path = os.path.join(self.output_dir, f'restore_{name}.jpg')
                cv2.imwrite(save_path, img)

            fig_restore = plt.figure(figsize=(15, 10))
            restoration_info = [
                (gray, '原始灰度图'),
                (noise_sp, '椒盐噪声'),
                (denoise_sp, '中值滤波去噪'),
                (noise_gaussian, '高斯噪声'),
                (denoise_gaussian, '高斯滤波去噪'),
                (blurred, '高斯模糊'),
                (deblurred, '高斯滤波复原')
            ]

            for i, (img, title) in enumerate(restoration_info, 1):
                plt.subplot(3, 3, i)
                plt.imshow(img, cmap='gray')
                plt.title(title)
                plt.axis('off')

            plt.suptitle('图像复原对比', fontsize=16, weight='bold')
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'restoration.png'), dpi=150, bbox_inches='tight')
            plt.show()

            print("图像增强与复原已完成")

        except Exception as e:
            print(f"图像增强错误: {str(e)}")