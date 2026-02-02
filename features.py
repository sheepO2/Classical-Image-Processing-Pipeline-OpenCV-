import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import graycomatrix, graycoprops


class FeatureExtraction:
    def __init__(self, output_dir='output'):
        self.output_dir = output_dir

    def extract_features(self, image_rgb):
        print("\n" + "=" * 60)
        print("模块5: 图像特征提取")
        print("=" * 60)

        try:
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            hist = hist.flatten()
            hist_mean = np.mean(gray)
            hist_var = np.var(gray)
            hist_peak = np.argmax(hist)

            print("颜色特征:")
            print(f"  - 均值: {hist_mean:.2f}")
            print(f"  - 方差: {hist_var:.2f}")
            print(f"  - 峰值位置: {hist_peak}")

            glcm = graycomatrix(gray, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
            contrast = graycoprops(glcm, 'contrast')[0, 0]
            correlation = graycoprops(glcm, 'correlation')[0, 0]
            energy = graycoprops(glcm, 'energy')[0, 0]
            homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]

            print("\n纹理特征:")
            print(f"  - 对比度: {contrast:.4f}")
            print(f"  - 相关性: {correlation:.4f}")
            print(f"  - 能量: {energy:.4f}")
            print(f"  - 同质性: {homogeneity:.4f}")

            edges_canny = cv2.Canny(gray, 100, 200)
            sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            sobel_combined = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
            sobel_combined = np.uint8(sobel_combined / sobel_combined.max() * 255)

            feature_images = {
                'canny': edges_canny,
                'sobel': sobel_combined
            }

            for name, img in feature_images.items():
                save_path = os.path.join(self.output_dir, f'feature_{name}.jpg')
                cv2.imwrite(save_path, img)

            plt.figure(figsize=(15, 10))

            plt.subplot(2, 3, 1)
            plt.imshow(gray, cmap='gray')
            plt.title('原始灰度图')
            plt.axis('off')

            plt.subplot(2, 3, 2)
            plt.bar(range(256), hist, width=1.0, color='blue', alpha=0.7)
            plt.title('灰度直方图')
            plt.xlabel('像素值')
            plt.ylabel('频数')
            plt.grid(True, alpha=0.3)

            plt.subplot(2, 3, 3)
            plt.text(0.1, 0.9, "颜色特征统计", fontsize=12, weight='bold')
            plt.text(0.1, 0.7, f"均值: {hist_mean:.2f}", fontsize=10)
            plt.text(0.1, 0.6, f"方差: {hist_var:.2f}", fontsize=10)
            plt.text(0.1, 0.5, f"峰值: {hist_peak}", fontsize=10)
            plt.axis('off')
            plt.title('颜色特征')

            plt.subplot(2, 3, 4)
            plt.text(0.1, 0.9, "纹理特征", fontsize=12, weight='bold')
            plt.text(0.1, 0.7, f"对比度: {contrast:.4f}", fontsize=10)
            plt.text(0.1, 0.6, f"相关性: {correlation:.4f}", fontsize=10)
            plt.text(0.1, 0.5, f"能量: {energy:.4f}", fontsize=10)
            plt.text(0.1, 0.4, f"同质性: {homogeneity:.4f}", fontsize=10)
            plt.axis('off')
            plt.title('纹理特征')

            plt.subplot(2, 3, 5)
            plt.imshow(edges_canny, cmap='gray')
            plt.title('Canny边缘检测')
            plt.axis('off')

            plt.subplot(2, 3, 6)
            plt.imshow(sobel_combined, cmap='gray')
            plt.title('Sobel边缘检测')
            plt.axis('off')

            plt.suptitle('图像特征提取', fontsize=16, weight='bold')
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'features.png'), dpi=150, bbox_inches='tight')
            plt.show()

            print("特征提取已完成")

        except Exception as e:
            print(f"特征提取错误: {str(e)}")