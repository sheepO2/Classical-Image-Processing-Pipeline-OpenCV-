import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


class ImageSegmentation:
    def __init__(self, output_dir='output'):
        self.output_dir = output_dir

    def segment_image(self, image_path=None, image_gray=None):
        print("\n" + "=" * 60)
        print("模块6: 图像分割")
        print("=" * 60)

        try:
            if image_gray is not None:
                gray = image_gray
            elif image_path is not None:
                cell_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                if cell_image is None:
                    raise ValueError(f"无法读取图像: {image_path}")
                gray = cell_image
            else:
                raise ValueError("未提供分割图像")

            _, otsu_threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            def region_growing(img, seed_point, threshold=5):
                height, width = img.shape
                segmented = np.zeros_like(img)
                seed_list = [seed_point]

                while len(seed_list) > 0:
                    current_point = seed_list.pop(0)
                    x, y = current_point

                    if segmented[y, x] == 255:
                        continue

                    segmented[y, x] = 255

                    for dx in [-1, 0, 1]:
                        for dy in [-1, 0, 1]:
                            nx, ny = x + dx, y + dy

                            if 0 <= nx < width and 0 <= ny < height:
                                if segmented[ny, nx] == 0:
                                    if abs(int(img[ny, nx]) - int(img[y, x])) <= threshold:
                                        seed_list.append((nx, ny))

                return segmented

            height, width = gray.shape
            center_region = gray[height // 4:3 * height // 4, width // 4:3 * width // 4]
            mean_intensity = np.mean(center_region)
            seed_y, seed_x = np.unravel_index(np.argmin(np.abs(gray - mean_intensity)), gray.shape)
            seed_point = (seed_x, seed_y)
            region_grown = region_growing(gray, seed_point, threshold=5)

            segmentation_images = {
                'otsu': otsu_threshold,
                'region_growing': region_grown
            }

            for name, img in segmentation_images.items():
                save_path = os.path.join(self.output_dir, f'segment_{name}.jpg')
                cv2.imwrite(save_path, img)

            plt.figure(figsize=(15, 5))
            segmentation_info = [
                (gray, '原始灰度图'),
                (otsu_threshold, 'Otsu阈值分割'),
                (region_grown, f'区域生长法分割')
            ]

            for i, (img, title) in enumerate(segmentation_info, 1):
                plt.subplot(1, 3, i)
                plt.imshow(img, cmap='gray')
                plt.title(title)
                plt.axis('off')

                if title == '区域生长法分割':
                    plt.plot(seed_x, seed_y, 'r+', markersize=12, markeredgewidth=2)

            plt.suptitle('图像分割方法对比', fontsize=16, weight='bold')
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'segmentation.png'), dpi=150, bbox_inches='tight')
            plt.show()

            print("\n分割算法分析:")
            print("1. Otsu阈值分割: 自动确定最优阈值")
            print("2. 区域生长法分割: 能分割出连续区域")

            print("\n图像分割已完成")

        except Exception as e:
            print(f"图像分割错误: {str(e)}")
            self._create_sample_cell_image()

    def _create_sample_cell_image(self):
        img = np.ones((300, 300), dtype=np.uint8) * 200

        cv2.circle(img, (80, 80), 30, 100, -1)
        cv2.circle(img, (80, 80), 15, 60, -1)
        cv2.circle(img, (200, 100), 40, 90, -1)
        cv2.circle(img, (200, 100), 20, 50, -1)
        cv2.circle(img, (120, 200), 35, 110, -1)
        cv2.circle(img, (120, 200), 18, 70, -1)
        cv2.circle(img, (250, 220), 25, 80, -1)
        cv2.circle(img, (250, 220), 12, 40, -1)

        noise = np.random.randint(0, 30, img.shape[:2], dtype=np.uint8)
        img = cv2.add(img, noise)

        sample_path = os.path.join(self.output_dir, 'sample_cell_image.jpg')
        cv2.imwrite(sample_path, img)
        self.segment_image(sample_path)