import os
import numpy as np
import cv2
from image_io import ImageIO
from transform import ImageTransform
from enhance import ImageEnhancement
from features import FeatureExtraction
from segment import ImageSegmentation


class ImageProcessingDemo:
    def __init__(self, output_dir='output'):
        self.output_dir = output_dir
        self.io = ImageIO(output_dir)
        self.transform = ImageTransform(output_dir)
        self.enhance = ImageEnhancement(output_dir)
        self.features = FeatureExtraction(output_dir)
        self.segment = ImageSegmentation(output_dir)

    def run_pipeline(self, lena_image_path, cell_image_path=None):
        print("=" * 80)
        print("图像处理系统 - 完整流水线")
        print("=" * 80)

        try:
            image_rgb, image_info = self.io.read_image_and_info(lena_image_path)
            if image_rgb is None:
                raise ValueError("图像读取失败")

            self.transform.apply_transformations(image_rgb)
            self.enhance.enhance_and_restore(image_rgb)
            self.features.extract_features(image_rgb)

            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

            if cell_image_path:
                self.segment.segment_image(cell_image_path)
            else:
                self.segment.segment_image(image_gray=gray)

            self.generate_report(image_info)
            print("\n" + "=" * 80)
            print("图像处理流水线完成!")
            print(f"所有结果已保存到: {os.path.abspath(self.output_dir)}")
            print("=" * 80)

        except Exception as e:
            print(f"\n系统执行错误: {str(e)}")

    def generate_report(self, image_info):
        report_path = os.path.join(self.output_dir, 'processing_report.txt')

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("图像处理系统 - 处理报告\n")
            f.write("=" * 60 + "\n\n")

            f.write("一、处理模块概览\n")
            f.write("=" * 40 + "\n")
            f.write("1. 图像读取与基础信息展示 ✓\n")
            f.write("2. 图像的基本变换 ✓\n")
            f.write("3. 图像的增强和复原 ✓\n")
            f.write("4. 图像特征提取 ✓\n")
            f.write("5. 图像分割 ✓\n\n")

            if image_info:
                f.write("二、原始图像信息\n")
                f.write("=" * 40 + "\n")
                for key, value in image_info.items():
                    f.write(f"{key}: {value}\n")

            f.write("\n处理完成时间: " + str(np.datetime64('now')) + "\n")
            f.write("=" * 60 + "\n")

        print(f"处理报告已生成: {report_path}")


if __name__ == "__main__":
    processor = ImageProcessingDemo(output_dir='image_processing_output')
    lena_path = 'lena.jpg'
    cell_path = 'cell_image.jpg' if os.path.exists('cell_image.jpg') else None

    if not os.path.exists(lena_path):
        print(f"注意: {lena_path} 不存在")
        print("创建示例图像...")

        try:
            example_img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
            cv2.imwrite('lena.jpg', example_img)
            print("已保存示例图像: lena.jpg")
        except:
            print("使用随机生成的示例图像")

    try:
        processor.run_pipeline(lena_path, cell_path)
    except Exception as e:
        print(f"运行错误: {str(e)}")
        print("尝试使用内置示例运行...")
        processor.segment._create_sample_cell_image()
