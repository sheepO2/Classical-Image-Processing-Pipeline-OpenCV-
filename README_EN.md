# Lena Image Processing Experiment (Course Project)

Project Background

This project was completed as a final assignment for my image processing course,
based on the classic Lena test image, implementing and validating various traditional image processing algorithms.

The project goal is not industrial-grade performance, but to understand the principles, applicability, and limitations of classical image processing methods through a complete experimental workflow, serving as foundational training for computer vision and AI.

## Project Positioning

This project is a **course experiment / educational project**  
Uses the classic Lena image as a standard test sample.

- For course learning and algorithm validation  
- To demonstrate the complete workflow of traditional image processing  
- Does not involve deep learning models or training processes  

**This project does not disguise itself as a production system and fully retains its course-oriented nature.**

## Data Used

Test Image: Lena (Lenna)  

Purpose: Classic standard test image in image processing and computer vision  

Application Scenarios: Algorithm comparison, effect demonstration, teaching and experimental validation  

The Lena image is widely used for:  

- Comparing image enhancement effects  
- Noise processing and filtering experiments  
- Image segmentation and feature extraction validation  

## Project Structure
~~~
demo/
├──image_processing_output/ # Output images
  ├── original_rgb.jpg      # Original RGB image
  ├── image_info.png        # Image info chart
  ├── enhancement.png      # Enhancement comparison
  ├── restoration.png      # Restoration comparison
  ├── features.png      # Feature extraction result
  ├── segmentation.png    # Segmentation comparison
  ├── processing_report.txt     # Processing report
  └── sample_cell_image.jpg     # Sample cell image
├── image_io.py     # Lena image loading and basic info display
├── enhance.py    # Image enhancement and restoration experiments
├── features.py        # Feature extraction experiments (histogram / GLCM / edges)
├── segment.py         # Image segmentation experiments (Otsu / region growing)
├── demo.py            # Experiment workflow demo entry
└── README.md
~~~
Modules are divided according to experimental content, making them easy to run and understand individually.

## Experiment Overview

### **1️. Image Loading and Basic Info Analysis**

- Analyze Lena image dimensions, channels, and pixel range  
- Convert between RGB and grayscale images  

### **2️. Image Enhancement and Restoration Experiments**

- Global histogram equalization  
- CLAHE adaptive histogram equalization  
- Salt-and-pepper / Gaussian noise simulation  
- Median filtering, Gaussian filtering for denoising  

***Used to observe the impact of different enhancement and denoising methods on image details***

### **3️. Image Feature Extraction Experiments**

- Grayscale histogram statistical features  
- Gray Level Co-occurrence Matrix (GLCM) texture features  
- Canny / Sobel edge detection  

***Used to understand traditional representations of image features***

### **4️. Image Segmentation Experiments**

- Otsu automatic thresholding  
- Region growing segmentation  

***Compare segmentation methods under different conditions for stability and sensitivity***

Technical Stack
---
- Python  
- OpenCV  
- scikit-image  
- NumPy  
- Matplotlib  

Run Instructions
---
```bash
python demo.py
