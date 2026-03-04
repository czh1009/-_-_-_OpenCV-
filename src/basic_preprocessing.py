"""
任务1：图像基础预处理
功能：实现图片读取、灰度转换、高斯模糊去噪、直方图均衡化
输出：预处理前后的对比图
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def preprocess_image(img_path, output_dir='images/task1_output'):
    """
    图像预处理函数

    参数:
        img_path: 输入图像路径
        output_dir: 输出目录

    返回:
        原始BGR图像, 灰度图, 模糊图, 均衡化图
    """
    # 创建输出目录（如果不存在）
    os.makedirs(output_dir, exist_ok=True)

    # 1. 读取图像
    print(f"正在读取图像: {img_path}")
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        print(f"错误: 无法读取图像 {img_path}")
        return None

    # 2. 灰度转换
    # 使用cv2.COLOR_BGR2GRAY将BGR彩色图转为单通道灰度图
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # 3. 高斯模糊去噪
    # 参数: (5,5)是高斯核大小，1.5是标准差
    # 核大小必须为正奇数，越大模糊效果越强
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 1.5)

    # 4. 直方图均衡化
    # 增强图像对比度，使分布更均匀
    img_eq = cv2.equalizeHist(img_blur)

    # 保存各阶段结果
    cv2.imwrite(os.path.join(output_dir, '1_original.jpg'), img_bgr)
    cv2.imwrite(os.path.join(output_dir, '2_gray.jpg'), img_gray)
    cv2.imwrite(os.path.join(output_dir, '3_blur.jpg'), img_blur)
    cv2.imwrite(os.path.join(output_dir, '4_equalized.jpg'), img_eq)
    print(f"结果已保存至 {output_dir}")

    # 生成对比图（使用matplotlib）
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 原始图像（需转换BGR为RGB以便正确显示）
    axes[0, 0].imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('1. 原始图像', fontsize=14)
    axes[0, 0].axis('off')

    # 灰度图像
    axes[0, 1].imshow(img_gray, cmap='gray')
    axes[0, 1].set_title('2. 灰度转换', fontsize=14)
    axes[0, 1].axis('off')

    # 高斯模糊
    axes[1, 0].imshow(img_blur, cmap='gray')
    axes[1, 0].set_title('3. 高斯模糊(5x5)', fontsize=14)
    axes[1, 0].axis('off')

    # 直方图均衡化
    axes[1, 1].imshow(img_eq, cmap='gray')
    axes[1, 1].set_title('4. 直方图均衡化', fontsize=14)
    axes[1, 1].axis('off')

    plt.tight_layout()
    comparison_path = os.path.join(output_dir, 'preprocessing_comparison.png')
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"对比图已保存至 {comparison_path}")

    return img_bgr, img_gray, img_blur, img_eq

def show_histogram(img, title, output_dir):
    """显示并保存直方图"""
    plt.figure(figsize=(8, 4))
    plt.hist(img.ravel(), 256, [0, 256])
    plt.title(title)
    plt.xlabel('像素值')
    plt.ylabel('频数')
    hist_path = os.path.join(output_dir, f'{title}_histogram.png')
    plt.savefig(hist_path)
    plt.show()
    print(f"直方图已保存至 {hist_path}")

if __name__ == "__main__":
    # 测试图像路径（请根据实际情况修改）
    test_image = r"C:\Users\czh18\Desktop\test\basic_test.jpg"

    # 执行预处理
    result = preprocess_image(test_image)

    if result:
        img_bgr, img_gray, img_blur, img_eq = result

        # 显示直方图对比
        show_histogram(img_gray, '灰度图直方图', 'images/task1_output')
        show_histogram(img_eq, '均衡化后直方图', 'images/task1_output')

        print("\n任务1完成！处理流程：")
        print("1. 图像读取 -> 2. 灰度转换 -> 3. 高斯模糊 -> 4. 直方图均衡化")