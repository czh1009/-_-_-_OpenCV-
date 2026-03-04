# OpenCV图像识别测试项目

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.5+-green.svg)](https://opencv.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> 视觉组 - OpenCV图像识别基础实现

## 项目简介

本项目是视觉组OpenCV图像识别的实现，涵盖图像基础预处理,颜色阈值色块识别,简单特征识别的计算机视觉基础任务。

## 考核完成项目

### 一、基础必做任务

| 任务 | 描述 | 实现文件 |
|------|------|----------|
| 图像基础预处理 | 读取、灰度转换、高斯模糊、直方图均衡化 | `basic_preprocessing.py` |
| 颜色阈值色块识别 | HSV空间红/蓝色分割、形态学去噪、轮廓标注 | `color_detection.py` |
| 简单特征识别 | 矩形/圆形识别、0-9印刷体数字识别（无OCR库） | `detect_shapes_digits.py` |

## 项目结构

```
OpenCV_Test/
├── src/                          # 源代码目录
│   ├── main.py                   # 主程序入口
│   ├── basic_preprocessing.py    # 图像基础预处理
│   ├── color_detection.py        # 颜色阈值色块识别
│   ├── shape_number_recognition.py # 几何图形与数字识别
│   ├── armor_detection.py        # 装甲板精定位
│   ├── parameter_tuner.py        # 交互调参工具
│   └── generate_test_images.py   # 测试图像生成
├── images/                       # 测试图像目录
│   ├── basic_test.jpg            # 基础预处理测试图
│   ├── color_test.jpg            # 颜色识别测试图
│   ├── shape_number_test.jpg     # 形状数字识别测试图
│   ├── armor_test.jpg            # 装甲板检测测试图
│   ├── lighting/                 # 光照测试图像组
│   └── occlusion/                # 遮挡测试图像组
├── output/                       # 输出结果目录
├── docs/                         # 文档目录
└── README.md                     # 项目说明文档
```

## 快速开始

### 环境配置

#### 系统要求

- Python 3.7+
- OpenCV 4.5+
- NumPy 1.19+

#### 安装依赖

```bash
# 创建虚拟环境（推荐）
python -m venv venv

# 激活环境 (Windows)
.\venv\Scripts\activate

# 安装依赖包
pip install opencv-python numpy matplotlib

### 运行项目

#### 方式一：交互模式
```bash
cd src
python main.py
```

#### 方式二：命令行模式

```bash
# 运行全部测试
cd src
python main.py -t all

# 运行指定测试
cd src
python main.py -t preprocess -i ../images/basic_test.jpg
python main.py -t color -i ../images/color_test.jpg
python main.py -t shape -i ../images/shape_number_test.jpg

# 生成测试图像
cd src
python main.py --generate

# 交互调参
cd src
python main.py --tuner ../images/color_test.jpg color
```

#### 方式三：单独运行模块

```bash
cd src

# 图像预处理
python basic_preprocessing.py ../images/basic_test.jpg

# 颜色识别
python color_detection.py ../images/color_test.jpg

# 形状数字识别
python shape_number_recognition.py ../images/shape_number_test.jpg


## 实现思路

### 1. 图像基础预处理

```

输入图像 → 灰度转换 → 高斯模糊去噪 → 直方图均衡化 → 输出对比图

```

**关键算法：**
- `cv2.imread()`: 读取输入图像
- `cv2.cvtColor()`: 将 BGR 图像转换为单通道灰度图 
- `cv2.GaussianBlur()`: 高斯滤波去噪，抑制图像中的高频噪声
- `cv2.equalizeHist()`: 增强图像对比度，使输出图像的直方图平坦，从而让原本过暗或过亮的区域细节变清晰
- `matplotlib`:生成四格对比图。

### 2. 颜色阈值色块识别

```

输入图像 → BGR转HSV → 颜色阈值分割 → 形态学去噪 → 轮廓查找 → 标注信息

```

**关键算法：**
- `cv2.cvtColor(img, cv2.COLOR_BGR2HSV)`:将颜色、饱和度和亮度分离
- `cv2.inRange()`:定义颜色阈值（红色和蓝色）
- 形态学操作:先腐蚀后膨胀的操作
- `cv2.findContours()`:查找处理后的掩膜中的所有轮廓

### 3. 几何图形与数字识别

#### 几何图形识别
```

二值图 → 查找轮廓 → 轮廓近似 → 形状判定 → 标注

```

**形状分类规则：**
|判定条件| 识别为 | 标注颜色 |
|--------|------|------|
| 顶点数=4 且 内角平均值在75°~105 | 矩形 |  绿色框 + "Rect" |
| 圆度>0.8 且 外接矩形长宽比<1.2 且 面积>200 | 圆形 |  蓝色框 + "Circle" |
注：代码中只实现了矩形和圆形的检测，未实现三角形、五边形等分类

**关键算法：**
- 动态模板生成,从输入图像中提取实际数字区域作为模板,
- 模板匹配,置信度阈值：> 0.5
- 后处理,合并距离过近的候选区域

#### 数字识别（模板匹配法）
```

从输入图像中提取实际数字 → 缩放到40×60 → 保存为模板

```


## 测试结果

### 基础任务测试结果

| 测试项目 | 测试图像 | 输出结果 | 状态 |
|----------|----------|----------|------|
| 图像预处理 | basic_test.jpg | 灰度/模糊/均衡化对比图 | 实现 |
| 颜色识别 | color_test.jpg | 红/蓝色掩码+标注图 | 实现 |
| 形状识别 | shape_number_test.jpg | 几何图形分类标注 | 部分实现 |
| 数字识别 | shape_number_test.jpg | 0-9数字识别结果 | 实现 |


## 参数调优报告

### HSV颜色阈值参考值

| 颜色 | H下限 | H上限 | S下限 | S上限 | V下限 | V上限 |
|------|-------|-------|-------|-------|-------|-------|
| 红色 | 0-10, 160-179 | - | 100 | 255 | 100 | 255 |
| 蓝色 | 100 | 130 | 100 | 255 | 100 | 255 |

### 形态学操作参数

| 操作 | 核大小 | 迭代次数 | 用途 |
|------|--------|----------|------|
| 开运算 | 5×5 | 1 | 去除小噪点 |
| 闭运算 | 5×5 | 1 | 填补小孔洞 |

## 参考资料

### 官方文档
- [OpenCV官方文档](https://docs.opencv.org/4.x/)
- [OpenCV-Python教程](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html)
- LearnOpenCV：(https://learnopencv.com/)
- NumPy文档：(https://numpy.org/doc/stable/)
- HSV颜色空间详解(https://en.wikipedia.org/wiki/HSL_and_HSV)
- 形状识别算法：基于轮廓逼近的几何分类方法

### 核心算法参考
1. **图像预处理部分**
   - 形态学操作组合
   - 腐蚀膨胀分离数字笔画

2. **形状检测部分**
   - 圆度计算公式
   - 角度计算公式

3. **数字识别部分**
   - 多尺度匹配技术
   - 轮廓合并算法
   - 动态模板生成

## 许可证

本项目采用 MIT 许可证，详见 LICENSE 文件。

---
