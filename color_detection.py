"""
任务2：颜色阈值色块识别
功能：基于HSV颜色空间实现红/蓝色目标分割
      结合形态学腐蚀膨胀操作去除噪点
      筛选有效目标轮廓并标注坐标、面积信息
"""

'''
实现思路
1. 色彩空间转换：BGR转HSV，cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
  HSV对光照变化更鲁棒，便于颜色分割
2. 红色阈值设定：红色在HSV空间分布在0°两端
 区间1：Hue 0~10，Saturation 50~255，Value 50~255
 区间2：Hue 160~180，Saturation 50~255，Value 50~255
 两个区间取并集
3. 蓝色阈值设定：
 Hue 100~130，Saturation 50~255，Value 50~255
4. 形态学操作：
 腐蚀：cv2.erode()去除小噪点，迭代1次
 膨胀：cv2.dilate()恢复目标大小，迭代2次
 核大小：5×5矩形结构元素
5. 轮廓筛选与标注：
 cv2.findContours()查找轮廓
 筛选面积>100像素的有效目标
 绘制边界框，标注中心坐标和面积
'''

import cv2
import numpy as np
import os


def create_color_mask(hsv_img, color_range):
    """
    根据HSV范围创建颜色掩膜

    参数:
        hsv_img: HSV格式图像
        color_range: 颜色范围列表，每个元素为(lower, upper)

    返回:
        二值掩膜图像
    """
    mask = None
    for lower, upper in color_range:
        # 创建当前区间的掩膜
        current_mask = cv2.inRange(hsv_img, lower, upper)
        # 合并多个区间（适用于红色有两个区间的情况）
        if mask is None:
            mask = current_mask
        else:
            mask = cv2.bitwise_or(mask, current_mask)

    return mask


def morphological_operations(mask, kernel_size=5, erode_iter=1, dilate_iter=2):
    """
    应用形态学操作：先腐蚀后膨胀，用于去除噪点并恢复目标

    参数:
        mask: 输入二值掩膜
        kernel_size: 结构元素大小
        erode_iter: 腐蚀迭代次数
        dilate_iter: 膨胀迭代次数

    返回:
        处理后的掩膜
    """
    # 创建结构元素（矩形）
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # 腐蚀：去除小白点噪点
    eroded = cv2.erode(mask, kernel, iterations=erode_iter)

    # 膨胀：恢复目标大小，连接断裂区域
    dilated = cv2.dilate(eroded, kernel, iterations=dilate_iter)

    return dilated


def process_contours(img, mask_red, mask_blue, min_area=100):
    """
    处理红色和蓝色掩膜的轮廓，标注识别结果

    参数:
        img: 原始BGR图像
        mask_red: 红色掩膜
        mask_blue: 蓝色掩膜
        min_area: 最小轮廓面积（用于过滤噪点）

    返回:
        标注后的图像
    """
    result = img.copy()

    # 处理红色轮廓
    contours_red, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    red_count = 0

    for contour in contours_red:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue  # 忽略小面积噪点

        red_count += 1

        # 获取边界框
        x, y, w, h = cv2.boundingRect(contour)
        center = (x + w // 2, y + h // 2)

        # 绘制红色边界框 (BGR: 0,0,255 表示红色)
        cv2.rectangle(result, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # 标注中心点（黄色圆点）
        cv2.circle(result, center, 4, (0, 255, 255), -1)

        # 标注坐标信息
        coord_text = f"R({center[0]},{center[1]})"
        cv2.putText(result, coord_text, (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # 标注面积信息
        area_text = f"Area:{int(area)}"
        cv2.putText(result, area_text, (x, y + h + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        print(f"红色目标{red_count}: 中心{center}, 面积{int(area)}像素")

    # 处理蓝色轮廓
    contours_blue, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blue_count = 0

    for contour in contours_blue:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        blue_count += 1

        x, y, w, h = cv2.boundingRect(contour)
        center = (x + w // 2, y + h // 2)

        # 绘制蓝色边界框 (BGR: 255,0,0 表示蓝色)
        cv2.rectangle(result, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # 标注中心点
        cv2.circle(result, center, 4, (0, 255, 255), -1)

        # 标注坐标信息
        coord_text = f"B({center[0]},{center[1]})"
        cv2.putText(result, coord_text, (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        # 标注面积信息
        area_text = f"Area:{int(area)}"
        cv2.putText(result, area_text, (x, y + h + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

    print(f"\n总计检测到: {red_count}个红色目标, {blue_count}个蓝色目标")
    return result


def detect_colors(img_path, output_dir='images/task2_output'):
    """
    主函数：颜色识别

    参数:
        img_path: 输入图像路径
        output_dir: 输出目录
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 1. 读取图像
    print(f"正在处理图像: {img_path}")
    img = cv2.imread(img_path)
    if img is None:
        print(f"错误: 无法读取图像 {img_path}")
        return None

    # 2. BGR转HSV
    # HSV对光照变化更鲁棒，Hue表示颜色，Saturation表示饱和度，Value表示亮度
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 3. 定义颜色范围
    # 红色在HSV色环中分布在0°和180°附近，因此需要两个区间
    red_ranges = [
        (np.array([0, 50, 50]), np.array([10, 255, 255])),  # 红色区间1
        (np.array([160, 50, 50]), np.array([180, 255, 255]))  # 红色区间2
    ]

    # 蓝色范围
    blue_ranges = [
        (np.array([100, 50, 50]), np.array([130, 255, 255]))  # 蓝色区间
    ]

    # 4. 创建颜色掩膜
    mask_red = create_color_mask(hsv, red_ranges)
    mask_blue = create_color_mask(hsv, blue_ranges)

    # 5. 保存原始掩膜（用于对比）
    cv2.imwrite(os.path.join(output_dir, 'mask_red_raw.jpg'), mask_red)
    cv2.imwrite(os.path.join(output_dir, 'mask_blue_raw.jpg'), mask_blue)

    # 6. 形态学操作（腐蚀+膨胀）
    mask_red_processed = morphological_operations(mask_red, kernel_size=5, erode_iter=1, dilate_iter=2)
    mask_blue_processed = morphological_operations(mask_blue, kernel_size=5, erode_iter=1, dilate_iter=2)

    # 7. 保存处理后的掩膜
    cv2.imwrite(os.path.join(output_dir, 'mask_red_processed.jpg'), mask_red_processed)
    cv2.imwrite(os.path.join(output_dir, 'mask_blue_processed.jpg'), mask_blue_processed)

    # 8. 轮廓处理与标注
    result = process_contours(img, mask_red_processed, mask_blue_processed, min_area=100)

    # 9. 保存最终结果
    result_path = os.path.join(output_dir, 'color_detection_result.jpg')
    cv2.imwrite(result_path, result)
    print(f"\n识别结果已保存至: {result_path}")

    # 10. 创建颜色提取效果图（在原图上显示颜色区域）
    # 红色区域提取
    red_extract = cv2.bitwise_and(img, img, mask=mask_red_processed)
    cv2.imwrite(os.path.join(output_dir, 'red_extract.jpg'), red_extract)

    # 蓝色区域提取
    blue_extract = cv2.bitwise_and(img, img, mask=mask_blue_processed)
    cv2.imwrite(os.path.join(output_dir, 'blue_extract.jpg'), blue_extract)

    return result


if __name__ == "__main__":
    # 测试图像路径（请根据实际情况修改）
    test_image = r"C:\Users\czh18\Desktop\lighting\lighting_backlit.jpg"

    # 执行颜色识别
    result = detect_colors(test_image)

    if result is not None:
        # 显示结果
        cv2.imshow('颜色识别结果', result)
        print("\n按任意键关闭窗口...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()