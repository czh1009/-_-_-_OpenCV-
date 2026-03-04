import cv2
import numpy as np
import os

# ----------------------------- 配置 -----------------------------
# 输出文件名
OUT_SHAPES = "shape_detection.png"
OUT_DIGITS = "digit_detection.png"
OUT_COMBINED = "combined_result.png"
OUT_BINARY = "binary_result.png"  # 新增：保存二值化结果用于调试

# 模板生成参数
TEMPLATE_SIZE = (40, 60)  # 模板和待匹配ROI统一尺寸 (w, h)
FONT = cv2.FONT_HERSHEY_SIMPLEX


# ----------------------------- 辅助函数 -----------------------------

def load_image(path):
    """读取图像，返回彩色和灰度图"""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"图像未找到: {path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img, gray


def preprocess_for_contours(gray, blur_ksize=(5, 5)):
    """改进的预处理：使用自适应阈值分离数字"""
    # 1. 对比度增强
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # 2. 高斯模糊
    blur = cv2.GaussianBlur(enhanced, blur_ksize, 0)

    # 3. 使用自适应阈值（重要！）
    bw = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 15, 5)

    # 4. 形态学操作 - 先腐蚀再膨胀
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

    # 腐蚀：分离相连的笔画
    bw = cv2.erode(bw, kernel, iterations=1)

    # 膨胀：恢复大小
    bw = cv2.dilate(bw, kernel, iterations=1)

    # 5. 去除小噪点
    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 50:  # 去除小噪点
            cv2.drawContours(bw, [cnt], -1, 0, -1)

    return bw


def generate_digit_templates(font=FONT, size=TEMPLATE_SIZE):
    """
    生成0-9的模板图像 - 恢复版
    """
    templates = {}
    w, h = size

    for d in range(10):
        img = np.zeros((h, w), dtype=np.uint8)
        text = str(d)

        # 恢复所有数字都用文字
        if d == 1:
            font_scale = 2.2
            thickness = 4
        elif d == 4:
            font_scale = 2.0
            thickness = 3
        elif d == 7:
            font_scale = 2.0
            thickness = 3
        else:
            font_scale = 1.8
            thickness = 3

        (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
        org = ((w - tw) // 2, (h + th) // 2)
        cv2.putText(img, text, org, font, font_scale, (255,), thickness, cv2.LINE_AA)

        # 轻微膨胀使数字更粗
        kernel = np.ones((2, 2), np.uint8)
        img = cv2.dilate(img, kernel, iterations=1)

        templates[d] = img
        cv2.imwrite(f"template_{d}.png", img)

    print("已生成恢复后的模板")
    return templates
# ----------------------------- 形状检测 -----------------------------

def detect_rectangles_and_circles(src_img, gray):
    """
    改进的形状检测函数：
    1. 增加圆形检测的严格性
    2. 过滤掉过小的圆形
    3. 添加面积比例过滤
    """
    vis = src_img.copy()
    bw = preprocess_for_contours(gray)

    # 保存二值化结果用于调试
    cv2.imwrite(OUT_BINARY, bw)
    print(f"已保存二值化结果: {OUT_BINARY}")

    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rectangles = []
    circles = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 150:  # 提高面积阈值，减少噪点
            continue

        # 轮廓近似
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        # 矩形检测
        if len(approx) == 4 and cv2.isContourConvex(approx):
            # 计算角度
            def angle(pt1, pt2, pt0):
                dx1 = pt1[0] - pt0[0]
                dy1 = pt1[1] - pt0[1]
                dx2 = pt2[0] - pt0[0]
                dy2 = pt2[1] - pt0[1]
                num = dx1 * dx2 + dy1 * dy2
                den = np.sqrt((dx1 * dx1 + dy1 * dy1) * (dx2 * dx2 + dy2 * dy2) + 1e-8)
                return abs(np.degrees(np.arccos(np.clip(num / den, -1.0, 1.0))))

            pts = approx.reshape(-1, 2)
            angles = []
            for i in range(4):
                angles.append(angle(pts[(i - 1) % 4], pts[(i + 1) % 4], pts[i]))

            # 矩形角度判断
            if np.mean(angles) > 75 and np.mean(angles) < 105:  # 更严格的矩形判断
                rect = cv2.minAreaRect(cnt)
                box = cv2.boxPoints(rect).astype(int)
                M = cv2.moments(cnt)
                if M['m00'] != 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                else:
                    cx, cy = 0, 0

                rectangles.append({
                    "contour": cnt,
                    "box_pts": box,
                    "area": area,
                    "center": (cx, cy),
                    "angle": rect[2]
                })
                cv2.drawContours(vis, [box], 0, (0, 255, 0), 2)
                cv2.putText(vis, f"Rect", (cx - 30, cy - 10), FONT, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                continue

        # 圆形检测 - 更严格的判断
        perimeter = cv2.arcLength(cnt, True)
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter * perimeter)

            # 获取外接矩形，检查是否接近正方形（圆的外接矩形应该是正方形）
            x, y, w, h = cv2.boundingRect(cnt)
            rect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 0

            # 圆形判断条件：圆度 > 0.8 且 外接矩形长宽比 < 1.2
            if circularity > 0.8 and rect_ratio < 1.2 and area > 200:
                (cx, cy), radius = cv2.minEnclosingCircle(cnt)

                # 进一步验证：轮廓面积与最小外接圆面积的比例
                circle_area = np.pi * radius * radius
                area_ratio = area / circle_area

                if area_ratio > 0.6:  # 轮廓应占圆面积的60%以上
                    circles.append({
                        "center": (int(cx), int(cy)),
                        "radius": int(radius),
                        "contour": cnt,
                        "area": area
                    })
                    cv2.circle(vis, (int(cx), int(cy)), int(radius), (255, 0, 0), 2)
                    cv2.putText(vis, f"Circle", (int(cx) - 30, int(cy) - 10), FONT, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

    return vis, rectangles, circles, bw


# ----------------------------- 数字识别（模板匹配） -----------------------------

def extract_digit_candidates(bw):
    """
    提取数字候选区域 - 宽松版本
    """
    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rois = []

    # 过滤掉太大的轮廓（可能是整个图像）
    image_area = bw.shape[0] * bw.shape[1]

    for i, cnt in enumerate(contours):
        x, y, w, h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)

        # 排除整个图像的轮廓
        if area > image_area * 0.3:
            continue

        # 放宽条件
        if area < 20:  # 更小的面积
            continue

        if w < 4 or h < 6:  # 更小的尺寸
            continue

        # 长宽比 - 几乎不加限制
        aspect = w / float(h)
        if aspect > 3.0:  # 很宽松
            continue

        # 填充率
        rect_area = w * h
        fill_ratio = area / rect_area
        if fill_ratio < 0.1 or fill_ratio > 0.95:
            continue

        rois.append((x, y, w, h, cnt))

    # 合并距离太近的ROI
    merged_rois = []
    used = [False] * len(rois)

    for i in range(len(rois)):
        if used[i]:
            continue

        x1, y1, w1, h1, _ = rois[i]
        group = [rois[i]]
        used[i] = True

        for j in range(i + 1, len(rois)):
            if used[j]:
                continue

            x2, y2, w2, h2, _ = rois[j]

            # 计算中心距离
            center1 = (x1 + w1 // 2, y1 + h1 // 2)
            center2 = (x2 + w2 // 2, y2 + h2 // 2)
            distance = np.sqrt((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2)

            if distance < 15:  # 距离阈值
                group.append(rois[j])
                used[j] = True

        if len(group) > 1:
            # 合并
            min_x = min(r[0] for r in group)
            min_y = min(r[1] for r in group)
            max_x = max(r[0] + r[2] for r in group)
            max_y = max(r[1] + r[3] for r in group)
            merged_rois.append((min_x, min_y, max_x - min_x, max_y - min_y, None))
        else:
            merged_rois.append(group[0])

    merged_rois.sort(key=lambda r: r[0])
    print(f"合并后共 {len(merged_rois)} 个候选区域")
    return merged_rois


def recognize_digits_by_template(src_img, bw, templates, template_size=TEMPLATE_SIZE):
    """
    使用模板匹配识别数字 - 恢复版
    """
    vis = src_img.copy()
    results = []
    rois = extract_digit_candidates(bw)

    print(f"\n处理 {len(rois)} 个候选区域")

    for i, (x, y, w, h, cnt) in enumerate(rois):
        roi = bw[y:y + h, x:x + w]

        if roi.size == 0:
            continue

        cv2.imwrite(f"roi_{i}.png", roi)

        best_score = -1
        best_digit = None
        scores = {}

        # 尝试多种缩放
        for scale in [0.9, 1.0, 1.1]:
            try:
                new_w = int(template_size[0] * scale)
                new_h = int(template_size[1] * scale)
                roi_scaled = cv2.resize(roi, (new_w, new_h))
                roi_final = cv2.resize(roi_scaled, template_size)
            except:
                continue

            for d, timg in templates.items():
                res = cv2.matchTemplate(roi_final, timg, cv2.TM_CCOEFF_NORMED)
                score = res.max()
                scores[d] = max(scores.get(d, -1), score)

                if score > best_score:
                    best_score = score
                    best_digit = d

        if best_score > 0.2:  # 保持0.2的阈值
            results.append((best_digit, (x, y, w, h), float(best_score)))
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(vis, f"{best_digit}:{best_score:.2f}",
                        (x, y - 6), FONT, 0.5, (0, 0, 255), 1, cv2.LINE_AA)


    # 按x坐标排序
    results.sort(key=lambda r: r[1][0])

    if results:
        digits_str = "".join([str(r[0]) for r in results])

    print(f"\n总共识别到 {len(results)} 个数字")
    return vis, results

# ----------------------------- 主流程 -----------------------------
def process_image(image_path):
    """处理图像的主函数"""
    src, gray = load_image(image_path)

    # 保存原始灰度图用于调试
    cv2.imwrite("original_gray.png", gray)
    print("已保存原始灰度图: original_gray.png")

    templates = generate_digit_templates()

    # 形状检测
    vis_shapes, rects, circs, bw = detect_rectangles_and_circles(src, gray)
    cv2.imwrite(OUT_SHAPES, vis_shapes)
    print(f"检测到矩形数量: {len(rects)}, 圆形数量: {len(circs)}")

    # 数字检测
    vis_digits, digit_results = recognize_digits_by_template(src, bw, templates)
    cv2.imwrite(OUT_DIGITS, vis_digits)
    print(f"数字识别结果: {digit_results}")

    # 合并显示
    combined = vis_shapes.copy()
    combined = cv2.addWeighted(combined, 0.7, vis_digits, 0.3, 0)
    cv2.imwrite(OUT_COMBINED, combined)
    print(f"已保存合并结果: {OUT_COMBINED}")

    return rects, circs, digit_results


# ----------------------------- 命令行运行 -----------------------------
# ----------------------------- 测试预处理函数 -----------------------------
def test_preprocessing(image_path):
    """专门测试预处理的函数"""
    # 读取图像
    img = cv2.imread(image_path)
    if img is None:
        print(f"无法读取图像: {image_path}")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print(f"图像大小: {gray.shape}")

    # 1. 保存原始灰度图
    cv2.imwrite("test_original_gray.png", gray)
    print("已保存: test_original_gray.png")

    # 2. 自适应阈值
    bw_adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY_INV, 15, 5)
    cv2.imwrite("test_bw_adaptive.png", bw_adaptive)
    print("已保存: test_bw_adaptive.png")

    # 3. OTSU阈值
    _, bw_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cv2.imwrite("test_bw_otsu.png", bw_otsu)
    print("已保存: test_bw_otsu.png")

    # 4. 统计轮廓数量
    contours_adaptive, _ = cv2.findContours(bw_adaptive, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_otsu, _ = cv2.findContours(bw_otsu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    print(f"自适应阈值找到 {len(contours_adaptive)} 个轮廓")
    print(f"OTSU阈值找到 {len(contours_otsu)} 个轮廓")

    return bw_adaptive, bw_otsu


# ----------------------------- 命令行运行 -----------------------------
# ----------------------------- 调试函数 -----------------------------
def debug_digit_rois(image_path):
    """调试函数：查看实际的数字ROI"""
    import cv2
    import numpy as np

    # 读取图像
    img = cv2.imread(image_path)
    if img is None:
        print(f"无法读取图像: {image_path}")
        return []

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 使用自适应阈值
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 15, 5)

    # 找到轮廓
    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    print(f"找到 {len(contours)} 个轮廓")

    # 筛选数字区域（下排的小轮廓）
    digit_rois = []
    for i, cnt in enumerate(contours):
        x, y, w, h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)

        # 筛选下排的数字（y坐标较大，面积适中）
        if y > 400 and area > 100 and area < 1500:
            digit_rois.append((x, y, w, h, area))
            # 保存每个数字的ROI
            roi = bw[y:y + h, x:x + w]
            cv2.imwrite(f"actual_digit_{i:02d}.png", roi)

    # 按x坐标排序
    digit_rois.sort(key=lambda r: r[0])

    print(f"\n找到 {len(digit_rois)} 个数字区域")

    # 按顺序显示数字
    if digit_rois:
        print("\n按顺序的数字区域:")
        for j, (x, y, w, h, area) in enumerate(digit_rois):
            print(f"  位置{j}: x={x}, 应该是数字 {j}")

    return digit_rois


# ----------------------------- 命令行运行 -----------------------------
# ----------------------------- 基于实际模板的数字识别 -----------------------------
def recognize_with_actual_templates(image_path):
    """主函数：使用实际模板识别数字"""
    # 1. 读取图像
    src, gray = load_image(image_path)

    # 2. 二值化
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 15, 5)
    cv2.imwrite("debug_binary.png", bw)

    # 3. 获取实际数字图片作为模板

    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 收集所有数字区域
    digit_rois = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        if y > 400 and area > 100 and area < 1500:
            digit_rois.append((x, y, w, h))

    digit_rois.sort(key=lambda r: r[0])

    # 保存每个数字作为模板
    templates = {}
    template_files = []

    for i, (x, y, w, h) in enumerate(digit_rois):
        roi = bw[y:y + h, x:x + w]
        # 缩放到统一大小
        roi_resized = cv2.resize(roi, (40, 60))
        templates[i] = roi_resized
        filename = f"template_digit_{i}.png"
        cv2.imwrite(filename, roi_resized)
        template_files.append(filename)

    # 4. 使用这些模板重新识别
    vis = src.copy()
    results = []

    for i, (x, y, w, h) in enumerate(digit_rois):
        roi = bw[y:y + h, x:x + w]
        roi_resized = cv2.resize(roi, (40, 60))

        best_score = -1
        best_digit = None

        for d, timg in templates.items():
            res = cv2.matchTemplate(roi_resized, timg, cv2.TM_CCOEFF_NORMED)
            score = res.max()
            if score > best_score:
                best_score = score
                best_digit = d



        if best_score > 0.5:  # 提高阈值
            results.append((best_digit, (x, y, w, h)))
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(vis, str(best_digit), (x + 5, y + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # 5. 输出结果
    if results:
        digits_str = "".join([str(r[0]) for r in results])
        print(f"\n识别结果: {digits_str}")
    else:
        print("\n未识别到任何数字")

    cv2.imwrite("final_result.png", vis)
    print("最终结果已保存: final_result.png")

    return results


# ----------------------------- 命令行运行 -----------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="使用实际模板识别数字")
    parser.add_argument("image", help="待处理图像路径")
    args = parser.parse_args()

    results = recognize_with_actual_templates(args.image)
    print("\n完成！")