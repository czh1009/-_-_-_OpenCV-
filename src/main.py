"""
OpenCV图像识别考核 - 主程序入口
整合所有模块，提供统一的运行接口
"""

import cv2
import numpy as np
import os
import sys
import argparse
import importlib.util
from pathlib import Path


# ============================ 导入各模块 ============================

# 动态导入模块，避免导入错误导致程序崩溃
def import_module(module_name, file_path):
    """动态导入Python模块"""
    try:
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    except Exception as e:
        print(f"[警告] 无法导入模块 {module_name}: {e}")
        return None


# 获取当前文件所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 导入各功能模块
basic_preprocessing = import_module("basic_preprocessing",
                                    os.path.join(current_dir, "basic_preprocessing.py"))
detect_shapes_digits = import_module("detect_shapes_digits",
                                     os.path.join(current_dir, "detect_shapes_digits.py"))
color_detection = import_module("color_detection",
                                os.path.join(current_dir, "color_detection.py"))


# ============================ 辅助函数 ============================



def print_menu():
    """打印功能菜单"""
    print("""
【功能菜单】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

基础必做任务:
  1. 图像基础预处理 (basic_preprocessing.py)
  2. 颜色阈值色块识别 (color_detection.py)
  3. 几何图形与数字识别 (detect_shapes_digits.py)

辅助功能:
  4. 调试模式 - 查看二值化结果
  5. 批量处理 - 处理文件夹内所有图片
  6. 生成测试报告
  7. 检查模块可用性

其他:
  0. 退出程序
  h. 显示帮助
  m. 显示本菜单

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    """)


def check_modules():
    """检查各模块是否成功导入"""
    print("\n【模块检查】")
    print("-" * 40)

    modules = [
        ("basic_preprocessing", basic_preprocessing, "图像预处理"),
        ("color_detection", color_detection, "颜色识别"),
        ("detect_shapes_digits", detect_shapes_digits, "形状数字识别")
    ]

    all_ok = True
    for module_name, module, desc in modules:
        if module:
            print(f" {desc:<12} ({module_name}.py) - 已加载")
        else:
            print(f" {desc:<12} ({module_name}.py) - 加载失败")
            all_ok = False

    print("-" * 40)
    if all_ok:
        print(" 所有模块加载成功！")
    else:
        print(" 部分模块加载失败，请检查文件是否存在")
    print()

    return all_ok


def check_image_path(image_path):
    """
    检查图像路径并给出建议

    参数:
        image_path: 用户输入的图像路径
    返回:
        str: 修正后的路径或None
    """
    if not image_path:
        return None

    # 移除引号和空格
    image_path = image_path.strip().strip('"').strip("'")

    # 替换反斜杠
    image_path = image_path.replace('\\', '/')

    # 检查文件是否存在
    if os.path.exists(image_path):
        return image_path

    # 尝试在当前目录查找
    basename = os.path.basename(image_path)
    local_path = basename
    if os.path.exists(local_path):
        return local_path

    # 尝试在images目录查找
    images_path = f"images/{basename}"
    if os.path.exists(images_path):
        return images_path

    images_path = f"images/raw/{basename}"
    if os.path.exists(images_path):
        return images_path

    return None


def get_image_path(prompt="请输入图像路径: "):
    """
    交互式获取图像路径

    参数:
        prompt: 提示信息
    返回:
        str: 有效的图像路径，或None
    """
    while True:
        user_input = input(prompt).strip()

        if not user_input:
            print("[提示] 输入为空，返回上级菜单")
            return None

        image_path = check_image_path(user_input)

        if image_path:
            print(f"[成功] 找到图像: {image_path}")
            return image_path
        else:
            print(f"[错误] 无法找到图像: {user_input}")
            print("[提示] 请检查路径是否正确")

            retry = input("是否重新输入？(y/n): ").strip().lower()
            if retry != 'y':
                return None


def batch_process(directory, function, output_dir="output"):
    """
    批量处理目录中的所有图像

    参数:
        directory: 输入目录
        function: 处理函数
        output_dir: 输出目录
    """
    if not os.path.exists(directory):
        print(f"[错误] 目录不存在: {directory}")
        return

    # 支持的图片格式
    extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_files = []

    for file in os.listdir(directory):
        file_path = os.path.join(directory, file)
        if os.path.isfile(file_path) and any(file.lower().endswith(ext) for ext in extensions):
            image_files.append(file_path)

    if not image_files:
        print(f"[警告] 目录中没有图片文件: {directory}")
        return

    print(f"\n找到 {len(image_files)} 张图片，开始批量处理...")

    successful = 0
    failed = 0

    for i, img_path in enumerate(image_files, 1):
        print(f"\n--- 处理第 {i}/{len(image_files)} 张: {os.path.basename(img_path)} ---")
        try:
            if function == "preprocess" and basic_preprocessing:
                result = basic_preprocessing.preprocess_image(img_path,
                                                              os.path.join(output_dir, "preprocess"))
                if result:
                    successful += 1
                else:
                    failed += 1

            elif function == "color" and color_detection:
                result = color_detection.detect_colors(img_path,
                                                       os.path.join(output_dir, "color"))
                if result is not None:
                    successful += 1
                else:
                    failed += 1

            elif function == "shape" and detect_shapes_digits:
                # 使用detect_shapes_digits.py的process_image函数
                if hasattr(detect_shapes_digits, 'process_image'):
                    result = detect_shapes_digits.process_image(img_path)
                    if result:
                        successful += 1
                    else:
                        failed += 1
                else:
                    print("[错误] process_image 函数不存在")
                    failed += 1
        except Exception as e:
            print(f"[错误] 处理失败: {e}")
            failed += 1

    print(f"\n 批量处理完成！")
    print(f"   成功: {successful} | 失败: {failed}")
    print(f"   结果保存在: {output_dir}")


def generate_report(output_dir="report"):
    """
    生成测试报告
    """
    print("\n【生成测试报告】")
    print("-" * 40)

    os.makedirs(output_dir, exist_ok=True)

    report_lines = []
    report_lines.append("# OpenCV图像识别考核 - 测试报告")
    report_lines.append(f"\n生成时间: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("\n## 1. 模块信息")

    # 模块信息
    if basic_preprocessing:
        report_lines.append("-  basic_preprocessing.py: 已加载")
    else:
        report_lines.append("-  basic_preprocessing.py: 未加载")

    if color_detection:
        report_lines.append("-  color_detection.py: 已加载")
    else:
        report_lines.append("-  color_detection.py: 未加载")

    if detect_shapes_digits:
        report_lines.append("-  detect_shapes_digits.py: 已加载")
    else:
        report_lines.append("-  detect_shapes_digits.py: 未加载")

    # 检查输出目录
    report_lines.append("\n## 2. 输出文件检查")

    output_dirs = [
        "images/task1_output",
        "images/task2_output",
        "images/task3_output",
        "output"
    ]

    for dir_path in output_dirs:
        if os.path.exists(dir_path):
            files = [f for f in os.listdir(dir_path) if f.endswith(('.jpg', '.png'))]
            report_lines.append(f"- 📁 {dir_path}: {len(files)} 个文件")
        else:
            report_lines.append(f"- 📁 {dir_path}: 目录不存在")

    # 保存报告
    report_path = os.path.join(output_dir, "test_report.md")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(report_lines))

    print(f" 报告已生成: {report_path}")
    print(report_path)


# ============================ 任务运行函数 ============================

def run_task1():
    """运行任务1：图像基础预处理"""
    print("\n" + "=" * 60)
    print("【任务1：图像基础预处理】")
    print("=" * 60)

    if not basic_preprocessing:
        print("[错误] basic_preprocessing模块未加载")
        return

    image_path = get_image_path("请输入图像路径 (直接回车返回): ")
    if not image_path:
        return

    # 执行预处理
    result = basic_preprocessing.preprocess_image(image_path)

    if result:
        print("\n 任务1执行成功！")
    else:
        print("\n 任务1执行失败！")


def run_task2():
    """运行任务2：颜色阈值色块识别"""
    print("\n" + "=" * 60)
    print("【任务2：颜色阈值色块识别】")
    print("=" * 60)

    if not color_detection:
        print("[错误] color_detection模块未加载")
        return

    image_path = get_image_path("请输入图像路径 (直接回车返回): ")
    if not image_path:
        return

    # 执行颜色识别
    result = color_detection.detect_colors(image_path)

    if result is not None:
        print("\n 任务2执行成功！")
    else:
        print("\n 任务2执行失败！")


def run_task3():
    """运行任务3：几何图形与数字识别"""
    print("\n" + "=" * 60)
    print("【任务3：几何图形与数字识别】")
    print("=" * 60)

    if not detect_shapes_digits:
        print("[错误] detect_shapes_digits模块未加载")
        return

    image_path = get_image_path("请输入图像路径 (直接回车返回): ")
    if not image_path:
        return

    # 检查模块中的函数
    if hasattr(detect_shapes_digits, 'process_image'):
        # 使用主处理函数
        result = detect_shapes_digits.process_image(image_path)
        print("\n 任务3执行完成！")
        print(f"   结果已保存到当前目录")
    elif hasattr(detect_shapes_digits, 'recognize_with_actual_templates'):
        # 使用实际模板识别函数
        result = detect_shapes_digits.recognize_with_actual_templates(image_path)
        print("\n 任务3执行完成！")
        print(f"   结果已保存到当前目录")
    else:
        print("[错误] 未找到可用的处理函数")
        print("   可用函数:")
        for name in dir(detect_shapes_digits):
            if not name.startswith('_'):
                print(f"     - {name}")


def run_debug_mode():
    """运行调试模式：查看二值化结果"""
    print("\n" + "=" * 60)
    print("【调试模式 - 查看二值化结果】")
    print("=" * 60)

    if not detect_shapes_digits:
        print("[错误] detect_shapes_digits模块未加载")
        return

    image_path = get_image_path("请输入图像路径 (直接回车返回): ")
    if not image_path:
        return

    # 检查是否有测试预处理函数
    if hasattr(detect_shapes_digits, 'test_preprocessing'):
        detect_shapes_digits.test_preprocessing(image_path)
        print("\n 调试完成！")
    elif hasattr(detect_shapes_digits, 'debug_digit_rois'):
        detect_shapes_digits.debug_digit_rois(image_path)
        print("\n 调试完成！")
    else:
        print("[提示] 没有专门的调试函数，直接运行主函数")
        if hasattr(detect_shapes_digits, 'process_image'):
            detect_shapes_digits.process_image(image_path)


# ============================ 交互模式 ============================

def interactive_mode():

    # 检查模块
    modules_ok = check_modules()
    if not modules_ok:
        print("[警告] 部分模块加载失败，可能会影响功能使用\n")

    print_menu()

    while True:
        choice = input("\n请输入选项 (0-7/h/m): ").strip().lower()

        if choice == '0':
            print("\n感谢使用，再见！")
            break

        elif choice == 'm':
            print_menu()

        elif choice == '1':
            run_task1()

        elif choice == '2':
            run_task2()

        elif choice == '3':
            run_task3()

        elif choice == '4':
            run_debug_mode()

        elif choice == '5':
            print("\n【批量处理】")
            print("-" * 40)
            print("选择处理类型:")
            print("  1. 批量预处理")
            print("  2. 批量颜色识别")
            print("  3. 批量形状识别")

            sub_choice = input("请输入选项 (1-3): ").strip()

            dir_path = input("请输入图片目录路径: ").strip()
            if not dir_path:
                continue

            if sub_choice == '1':
                batch_process(dir_path, "preprocess", "output/batch_preprocess")
            elif sub_choice == '2':
                batch_process(dir_path, "color", "output/batch_color")
            elif sub_choice == '3':
                batch_process(dir_path, "shape", "output/batch_shape")

        elif choice == '6':
            generate_report()

        elif choice == '7':
            check_modules()

        else:
            print("[提示] 无效选项，请输入 0-7/h/m")


# ============================ 命令行模式 ============================

def command_line_mode():
    """命令行模式"""
    parser = argparse.ArgumentParser(
        description='OpenCV图像识别考核 - 主程序',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python main.py                    # 交互模式
  python main.py task1 -i test.jpg  # 运行任务1
  python main.py task2 -i test.jpg  # 运行任务2
  python main.py task3 -i test.jpg  # 运行任务3
  python main.py --check             # 检查模块
  python main.py --report            # 生成报告
        """
    )

    parser.add_argument('task', nargs='?',
                        choices=['task1', 'task2', 'task3', 'debug'],
                        help='选择任务: task1/task2/task3/debug')
    parser.add_argument('-i', '--image',
                        help='输入图像路径')
    parser.add_argument('--check', action='store_true',
                        help='检查模块可用性')
    parser.add_argument('--report', action='store_true',
                        help='生成测试报告')

    args = parser.parse_args()

    # 检查模块
    if args.check:
        check_modules()
        return

    # 生成报告
    if args.report:
        generate_report()
        return

    # 运行指定任务
    if args.task:
        if args.task == 'task1':
            if args.image:
                if basic_preprocessing:
                    basic_preprocessing.preprocess_image(args.image)
                else:
                    print("[错误] 模块未加载")
            else:
                print("[错误] 请指定图像路径: -i IMAGE")

        elif args.task == 'task2':
            if args.image:
                if color_detection:
                    color_detection.detect_colors(args.image)
                else:
                    print("[错误] 模块未加载")
            else:
                print("[错误] 请指定图像路径: -i IMAGE")

        elif args.task == 'task3':
            if args.image:
                if detect_shapes_digits:
                    if hasattr(detect_shapes_digits, 'process_image'):
                        detect_shapes_digits.process_image(args.image)
                    else:
                        print("[错误] process_image 函数不存在")
                else:
                    print("[错误] 模块未加载")
            else:
                print("[错误] 请指定图像路径: -i IMAGE")

        elif args.task == 'debug':
            if args.image:
                if detect_shapes_digits:
                    if hasattr(detect_shapes_digits, 'test_preprocessing'):
                        detect_shapes_digits.test_preprocessing(args.image)
                    else:
                        print("[错误] test_preprocessing 函数不存在")
                else:
                    print("[错误] 模块未加载")
            else:
                print("[错误] 请指定图像路径: -i IMAGE")

        return

    # 没有参数，进入交互模式
    interactive_mode()


# ============================ 程序入口 ============================

if __name__ == "__main__":
    try:
        # 检查是否有命令行参数
        if len(sys.argv) > 1:
            command_line_mode()
        else:
            interactive_mode()
    except KeyboardInterrupt:
        print("\n\n[信息] 用户中断程序")
        sys.exit(0)
    except Exception as e:
        print(f"\n[错误] 程序异常: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)