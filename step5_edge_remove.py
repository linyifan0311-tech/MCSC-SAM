import os
import numpy as np
import cv2
import rasterio
from rasterio.transform import Affine


def process_mask(mask_path, output_mask_path, shrink_ratio=0.05, crop_ratio=1.0):
    """处理蒙版：保留中心区域（不改变分辨率尺寸，外圈填黑），并使用几何缩放完美保留边缘形状进行内缩"""
    # 1. 读取并预处理蒙版图
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"警告：无法读取蒙版图 {mask_path}")
        return

    # 按比例处理中心区域，但不改变图像尺寸，外围用黑底填充
    if crop_ratio < 1.0:
        original_height, original_width = mask.shape[:2]
        new_width = int(original_width * crop_ratio)
        new_height = int(original_height * crop_ratio)

        # 计算保留区域（中心区域）
        start_x = (original_width - new_width) // 2
        start_y = (original_height - new_height) // 2
        end_x = start_x + new_width
        end_y = start_y + new_height

        # 创建全黑画布，仅保留中心区域的像素，保证分辨率不变
        mask_padded = np.zeros_like(mask)
        mask_padded[start_y:end_y, start_x:end_x] = mask[start_y:end_y, start_x:end_x]
        mask = mask_padded

    # 二值化
    _, binary_mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
    original_white_area = cv2.countNonZero(binary_mask)

    if original_white_area < 10:
        cv2.imwrite(output_mask_path, np.zeros_like(binary_mask))
        return

    contours, hierarchy = cv2.findContours(
        binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        cv2.imwrite(output_mask_path, np.zeros_like(binary_mask))
        return

    main_contour = max(contours, key=cv2.contourArea)

    # 使用几何缩放替代腐蚀，保留原始边缘形状
    # 计算轮廓中心
    M = cv2.moments(main_contour)
    if M['m00'] != 0:
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
    else:
        x, y, w, h = cv2.boundingRect(main_contour)
        cx = x + w // 2
        cy = y + h // 2

    # 按比例缩小顶点坐标
    scale = 1.0 - shrink_ratio
    scaled_contour = np.zeros_like(main_contour)
    for i in range(len(main_contour)):
        pt = main_contour[i][0]
        new_x = int(cx + (pt[0] - cx) * scale)
        new_y = int(cy + (pt[1] - cy) * scale)
        scaled_contour[i][0] = [new_x, new_y]

    # 绘制缩放后的轮廓
    contour_mask = np.zeros_like(binary_mask)
    cv2.drawContours(contour_mask, [scaled_contour], -1, 255, thickness=cv2.FILLED)

    # 取交集避免异常越界
    processed_mask = cv2.bitwise_and(binary_mask, contour_mask)

    # 连通域分析，只保留最大连通域
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        processed_mask, connectivity=8
    )
    if num_labels <= 1:
        cv2.imwrite(output_mask_path, np.zeros_like(processed_mask))
        return

    max_area = -1
    max_label = 0
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area > max_area:
            max_area = area
            max_label = i

    final_mask = np.zeros_like(processed_mask)
    final_mask[labels == max_label] = 255

    cv2.imwrite(output_mask_path, final_mask)


def apply_mask_to_ortho(ortho_path, mask_path, output_ortho_path, crop_ratio=1.0):
    """应用蒙版，按掩模置黑非目标区域，严格保持原分辨率与地理坐标不变"""
    # 读取蒙版
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"警告：无法读取蒙版 {mask_path}")
        return

    try:
        with rasterio.open(ortho_path) as src:
            # 获取原始地理信息，无需进行任何调整
            profile = src.profile.copy()
            ortho_data = src.read()  # (bands, height, width)

            # 1. 应用蒙版（直接对原始尺寸处理）
            mask_resized = cv2.resize(mask, (src.width, src.height), interpolation=cv2.INTER_NEAREST)
            mask_bool = mask_resized == 255

            # 移除所有的图像裁剪与坐标点偏移计算
            masked_data = ortho_data.copy()
            for i in range(masked_data.shape[0]):
                # 将非蒙版区域一律填充为 0 (黑色)
                masked_data[i] = np.where(mask_bool, masked_data[i], 0)

            # 写入输出文件（保留完整的原尺寸与原坐标系）
            with rasterio.open(output_ortho_path, 'w', **profile) as dst:
                dst.write(masked_data)

    except Exception as e:
        print(f"处理正射图 {ortho_path} 出错：{e}")
        return


def batch_process(mask_folder, ortho_folder, output_mask_folder, output_ortho_folder,
                  shrink_ratio=0.05, crop_ratio=1.0):
    """批量处理函数，保持所有的参数传递"""
    os.makedirs(output_mask_folder, exist_ok=True)
    os.makedirs(output_ortho_folder, exist_ok=True)

    mask_files = [f for f in os.listdir(mask_folder) if f.lower().endswith('.png')]
    if not mask_files:
        print(f"警告：蒙版文件夹 {mask_folder} 中无PNG文件")
        return

    for mask_file in mask_files:
        mask_path = os.path.join(mask_folder, mask_file)
        ortho_file = os.path.splitext(mask_file)[0] + '.tif'
        ortho_path = os.path.join(ortho_folder, ortho_file)
        output_mask_path = os.path.join(output_mask_folder, mask_file)
        output_ortho_path = os.path.join(output_ortho_folder, ortho_file)

        if not os.path.exists(ortho_path):
            print(f"跳过 {mask_file}：对应的正射图 {ortho_file} 不存在")
            continue

        print(f"正在处理：{mask_file} → 正射图：{ortho_file} (收缩: {shrink_ratio}, 裁剪效果: {crop_ratio})")
        process_mask(mask_path, output_mask_path, shrink_ratio, crop_ratio)
        apply_mask_to_ortho(ortho_path, output_mask_path, output_ortho_path, crop_ratio)
        print(f"完成处理：{output_mask_path} | {output_ortho_path}")

    print(f"\n所有文件处理完成！")
    print(f"- 处理后的蒙版：{output_mask_folder}")
    print(f"- 分割后的正射图：{output_ortho_folder}")


if __name__ == "__main__":
    # 修改为你的实际路径
    MASK_FOLDER = r"I:\baisha\2025.10.27\train\sam3_result\Ablation_Experiment\caijian\caijian_result\single\Kmeansdivide_3\masks"
    ORTHO_FOLDER = r"I:\baisha\2025.10.27\train\sam3_result\Ablation_Experiment\caijian\caijian_result\single\Kmeansdivide_3\tree_only"
    OUTPUT_MASK_FOLDER = r"I:\baisha\2025.10.27\train\sam3_result\Ablation_Experiment\caijian\caijian_result\single\edge_remove_and_connectivity_4\masks"
    OUTPUT_ORTHO_FOLDER = r"I:\baisha\2025.10.27\train\sam3_result\Ablation_Experiment\caijian\caijian_result\single\edge_remove_and_connectivity_4\orthos"

    # 收缩比例（0-1之间，控制边缘形状内缩）
    SHRINK_RATIO = 0.2

    # 裁剪比例（0-1之间，不改变输出分辨率，而是控制图像外围有多少比例被涂黑）
    CROP_RATIO = 0.9

    # 启动批量处理
    batch_process(
        mask_folder=MASK_FOLDER,
        ortho_folder=ORTHO_FOLDER,
        output_mask_folder=OUTPUT_MASK_FOLDER,
        output_ortho_folder=OUTPUT_ORTHO_FOLDER,
        shrink_ratio=SHRINK_RATIO,
        crop_ratio=CROP_RATIO
    )