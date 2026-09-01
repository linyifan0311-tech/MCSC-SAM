"""
OrchardQuant-3D 点云高空去噪脚本
功能：基于 DBSCAN 空间聚类去除无人机点云中的高空电线、飞点，并保存为干净的 LAS 文件。
输入：包含原始 LAS 文件的文件夹。
输出：去噪后的干净 LAS 文件（保留原文件头和所有附加属性）。
"""

import os
import laspy
import numpy as np
from sklearn.cluster import DBSCAN


def denoise_and_save_las(input_path: str, output_path: str, eps: float = 0.5, min_samples: int = 15):
    """
    对单个 LAS 文件进行去噪并保存
    """
    # 1. 读取原始点云
    in_las = laspy.file.File(input_path, mode="r")
    x, y, z = np.copy(in_las.x), np.copy(in_las.y), np.copy(in_las.z)
    total_points = len(z)

    if total_points == 0:
        in_las.close()
        raise ValueError("点云文件为空")

    # 2. 提取上半部分点云加速计算
    median_z = np.median(z)
    upper_mask = z > median_z  # 属于上半部的点
    lower_mask = ~upper_mask  # 属于下半部的点

    pts_upper = np.vstack((x[upper_mask], y[upper_mask], z[upper_mask])).T

    # 如果上半部分点极少，说明可能没有明显的树冠或噪点，跳过去噪
    if len(pts_upper) < 10:
        keep_mask = np.ones(total_points, dtype=bool)
    else:
        # 3. 对上半部进行 DBSCAN 聚类
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(pts_upper)

        unique_labels, counts = np.unique(labels, return_counts=True)
        valid_clusters = unique_labels[unique_labels != -1]  # 剔除噪点标签(-1)

        if len(valid_clusters) == 0:
            # 如果所有高处点都被判定为噪点（极端稀疏），全删掉
            upper_keep = np.zeros(len(pts_upper), dtype=bool)
        else:
            # 找到点数最多的主树冠簇
            cluster_counts = counts[unique_labels != -1]
            main_canopy_label = valid_clusters[np.argmax(cluster_counts)]

            # 只保留主树冠所在簇的点
            upper_keep = (labels == main_canopy_label)

        # 4. 合并保留掩码 (Mask)：完整的下半部 + 干净的上半部
        keep_mask = np.zeros(total_points, dtype=bool)
        keep_mask[lower_mask] = True
        keep_mask[upper_mask] = upper_keep

    # 5. 保存结果到新的 LAS 文件 (继承原始 Header 和数据结构)
    out_las = laspy.file.File(output_path, mode="w", header=in_las.header)
    out_las.points = in_las.points[keep_mask]

    out_las.close()
    in_las.close()

    removed_count = total_points - np.sum(keep_mask)
    return total_points, removed_count


def batch_denoise_folder(input_folder: str, output_folder: str, eps: float = 0.5):
    """
    批量处理文件夹
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    success_count = 0
    total_removed = 0

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.las', '.laz')):
            input_file = os.path.join(input_folder, filename)

            # 为了区分，可以在输出文件名前加个 clean_
            output_filename = f"{filename}"
            output_file = os.path.join(output_folder, output_filename)

            try:
                original_pts, removed_pts = denoise_and_save_las(input_file, output_file, eps=eps)
                success_count += 1
                total_removed += removed_pts
                print(f"✅ 完成: {filename} | 原始点数: {original_pts} | 剔除高空噪点: {removed_pts}")
            except Exception as e:
                print(f"❌ 失败: {filename} | 错误: {str(e)}")

    print("-" * 50)
    print(f"🎉 批量处理结束！共成功处理 {success_count} 个文件，累计剔除 {total_removed} 个高空噪点。")
    print(f"📁 干净点云已保存至：{os.path.abspath(output_folder)}")


if __name__ == "__main__":
    # 配置你的输入输出路径
    INPUT_DIR = r"I:\pinghemiyou\cailiao\train\MCSC_ph11\qt\mask_houchuli\step7_final_nonground_las"

    OUTPUT_DIR = r"I:\pinghemiyou\cailiao\train\MCSC_ph11\qt\cleaned_pointcloud_5"

    print(f"📂 开始扫描文件夹：{INPUT_DIR}")
    # 执行批量去噪
    batch_denoise_folder(INPUT_DIR, OUTPUT_DIR, eps=0.5)