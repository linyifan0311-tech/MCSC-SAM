"""
OrchardQuant-3D 最大树高计算脚本
功能：批量处理LAS点云文件夹，计算每棵树的最大树高（Heightₘₐₓ）
输入：预处理后的LAS点云文件夹路径（已去除地面点）
输出：最大树高结果表格（max_height_results.csv）
依赖库：laspy、numpy、pandas、os
"""

import laspy
import numpy as np
import pandas as pd
import os
from typing import List, Dict


def calculate_single_tree_max_height(las_file_path: str) -> float:
    """
    计算单个LAS文件的最大树高（适用于含地面点的点云）
    逻辑：地面点取z轴最小值（或底部5%点的平均值），作为基准
    """
    las = laspy.file.File(las_file_path, mode="r")
    z_coords = np.copy(las.z)  # 提取 z 坐标
    las.close()

    if len(z_coords) == 0:
        raise ValueError("点云文件为空")

    # 地面高程：取z轴最小的5%点的平均值（过滤地面噪声）
    ground_ratio = 0.05  # 底部5%视为地面点
    sorted_z = np.sort(z_coords)
    ground_idx = max(1, int(len(sorted_z) * ground_ratio))
    ground_z = np.mean(sorted_z[:ground_idx])  # 地面基准高程

    # 树冠顶端高程：z轴最大值
    canopy_top_z = np.max(z_coords)

    # 最大树高 = 顶端 - 地面
    max_height = canopy_top_z - ground_z
    return round(max_height, 3)


def batch_process_las_folder(folder_path: str) -> List[Dict]:
    """
    批量处理文件夹内所有LAS文件
    """
    results = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.las', '.laz')):
            file_path = os.path.join(folder_path, filename)
            try:
                max_height = calculate_single_tree_max_height(file_path)
                results.append({
                    "las_filename": filename,
                    "max_height_m": max_height
                })
                print(f"✅ 处理完成：{filename} | 最大树高：{max_height} m")
            except Exception as e:
                results.append({
                    "las_filename": filename,
                    "max_height_m": f"处理失败：{str(e)[:50]}"
                })
                print(f"❌ 处理失败：{filename} | 错误：{str(e)}")
    return results


def save_results_to_csv(results: List[Dict], output_csv_path: str = "max_height_results.csv"):
    """将结果保存为CSV表格"""
    df = pd.DataFrame(results)
    df.to_csv(output_csv_path, index=False, encoding="utf-8-sig")
    print(f"\n📊 结果已保存至：{os.path.abspath(output_csv_path)}")


if __name__ == "__main__":
    INPUT_LAS_FOLDER = r"I:\pinghemiyou\cailiao\train\MCSC_ph11\qt\cleaned_pointcloud_5"
    OUTPUT_CSV = r"I:\pinghemiyou\cailiao\train\MCSC_ph11\qt\max_height_results.csv"

    if not os.path.exists(INPUT_LAS_FOLDER):
        print(f"❌ 错误：输入文件夹不存在 → {INPUT_LAS_FOLDER}")
    else:
        print(f"📂 开始处理文件夹：{os.path.abspath(INPUT_LAS_FOLDER)}")
        batch_results = batch_process_las_folder(INPUT_LAS_FOLDER)
        save_results_to_csv(batch_results, OUTPUT_CSV)
        print("🎉 所有文件处理完成！")