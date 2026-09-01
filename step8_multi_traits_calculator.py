"""
OrchardQuant-3D 多冠层性状计算脚本（去噪优化版）
功能：批量处理预处理后的LAS点云文件夹，提取纯净去噪后的几何特征
优化内容：引入 SOR(统计滤波) 剔除离群噪点，防止单点噪声导致凸包体积和表面积异常膨胀
依赖库：laspy, numpy, scipy, pandas, os
"""

import laspy
import numpy as np
import pandas as pd
import os
from scipy.spatial import ConvexHull, cKDTree
from typing import List, Dict

def statistical_outlier_removal(points: np.ndarray, k: int = 20, std_ratio: float = 1.5) -> np.ndarray:
    """
    🌟 核心优化：统计滤波降噪 (SOR)
    计算每个点到它最近的 k 个点的平均距离。
    如果这个平均距离大于 (全局平均距离 + std_ratio * 标准差)，则判定为噪点（如天上的飞鸟、漂浮的噪点）并剔除。
    """
    if len(points) < k:
        return points

    # 使用 K-D 树加速近邻搜索
    tree = cKDTree(points)
    # 查找最近的 k+1 个点 (包含自身)
    distances, _ = tree.query(points, k=k+1)

    # 计算到其它 k 个邻居的平均距离（排除自身，即 distances[:, 0]）
    avg_distances = np.mean(distances[:, 1:], axis=1)

    mean_dist = np.mean(avg_distances)
    std_dist = np.std(avg_distances)

    # 设定阈值，剔除离群点
    threshold = mean_dist + std_ratio * std_dist
    valid_indices = np.where(avg_distances <= threshold)[0]

    return points[valid_indices]

def calculate_single_tree_multi_traits(las_file_path: str) -> Dict:
    """计算单棵树的多维几何性状（去噪优化版）"""
    try:
        las = laspy.file.File(las_file_path, mode="r")
        points = np.vstack([las.x, las.y, las.z]).T
        las.close()

        # 检查点云数量，构建三维凸包至少需要4个点
        if len(points) < 4:
            return {
                "las_filename": os.path.basename(las_file_path),
                "crown_proj_area_m2": np.nan,
                "crown_volume_m3": np.nan,
                "crown_surface_area_m2": np.nan,
                "crown_diameter_m": np.nan,
                "east_west_crown": np.nan,
                "north_south_crown": np.nan,
                "status": "点云太少(<4个点)"
            }

        # 1. 执行 SOR 滤波去噪，防止体积像气球一样被噪点撑大
        clean_points = statistical_outlier_removal(points, k=20, std_ratio=1.5)

        if len(clean_points) < 4:
            return {
                "las_filename": os.path.basename(las_file_path),
                "crown_proj_area_m2": np.nan,
                "crown_volume_m3": np.nan,
                "crown_surface_area_m2": np.nan,
                "crown_diameter_m": np.nan,
                "east_west_crown": np.nan,
                "north_south_crown": np.nan,
                "status": "滤波后有效点云太少"
            }

        traits = {"las_filename": os.path.basename(las_file_path)}

        # 2. 计算 3D 凸包 (获取树冠绝对体积与表面积)
        hull_3d = ConvexHull(clean_points)
        traits['crown_volume_m3'] = hull_3d.volume
        traits['crown_surface_area_m2'] = hull_3d.area

        # 3. 计算 2D 投影 (XY 平面，忽略高度 Z)
        xy_proj = clean_points[:, :2]
        hull_2d = ConvexHull(xy_proj)

        traits['crown_proj_area_m2'] = hull_2d.volume

        # 4. 计算衍生几何特征
        # 计算冠幅直径 (基于 2D 投影面积等效为一个正圆来近似计算直径)
        traits['crown_diameter_m'] = 2 * np.sqrt(traits['crown_proj_area_m2'] / np.pi)

        # 计算树冠长短轴 (沿 XY 坐标轴的极值跨度)
        traits['east_west_crown'] = np.max(xy_proj[:, 0]) - np.min(xy_proj[:, 0])
        traits['north_south_crown'] = np.max(xy_proj[:, 1]) - np.min(xy_proj[:, 1])

        traits["status"] = "成功"
        return traits

    except Exception as e:
        return {
            "las_filename": os.path.basename(las_file_path),
            "crown_proj_area_m2": np.nan,
            "crown_volume_m3": np.nan,
            "crown_surface_area_m2": np.nan,
            "crown_diameter_m": np.nan,
            "east_west_crown": np.nan,
            "north_south_crown": np.nan,
            "status": f"错误：{str(e)[:50]}"
        }

def batch_process_multi_traits(folder_path: str) -> List[Dict]:
    """批量处理文件夹内所有 LAS/LAZ 文件"""
    results = []

    # 检查文件夹是否存在
    if not os.path.exists(folder_path):
        print(f"❌ 找不到文件夹: {folder_path}")
        return results

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.las', '.laz')):
            file_path = os.path.join(folder_path, filename)
            traits = calculate_single_tree_multi_traits(file_path)
            results.append(traits)

            # 打印控制台日志，方便监控进度
            status_icon = "✅ 成功" if traits["status"] == "成功" else "❌ 失败"
            if traits["status"] == "成功":
                print(f"{status_icon} | {filename} | 体积：{traits['crown_volume_m3']:.2f} m³ | 投影面积：{traits['crown_proj_area_m2']:.2f} m²")
            else:
                print(f"{status_icon} | {filename} | 失败原因：{traits['status']}")

    return results

if __name__ == "__main__":
    # ================= 配置区域 =================
    INPUT_LAS_FOLDER = r"I:\pinghemiyou\qt\mask_process\step7_final_nonground_las"
    OUTPUT_CSV = r"I:\pinghemiyou\qt\mask_process\optimized_geometry_traits.csv"
    # ============================================

    print(f"🚀 开始批量提取去噪后的冠层几何特征...")
    print(f"📂 输入文件夹: {INPUT_LAS_FOLDER}")

    all_results = batch_process_multi_traits(INPUT_LAS_FOLDER)

    if all_results:
        df = pd.DataFrame(all_results)
        # 保存为 UTF-8 带有 BOM (utf-8-sig)，防止用 Excel 打开时中文乱码
        df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
        print(f"\n🎉 处理完成！")
        print(f"📊 结果已保存至：{os.path.abspath(OUTPUT_CSV)}")
    else:
        print("\n⚠️ 未处理任何文件或没有生成有效结果。")