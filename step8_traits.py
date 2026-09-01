import os
import cv2
import numpy as np
import laspy
import pyvista as pv
import pandas as pd
from osgeo import gdal
from skimage.feature import graycomatrix, graycoprops
from scipy.spatial import ConvexHull


def extract_ortho_features_optimized(ortho_path):
    """从单棵树正射图提取纯植被的纹理特征，剔除黑边干扰"""
    features = {}

    ds = gdal.Open(ortho_path)
    if ds is None:
        return {"texture_contrast": np.nan, "texture_correlation": np.nan, "texture_energy": np.nan,
                "texture_entropy": np.nan}

    R = ds.GetRasterBand(1).ReadAsArray().astype(float)
    G = ds.GetRasterBand(2).ReadAsArray().astype(float)
    B = ds.GetRasterBand(3).ReadAsArray().astype(float)

    # 1. 识别非黑背景（有效树冠区域）
    mask = ~((R <= 1e-6) & (G <= 1e-6) & (B <= 1e-6))
    if not np.any(mask):
        return {"texture_contrast": np.nan, "texture_correlation": np.nan, "texture_energy": np.nan,
                "texture_entropy": np.nan}

    # 2. 转换为灰度图
    gray = (0.299 * R + 0.587 * G + 0.114 * B)

    # 3. 量化灰度等级 (将有效像素量化到 1-255，背景严格保留为 0)
    gray_valid = gray[mask]
    min_val, max_val = gray_valid.min(), gray_valid.max()

    gray_quantized = np.zeros_like(gray, dtype=np.uint8)  # 背景为0
    if max_val > min_val:
        # 将有效像素线性拉伸到 1-255
        gray_quantized[mask] = np.digitize(gray[mask], np.linspace(min_val, max_val, 255))
    else:
        gray_quantized[mask] = 1

    # 4. 计算 GLCM (级别设为256)
    glcm = graycomatrix(gray_quantized, distances=[1], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], levels=256,
                        symmetric=True, normed=False)

    # 把涉及 0 (背景) 的行和列全部清零，这样 GLCM 中就只包含 "树叶-树叶" 的相邻关系
    glcm[0, :, :, :] = 0
    glcm[:, 0, :, :] = 0

    # 重新归一化
    glcm_norm = glcm.astype(float)
    sums = np.sum(glcm_norm, axis=(0, 1), keepdims=True)
    sums[sums == 0] = 1  # 避免除以0
    glcm_norm /= sums

    # 5. 计算真正的纯植被纹理特征
    features['texture_contrast'] = graycoprops(glcm, 'contrast').mean()
    features['texture_correlation'] = graycoprops(glcm, 'correlation').mean()
    features['texture_energy'] = graycoprops(glcm, 'energy').mean()

    entropy = -np.sum(glcm_norm * np.log2(glcm_norm + 1e-16), axis=(0, 1)).mean()
    features['texture_entropy'] = entropy

    return features


def extract_las_features(las_path):
    """从单棵树点云提取三维特征（兼容低版本pyvista）"""
    features = {}

    # 1. 读取点云
    try:
        las = laspy.file.File(las_path, mode='r')
        x = las.x.copy()
        y = las.y.copy()
        z = las.z.copy()
        las.close()
    except Exception as e:
        raise ValueError(f"无法读取点云：{str(e)}")

    points = np.vstack((x, y, z)).transpose()
    if len(points) < 4:
        return {k: np.nan for k in ["tree_height", "east_west_crown", "north_south_crown",
                                    "crown_volume", "leaf_density", "lai"]}

    # 2. 基本三维参数
    features["tree_height"] = np.max(z) - np.min(z)
    features["east_west_crown"] = np.max(y) - np.min(y)
    features["north_south_crown"] = np.max(x) - np.min(x)

    # 3. 冠层体积（凸包法）
    try:
        hull = ConvexHull(points)
        features["crown_volume"] = hull.volume
    except:
        features["crown_volume"] = np.nan

    # 4. 叶片密度（替换compute_point_density，用KDTree计算点密度）
    from scipy.spatial import KDTree

    # 计算每个点周围半径r范围内的点数（替代点密度）
    r = 0.3  # 搜索半径（米，根据点云分辨率调整，柑橘树建议0.2-0.5）
    tree = KDTree(points[:, :2])  # 仅用XY平面坐标计算密度
    densities = []
    for p in points[:, :2]:
        # 统计半径r内的点数（包括自身）
        count = len(tree.query_ball_point(p, r))
        densities.append(count / (np.pi * r ** 2))  # 密度=点数/圆面积（点/㎡）
    densities = np.array(densities)

    # 筛选叶片点（密度高于阈值）
    leaf_threshold = 5  # 叶片点密度阈值
    leaf_mask = densities > leaf_threshold
    leaf_points = points[leaf_mask]
    features["leaf_density"] = len(leaf_points) / (features["crown_volume"] + 1e-8)

    # 5. LAI估算
    try:
        leaf_cloud = pv.PolyData(leaf_points)
        # 增加点数量判断，避免少量点重建失败
        if len(leaf_cloud.points) < 10:
            features["lai"] = np.nan
        else:
            # 仅适用于接近平面的冠层点云
            surf = leaf_cloud.delaunay_2d()  # 2D三角化重建表面
            # 或用convex_hull（凸包表面，适用于密集点云）
            # surf = leaf_cloud.convex_hull()

            # 过滤无效表面
            if surf.area < 0.1:
                features["lai"] = np.nan
            else:
                features["lai"] = surf.area * 0.5  # 保留经验系数
    except Exception as e:
        print(f"LAI计算失败：{str(e)}")
        features["lai"] = np.nan

    return features


def batch_process(ortho_folder, las_folder, output_csv):
    """批量处理所有树，保存结果到CSV（保持不变）"""
    # 获取所有正射图文件名
    ortho_files = [f for f in os.listdir(ortho_folder) if f.endswith((".tif", ".TIF"))]
    all_features = []

    for ortho_file in ortho_files:
        tree_id = os.path.splitext(ortho_file)[0]  # 树ID（文件名）
        print(f"处理树：{tree_id}...")

        # 拼接文件路径
        ortho_path = os.path.join(ortho_folder, ortho_file)
        las_path = os.path.join(las_folder, f"{tree_id}.las")
        if not os.path.exists(las_path):
            print(f"警告：点云文件不存在 {las_path}，跳过该树")
            continue

        # 提取特征
        try:
            ortho_feats = extract_ortho_features_optimized(ortho_path)
            las_feats = extract_las_features(las_path)
        except Exception as e:
            print(f"处理失败 {tree_id}：{str(e)}，跳过")
            continue

        # 合并特征，添加树ID
        all_feats = {"tree_id": tree_id}
        all_feats.update(ortho_feats)
        all_feats.update(las_feats)
        all_features.append(all_feats)

    # 保存到CSV
    df = pd.DataFrame(all_features)
    df.to_csv(output_csv, index=False)
    print(f"所有结果已保存到：{output_csv}")


if __name__ == "__main__":
    # 配置路径
    ortho_folder = r"I:\pinghemiyou\qt\mask_process\step5_final_orthos"  # 正射图文件夹
    las_folder = r"I:\pinghemiyou\qt\mask_process\step7_final_nonground_las"  # 点云文件夹
    output_csv = r"I:\pinghemiyou\qt\mask_process\citrus_traits_results.csv"  # 输出CSV路径

    # 运行批量处理
    batch_process(ortho_folder, las_folder, output_csv)