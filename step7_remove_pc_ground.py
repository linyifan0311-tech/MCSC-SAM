import laspy
import numpy as np

def remove_ground_points(input_las_path, output_las_path, method='auto_threshold',
                         auto_percentile=5, ransac_max_iter=1000, ransac_inlier_dist=0.5):
    """
    从LAS点云文件中去除地面点（兼容laspy 1.7.0），增加空数据容错处理
    """
    # 读取LAS文件（兼容laspy 1.7.0）
    las = laspy.file.File(input_las_path, mode='r')
    points = np.vstack((las.x, las.y, las.z)).transpose()
    n_total = len(points)
    print(f"读取 {input_las_path} 完成，共 {n_total} 个点")

    if n_total == 0:
        print("文件为空，跳过处理\n")
        las.close()
        return

    # 计算掩码
    non_ground_mask = None
    if method == 'auto_threshold':
        z_values = points[:, 2]
        # 计算分位数时忽略极端值（避免因少数高点导致阈值过高）
        z_clean = z_values[(z_values >= np.percentile(z_values, 1)) & (z_values <= np.percentile(z_values, 99))]
        if len(z_clean) == 0:
            z_clean = z_values  # 极端情况处理
        ground_threshold = np.percentile(z_clean, auto_percentile)
        offset = 0.2  # 减小偏移量，避免误删所有点
        non_ground_mask = z_values > (ground_threshold + offset)
        print(f"自动计算阈值：{ground_threshold + offset}（基于第{auto_percentile}百分位高程）")

    elif method == 'ransac':
        non_ground_mask = ransac_ground_filter(points, ransac_max_iter, ransac_inlier_dist)

    else:
        las.close()
        raise ValueError("方法仅支持 'auto_threshold' 或 'ransac'")

    # 检查是否有保留的点
    n_remaining = np.sum(non_ground_mask)
    if n_remaining == 0:
        print("警告：所有点均被判定为地面点，未保存文件\n")
        las.close()
        return

    # 保存结果
    out_las = laspy.file.File(output_las_path, mode='w', header=las.header)
    out_las.x = las.x[non_ground_mask]
    out_las.y = las.y[non_ground_mask]
    out_las.z = las.z[non_ground_mask]
    for dim in las.point_format:
        if dim.name not in ['x', 'y', 'z']:
            setattr(out_las, dim.name, getattr(las, dim.name)[non_ground_mask])
    las.close()
    out_las.close()

    print(f"处理完成，保留 {n_remaining} 个点，保存至 {output_las_path}\n")


def ransac_ground_filter(points, max_iter, inlier_dist):
    """RANSAC平面拟合地面滤波"""
    best_inliers = 0
    best_inlier_mask = np.zeros(len(points), dtype=bool)
    n_points = len(points)
    if n_points < 3:
        return np.ones(n_points, dtype=bool)

    for _ in range(max_iter):
        sample_idx = np.random.choice(n_points, 3, replace=False)
        p1, p2, p3 = points[sample_idx]
        v1 = p2 - p1
        v2 = p3 - p1
        normal = np.cross(v1, v2)
        if np.linalg.norm(normal) < 1e-6:
            continue
        a, b, c = normal
        d = - (a * p1[0] + b * p1[1] + c * p1[2])
        distances = np.abs(a * points[:,0] + b * points[:,1] + c * points[:,2] + d) / np.linalg.norm(normal)
        current_inliers = distances < inlier_dist
        current_count = np.sum(current_inliers)
        if current_count > best_inliers:
            best_inliers = current_count
            best_inlier_mask = current_inliers

    return ~best_inlier_mask


# 批量处理示例
if __name__ == "__main__":
    import os

    input_dir = r"H:\baisha\2025.5.09\cailiao\Visible\caijianpointcloud_5"    # 输入文件夹
    output_dir = r"H:\baisha\2025.5.09\cailiao\Visible\non_ground_las_6"
    os.makedirs(output_dir, exist_ok=True)

    las_files = [f for f in os.listdir(input_dir) if f.endswith(".las")]

    for las_file in las_files:
        input_path = os.path.join(input_dir, las_file)
        output_path = os.path.join(output_dir, las_file)
        # 尝试降低分位数（如3），让阈值更低，保留更多点
        remove_ground_points(
            input_path,
            output_path,
            method='auto_threshold',
            auto_percentile=20  # 降低分位数，阈值更接近地面
        )