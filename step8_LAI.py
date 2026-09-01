import os
import numpy as np
import pandas as pd
from scipy import spatial
from tqdm import tqdm


def process_point_clouds_and_calculate_lai(input_dir, output_csv):
    """
    处理文件夹中的点云文件并计算每棵树的叶面积指数(LAI)

    参数:
        input_dir: 包含点云文件的文件夹路径（支持LAS和TXT格式）
        output_csv: 保存LAI结果的CSV文件路径
    """
    # 获取文件夹中所有点云文件
    point_cloud_files = []
    for file in os.listdir(input_dir):
        if file.lower().endswith(('.las', '.txt')):
            point_cloud_files.append(os.path.join(input_dir, file))

    if not point_cloud_files:
        print("未找到任何点云文件（支持LAS和TXT格式）")
        return

    lai_results = []
    filenames = []

    # 处理每个点云文件
    for file_path in tqdm(point_cloud_files, desc="处理点云文件"):
        filename = os.path.basename(file_path)
        filenames.append(filename)

        try:
            # 读取点云数据
            if file_path.lower().endswith('.las'):
                import laspy
                from laspy import file
                with file.File(file_path, mode='r') as f_las:
                    # 筛选高度在第1百分位以上的点（去除低矮噪声）
                    z_values = f_las.Z
                    threshold = np.percentile(z_values, 1)
                    mask = z_values >= threshold

                    x = f_las.X[mask]
                    y = f_las.Y[mask]
                    z = f_las.Z[mask]

                    # 如果有RGB数据则归一化（不影响LAI计算，仅为完整处理）
                    if hasattr(f_las, 'Red'):
                        r = f_las.Red[mask]
                        g = f_las.Green[mask]
                        b = f_las.Blue[mask]
                        r = ((r - r.min()) / (r.max() - r.min() + 1e-8)) * 255
                        g = ((g - g.min()) / (g.max() - g.min() + 1e-8)) * 255
                        b = ((b - b.min()) / (b.max() - b.min() + 1e-8)) * 255

            elif file_path.lower().endswith('.txt'):
                # 假设TXT文件格式: X Y Z [R G B]，空格分隔
                data = np.loadtxt(file_path, usecols=(0, 1, 2))  # 只读取XYZ
                x, y, z = data[:, 0], data[:, 1], data[:, 2]

                # 同样筛选高度在第1百分位以上的点
                threshold = np.percentile(z, 1)
                mask = z >= threshold
                x, y, z = x[mask], y[mask], z[mask]

            # 检查是否有足够的点进行计算
            if len(x) < 10:  # 过滤点数量过少的文件
                lai_results.append(np.nan)
                continue

            # 计算体素大小（基于最近2个邻居的平均距离）
            lasdata = list(zip(x, y, z))
            tree = spatial.cKDTree(lasdata)
            k = 2  # 取最近的2个邻居
            distances = []

            for i in range(len(lasdata)):
                dist, _ = tree.query(lasdata[i], k)
                distances.append(np.sum(dist))  # 累加每个点到2个邻居的距离

            mean_dist = np.mean(distances)
            voxel_size = 1.5 * mean_dist  # 体素大小为平均距离的1.5倍

            # 计算点云坐标范围
            x_min, x_max = x.min(), x.max()
            y_min, y_max = y.min(), y.max()
            z_min, z_max = z.min(), z.max()

            # 计算每个点的体素偏移量
            x_offset = np.ceil((x - x_min) / voxel_size).astype(int)
            y_offset = np.ceil((y - y_min) / voxel_size).astype(int)
            z_offset = np.ceil((z - z_min) / voxel_size).astype(int)

            # 计算体素网格维度
            cols = np.ceil((x_max - x_min) / voxel_size).astype(int)
            rows = np.ceil((y_max - y_min) / voxel_size).astype(int)
            heis = np.ceil((z_max - z_min) / voxel_size).astype(int)

            # 每层的总可能体素数量
            voxels_per_layer = rows * cols

            # 体素去重（每个体素只计一次）
            voxels = np.column_stack([x_offset, y_offset, z_offset])
            unique_voxels = np.unique(voxels, axis=0)

            # 计算树冠二维投影所覆盖的真实体素数量（树冠的占地面积）
            xy_voxels = np.column_stack([x_offset, y_offset])
            unique_xy = np.unique(xy_voxels, axis=0)
            true_voxels_per_layer = len(unique_xy)  # 真实的有效圆柱分母

            if true_voxels_per_layer == 0:
                lai_results.append(0)
                continue

            # 按高度层统计体素占比
            voxel_df = pd.DataFrame(unique_voxels, columns=['x', 'y', 'z'])
            layer_counts = voxel_df.groupby('z').size()  # 每层包含的树叶体素数量

            # 使用真实的投影面积体素数作为分母
            layer_ratios = layer_counts / true_voxels_per_layer

            # 计算LAI（使用修正后的占比）
            lai = layer_ratios.sum() * 1.1
            lai_results.append(lai)

        except Exception as e:
            print(f"处理文件 {filename} 时出错: {str(e)}")
            lai_results.append(np.nan)

    # 保存结果到CSV
    result_df = pd.DataFrame({
        'filename': filenames,
        'lai': lai_results
    })
    result_df.to_csv(output_csv, index=False, encoding='utf-8')
    print(f"LAI计算完成，结果已保存至 {output_csv}")


# 使用示例
if __name__ == "__main__":
    # 输入文件夹路径（包含每棵树的点云文件）
    input_directory = r"I:\pinghemiyou\qt\mask_process\step7_final_nonground_las"
    # 输出结果CSV路径
    output_csv_path = r"I:\pinghemiyou\qt\mask_process\nonground_lai_result.csv"

    # 执行计算
    process_point_clouds_and_calculate_lai(input_directory, output_csv_path)