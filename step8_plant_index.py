import rasterio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path


# -------------------------- 1. 读取 TIF 影像 --------------------------
def read_tif_image(tif_path):
    try:
        with rasterio.open(tif_path) as src:
            R = src.read(1).astype(np.float32)
            G = src.read(2).astype(np.float32)
            B = src.read(3).astype(np.float32)

        # 1. 剔除黑边 (原始掩码)
        valid_mask = ~((R <= 1e-6) & (G <= 1e-6) & (B <= 1e-6))

        # 为了避免除以0，给RGB加上极小值
        R_safe = R + 1e-6
        G_safe = G + 1e-6
        B_safe = B + 1e-6

        # 2. 计算 ExG 植被指数 (用于区分植物和土壤/阴影)
        r = R_safe / (R_safe + G_safe + B_safe)
        g = G_safe / (R_safe + G_safe + B_safe)
        b = B_safe / (R_safe + G_safe + B_safe)
        ExG = 2 * g - r - b

        # 3. 生成纯植被掩码
        veg_mask = valid_mask & (ExG > 0)

        return R, G, B, veg_mask, True
    except Exception as e:
        print(f"读取 {tif_path} 时出错: {str(e)}")
        return None, None, None, None, False


# -------------------------- 2. 计算植被指数 --------------------------
def calculate_vegetation_indices(R, G, B, mask):
    """
    根据公式计算各类植被指数，排除黑色像素
    :param R: 红波段数组
    :param G: 绿波段数组
    :param B: 蓝波段数组
    :param mask: 非黑色像素掩码（True表示有效像素）
    :return: 各指数计算结果（字典形式）
    """
    # 避免除以 0，给分母加极小值
    eps = 1e-8

    # 初始化指数数组（黑色区域设为NaN）
    ExG = np.full_like(R, np.nan)
    ExR = np.full_like(R, np.nan)
    NVI = np.full_like(R, np.nan)
    GLI = np.full_like(R, np.nan)
    VARI = np.full_like(R, np.nan)
    NDYI = np.full_like(R, np.nan)

    # 只在非黑色像素区域计算指数
    ExG[mask] = 2.0 * G[mask] - R[mask] - B[mask]
    ExR[mask] = 1.4 * R[mask] - B[mask]
    NVI[mask] = ExG[mask] - ExR[mask]
    GLI[mask] = (G[mask] - R[mask]) / (G[mask] + R[mask] + eps)
    VARI[mask] = (G[mask] - R[mask]) / (G[mask] + R[mask] - B[mask] + eps)
    NDYI[mask] = (G[mask] - B[mask]) / (G[mask] + B[mask] + eps)

    return {
        "ExG": ExG,
        "ExR": ExR,
        "NVI": NVI,
        "GLI": GLI,
        "VARI": VARI,
        "NDYI": NDYI
    }


# -------------------------- 3. 可视化 --------------------------
def visualize_indices(indices_dict, tif_filename, save_dir):
    """
    可视化各植被指数并保存图片
    :param indices_dict: 各指数的字典
    :param tif_filename: 原始TIF文件名，用于生成输出图片名
    :param save_dir: 图片保存目录
    """
    try:
        # 创建保存目录（如果不存在）
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        plt.figure(figsize=(15, 10))
        for i, (index_name, index_array) in enumerate(indices_dict.items(), start=1):
            plt.subplot(2, 3, i)
            # 使用nanma sked=True忽略NaN值（黑色区域）
            plt.imshow(index_array, cmap='viridis', interpolation='none')
            plt.title(index_name)
            plt.colorbar(label='Value')
            plt.axis('off')

        plt.tight_layout()
        save_path = os.path.join(save_dir, f"{os.path.splitext(tif_filename)[0]}_indices.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        return True
    except Exception as e:
        print(f"可视化并保存图片时出错: {str(e)}")
        return False


# -------------------------- 4. 批量处理文件夹 --------------------------
def process_folder(input_folder):
    """
    处理文件夹中的所有TIF文件，将结果汇总保存到单个CSV
    :param input_folder: 包含TIF文件的文件夹路径
    """
    # 检查输入文件夹是否存在
    if not os.path.isdir(input_folder):
        print(f"错误: 文件夹 {input_folder} 不存在")
        return

    # 创建输出目录
    output_root = os.path.join(input_folder, "vegetation_indices_results")
    plots_dir = os.path.join(output_root, "plots")
    combined_csv_path = os.path.join(output_root, "all_vegetation_indices.csv")

    print(f"开始处理文件夹: {input_folder}")
    print(f"汇总结果将保存至: {combined_csv_path}")
    print(f"可视化结果将保存至: {plots_dir}")

    # 遍历文件夹中的所有TIF文件
    tif_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.tif', '.tiff'))]

    if not tif_files:
        print(f"在 {input_folder} 中未找到任何TIF文件")
        return

    # 存储所有文件的结果
    all_results = []

    # 处理每个TIF文件
    for i, tif_file in enumerate(tif_files, 1):
        print(f"\n处理文件 {i}/{len(tif_files)}: {tif_file}")
        tif_path = os.path.join(input_folder, tif_file)
        file_basename = os.path.splitext(tif_file)[0]

        # 1. 读取影像（获取掩码）
        R_band, G_band, B_band, mask, success = read_tif_image(tif_path)
        if not success:
            continue

        # 检查有效像素数量
        valid_pixels = np.sum(mask)
        if valid_pixels == 0:
            print(f"警告: {tif_file} 中没有有效像素（全为黑色），跳过该文件")
            continue
        print(f"有效像素数量: {valid_pixels}")

        # 2. 计算指数（使用掩码排除黑色像素）
        indices_result = calculate_vegetation_indices(R_band, G_band, B_band, mask)

        # 3. 准备当前文件的统计数据（使用np.nanxxx函数自动忽略NaN值）
        file_stats = {"filename": file_basename}
        for index_name, index_array in indices_result.items():
            file_stats[f"{index_name}_mean"] = np.nanmean(index_array)
            # file_stats[f"{index_name}_max"] = np.nanmax(index_array)
            # file_stats[f"{index_name}_min"] = np.nanmin(index_array)
            # file_stats[f"{index_name}_std"] = np.nanstd(index_array)
            # file_stats[f"{index_name}_median"] = np.nanmedian(index_array)

        all_results.append(file_stats)

        # 4. 可视化并保存
        if visualize_indices(indices_result, tif_file, plots_dir):
            print(f"指数可视化结果已保存")

    # 5. 将所有结果保存到单个CSV
    if all_results:
        try:
            # 创建输出目录（如果不存在）
            Path(output_root).mkdir(parents=True, exist_ok=True)

            df = pd.DataFrame(all_results)
            df.to_csv(combined_csv_path, index=False)
            print(f"\n所有植被指数已汇总保存至 {combined_csv_path}")
            print(f"共处理 {len(all_results)} 个文件")
        except Exception as e:
            print(f"保存汇总CSV时出错: {str(e)}")
    else:
        print("\n没有成功处理任何文件，未生成CSV")


# -------------------------- 主程序入口 --------------------------
if __name__ == "__main__":
    import sys

    # 检查命令行参数
    if len(sys.argv) > 1:
        input_folder = sys.argv[1]
    else:
        # 如果没有提供命令行参数，手动输入文件夹路径
        input_folder = input("请输入包含TIF文件的文件夹路径: ").strip()

    # 处理文件夹
    process_folder(input_folder)
    print("\n处理完成！")