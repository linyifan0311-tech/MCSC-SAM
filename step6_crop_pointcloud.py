import os
import laspy
import rasterio
from rasterio.transform import rowcol, xy


def get_large_tif_info(large_tif_path):
    """获取原始大图的地理变换和坐标系统"""
    with rasterio.open(large_tif_path) as src:
        return {
            "transform": src.transform,  # 地理变换矩阵（像素坐标转地理坐标）
            "crs": src.crs,  # 坐标系统
            "width": src.width,  # 宽度（像素）
            "height": src.height  # 高度（像素）
        }


def get_small_tif_geo_bounds_in_large(small_tif_path, large_tif_info):
    """获取小正射图在原始大图中的地理坐标范围"""
    with rasterio.open(small_tif_path) as small_src:
        # 小图的地理边界
        small_left, small_bottom, small_right, small_top = small_src.bounds

        # 验证坐标系统一致性
        if small_src.crs != large_tif_info["crs"]:
            raise ValueError(f"坐标系统不一致：小图 {small_src.crs} vs 大图 {large_tif_info['crs']}")

        # 将小图边界转换为大图中的地理范围（直接使用小图的地理坐标，因为它是大图的一部分）
        return {
            "min_x": small_left,
            "max_x": small_right,
            "min_y": small_bottom,
            "max_y": small_top
        }


def crop_matching_las(small_tif_name, original_las_dir, output_path, geo_bounds):
    """
    裁剪与小正射图同名的点云文件
    small_tif_name: 小正射图文件名（不含扩展名）
    original_las_dir: 原始点云文件夹
    output_path: 裁剪后保存路径
    geo_bounds: 裁剪的地理范围
    """
    # 构建对应的点云文件名（假设仅扩展名不同）
    las_filename = f"{small_tif_name}.las"
    las_path = os.path.join(original_las_dir, las_filename)

    # 检查点云文件是否存在
    if not os.path.exists(las_path):
        print(f"警告：未找到与 {small_tif_name}.tif 对应的点云文件 {las_filename}")
        return False

    # 裁剪点云
    with laspy.file.File(las_path, mode='r') as in_las:
        # 筛选范围内的点
        mask = (
                (in_las.x >= geo_bounds["min_x"]) &
                (in_las.x <= geo_bounds["max_x"]) &
                (in_las.y >= geo_bounds["min_y"]) &
                (in_las.y <= geo_bounds["max_y"])
        )

        # 检查是否有有效点
        if not mask.any():
            print(f"注意：{las_filename} 中没有位于小图范围内的点")
            return False

        # 保存裁剪结果
        out_las = laspy.file.File(output_path, mode='w', header=in_las.header)
        out_las.points = in_las.points[mask]
        out_las.close()
        return True


def batch_process(small_tifs_dir, large_tif_path, original_las_dir, output_root_dir):
    """
    批量处理：仅裁剪与小正射图同名的点云文件
    small_tifs_dir: 小正射图文件夹
    large_tif_path: 原始大图正射图路径
    original_las_dir: 原始树点云文件夹
    output_root_dir: 输出根目录
    """
    # 获取原始大图信息
    try:
        large_tif_info = get_large_tif_info(large_tif_path)
    except Exception as e:
        print(f"读取原始大图失败：{e}")
        return

    # 获取所有小正射图
    small_tif_files = [f for f in os.listdir(small_tifs_dir) if f.lower().endswith('.tif')]
    if not small_tif_files:
        print("未找到小正射图文件")
        return

    # 创建输出根目录
    os.makedirs(output_root_dir, exist_ok=True)

    # 遍历小正射图
    for small_tif in small_tif_files:
        small_tif_name = os.path.splitext(small_tif)[0]  # 提取文件名（不含扩展名）
        small_tif_path = os.path.join(small_tifs_dir, small_tif)

        # 1. 计算小图在大图中的地理范围
        try:
            geo_bounds = get_small_tif_geo_bounds_in_large(small_tif_path, large_tif_info)
            print(f"\n处理小图：{small_tif}，地理范围：{geo_bounds}")
        except Exception as e:
            print(f"处理 {small_tif} 时出错：{e}，跳过")
            continue

        # 2. 构建输出路径
        output_las_path = os.path.join(output_root_dir, f"{small_tif_name}.las")

        # 3. 裁剪对应的点云文件（仅处理同名文件）
        success = crop_matching_las(
            small_tif_name=small_tif_name,
            original_las_dir=original_las_dir,
            output_path=output_las_path,
            geo_bounds=geo_bounds
        )

        if success:
            print(f"已成功裁剪并保存：{output_las_path}")


if __name__ == "__main__":
    # 请根据实际情况修改以下路径
    small_tifs_directory = r"H:\baisha\2025.10.27\cailiao\Visible\edge_remove_and_connectivity_4\orthos"  # 小正射图文件夹（仅包含树）
    large_tif_path = r"H:\baisha\2025.10.27\cailiao\Visible\caijian_1\cropped_polygon_image.tif"  # 原始大图正射图路径
    original_las_directory = r"H:\baisha\2025.10.27\cailiao\Visible\mask_divide_2\cropped_pointclouds"  # 原始树点云文件夹（每棵树一个las）
    output_root_directory = r"H:\baisha\2025.10.27\cailiao\Visible\caijianpointcloud_5"  # 输出目录（裁剪后的点云）

    batch_process(
        small_tifs_dir=small_tifs_directory,
        large_tif_path=large_tif_path,
        original_las_dir=original_las_directory,
        output_root_dir=output_root_directory
    )