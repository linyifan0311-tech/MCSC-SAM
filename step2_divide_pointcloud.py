import numpy as np
from PIL import Image
import os
import glob
import laspy  # 兼容1.7.0版本
from osgeo import gdal
import rasterio  # 用于保存带地理信息的图像
from rasterio.transform import Affine

Image.MAX_IMAGE_PIXELS = None  # 完全解除限制


def get_ortho_geo_info(ortho_path):
    """从正射图中自动提取地理参考信息及完整变换参数"""
    try:
        # 使用rasterio获取更完整的地理信息
        with rasterio.open(ortho_path) as src:
            transform = src.transform
            crs = src.crs
            min_x_geo = transform[2]
            max_y_geo = transform[5]
            pixel_width = transform[0]
            pixel_height = transform[4]

        print(f"成功提取地理信息:")
        print(f"  左上角坐标: ({min_x_geo}, {max_y_geo})")
        print(f"  像素尺寸: {pixel_width} x {pixel_height}")
        print(f"  坐标系: {crs}")

        # 返回完整地理信息（包含变换矩阵和坐标系统）
        return (min_x_geo, max_y_geo, pixel_width, pixel_height, transform, crs)

    except Exception as e:
        print(f"提取地理信息失败: {str(e)}")
        return None


def resize_mask_to_match_ortho(mask_path, ortho_size):
    """调整蒙版图大小以匹配正射图的尺寸"""
    with Image.open(mask_path) as mask_img:
        resized_mask = mask_img.resize(ortho_size, Image.NEAREST)
        mask_array = np.array(resized_mask)
        mask_array = (mask_array > 0).astype(np.uint8)
        return mask_array


def find_mask_bounding_box(mask_array):
    """找到蒙版图中白色区域的边界框（像素坐标）"""
    non_zero_coords = np.argwhere(mask_array > 0)
    if non_zero_coords.size == 0:
        return None
    min_y, min_x = np.min(non_zero_coords, axis=0)
    max_y, max_x = np.max(non_zero_coords, axis=0)
    return (min_x, min_y, max_x, max_y)


def pixel_to_geo_coords(pixel_x, pixel_y, ortho_geo_info):
    """将正射图像素坐标转换为地理坐标"""
    min_x_geo, max_y_geo, pixel_width, pixel_height = ortho_geo_info[:4]
    geo_x = min_x_geo + pixel_x * pixel_width
    geo_y = max_y_geo + pixel_y * pixel_height
    return (geo_x, geo_y)


def crop_ortho_by_mask(ortho_array, mask_array):
    """裁剪正射图有效区域并返回边界框"""
    bbox = find_mask_bounding_box(mask_array)
    if bbox is None:
        return None, None
    min_x, min_y, max_x, max_y = bbox
    cropped_ortho = ortho_array[min_y:max_y + 1, min_x:max_x + 1]
    cropped_mask = mask_array[min_y:max_y + 1, min_x:max_x + 1]
    if len(cropped_ortho.shape) == 3 and len(cropped_mask.shape) == 2:
        mask_3d = np.stack([cropped_mask] * cropped_ortho.shape[2], axis=-1)
    else:
        mask_3d = cropped_mask
    return cropped_ortho * mask_3d, bbox


def crop_pointcloud_by_bbox(las_path, bbox_pixel, ortho_geo_info, output_path):
    """修复属性访问方式的点云裁剪函数（保持不变）"""
    min_x_pix, min_y_pix, max_x_pix, max_y_pix = bbox_pixel

    # 将像素边界框转换为地理坐标
    min_x_geo, max_y_geo = pixel_to_geo_coords(min_x_pix, min_y_pix, ortho_geo_info)
    max_x_geo, min_y_geo = pixel_to_geo_coords(max_x_pix, max_y_pix, ortho_geo_info)

    # 读取LAS文件（laspy 1.7.0方式）
    las = laspy.file.File(las_path, mode="r")

    # 点云坐标筛选（使用正确的属性访问方式）
    mask = (
            (las.x >= min_x_geo) & (las.x <= max_x_geo) &
            (las.y >= min_y_geo) & (las.y <= max_y_geo)
    )

    # 统计符合条件的点数量
    point_count = np.sum(mask)

    if point_count > 0:
        # 创建新的LAS文件并复制头部信息
        out_las = laspy.file.File(output_path, mode="w", header=las.header)

        # 修复属性复制方式 - 使用getattr替代__getattr__
        for dimension in las.point_format:
            attr_name = dimension.name
            # 正确获取属性值的方式
            attr_values = getattr(las, attr_name)
            # 应用掩码并设置到新文件
            setattr(out_las, attr_name, attr_values[mask])

        # 关闭文件
        out_las.close()
        las.close()
        return True, point_count
    else:
        las.close()
        return False, 0


def process_single_mask(mask_path, ortho_array, ortho_size, ortho_geo_info,
                        las_path, output_img_folder, output_las_folder):
    """处理单个蒙版图"""
    try:
        filename = os.path.splitext(os.path.basename(mask_path))[0]
        mask_array = resize_mask_to_match_ortho(mask_path, ortho_size)
        bbox_pixel = find_mask_bounding_box(mask_array)

        if bbox_pixel is None:
            print(f"警告: {filename} 中没有有效区域，跳过处理")
            return False

        # 裁剪正射图
        cropped_array, bbox = crop_ortho_by_mask(ortho_array, mask_array)
        if cropped_array is not None and bbox is not None:
            min_x, min_y, _, _ = bbox
            img_output_path = os.path.join(output_img_folder, f"{filename}_cropped.tif")

            # 获取原始图像的变换矩阵和坐标系统
            original_transform, crs = ortho_geo_info[4], ortho_geo_info[5]

            # 计算裁剪后图像的变换矩阵
            new_transform = Affine(
                original_transform.a, original_transform.b, original_transform.c + min_x * original_transform.a,
                original_transform.d, original_transform.e, original_transform.f + min_y * original_transform.e
            )

            # 使用rasterio保存带地理信息的图像
            with rasterio.open(
                    img_output_path,
                    'w',
                    driver='GTiff',
                    height=cropped_array.shape[0],
                    width=cropped_array.shape[1],
                    count=cropped_array.shape[2] if len(cropped_array.shape) == 3 else 1,
                    dtype=cropped_array.dtype,
                    crs=crs,
                    transform=new_transform
            ) as dst:
                if len(cropped_array.shape) == 3:
                    # 对于RGB图像，需要按波段写入
                    for i in range(cropped_array.shape[2]):
                        dst.write(cropped_array[:, :, i], i + 1)
                else:
                    dst.write(cropped_array, 1)

        # 裁剪点云（保持不变）
        las_output_path = os.path.join(output_las_folder, f"{filename}_cropped.las")
        success, point_count = crop_pointcloud_by_bbox(
            las_path, bbox_pixel, ortho_geo_info, las_output_path
        )

        if success:
            print(f"已处理: {filename}，裁剪后点数量: {point_count}")
            return True
        else:
            print(f"警告: {filename} 未找到匹配的点云数据")
            return False

    except Exception as e:
        print(f"处理 {mask_path} 时出错: {str(e)}")
        return False


def batch_process(mask_folder, ortho_path, las_path,
                  output_img_folder, output_las_folder):
    """批量处理蒙版图"""
    if not os.path.isdir(mask_folder):
        print(f"错误: 蒙版图文件夹 '{mask_folder}' 不存在")
        return

    for path in [ortho_path, las_path]:
        if not os.path.exists(path):
            print(f"错误: 文件 '{path}' 不存在")
            return

    # 自动提取正射图地理信息
    ortho_geo_info = get_ortho_geo_info(ortho_path)
    if ortho_geo_info is None:
        print("无法继续处理，缺少地理参考信息")
        return

    # 创建输出文件夹
    os.makedirs(output_img_folder, exist_ok=True)
    os.makedirs(output_las_folder, exist_ok=True)

    # 读取正射图
    try:
        with Image.open(ortho_path) as ortho_img:
            ortho_array = np.array(ortho_img)
            ortho_size = ortho_img.size
            print(f"正射图尺寸: {ortho_size[0]}x{ortho_size[1]}")
    except Exception as e:
        print(f"读取正射图时出错: {str(e)}")
        return

    # 获取所有蒙版图
    mask_files = glob.glob(os.path.join(mask_folder, "*.png"))
    if not mask_files:
        print(f"警告: 在 '{mask_folder}' 中未找到任何PNG文件")
        return

    print(f"找到 {len(mask_files)} 个蒙版图文件，开始处理...")

    # 处理每个蒙版图
    success_count = 0
    for mask_file in mask_files:
        if process_single_mask(mask_file, ortho_array, ortho_size,
                               ortho_geo_info, las_path,
                               output_img_folder, output_las_folder):
            success_count += 1

    print(f"处理完成！成功处理 {success_count}/{len(mask_files)} 个蒙版图")
    print(f"正射图结果保存在: {output_img_folder}")
    print(f"点云结果保存在: {output_las_folder}")


def main():
    # 配置参数
    mask_folder = r"I:\baisha\2025.10.27\train\sam3_result\Ablation_Experiment\finally2\MFSC_new_eval\individual_masks"  # 蒙版图文件夹
    ortho_path = r"I:\baisha\2025.10.27\cailiao\Visible\caijian_1\cropped_polygon_image.tif"  # 正射图路径
    las_path = r"I:\baisha\2025.10.27\cailiao\Visible\caijian_1\cropped_polygon_point_cloud.las"  # 点云文件路径
    output_img_folder = r"I:\baisha\2025.10.27\cailiao\Visible\test\segmented_results"  # 裁剪后的正射图输出文件夹
    output_las_folder = r"I:\baisha\2025.10.27\cailiao\Visible\test\cropped_pointclouds"  # 裁剪后的点云输出文件夹

    # 执行批量处理
    batch_process(mask_folder, ortho_path, las_path,
                  output_img_folder, output_las_folder)


if __name__ == "__main__":
    main()