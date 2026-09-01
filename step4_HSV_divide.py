import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
# 【修改点1】引入了 binary_fill_holes
from scipy.ndimage import label, find_objects, binary_fill_holes
import os
import glob
from tqdm import tqdm
import rasterio
from rasterio.transform import Affine

def segment_tree_adaptive_center(image_path, min_tree_size_ratio=0.05):
    """
    自适应检测图像中心区域的树，根据树的大小动态调整中心范围
    返回结果增加地理变换和投影信息
    """
    # 使用rasterio读取图像以保留地理信息
    with rasterio.open(image_path) as src:
        # 读取图像数据（BGR格式，与OpenCV一致）
        image = src.read([3, 2, 1]).transpose(1, 2, 0)
        transform = src.transform
        crs = src.crs
        meta = src.meta.copy()

    if image is None:
        raise ValueError(f"无法读取图像: {image_path}")

    # 获取图像尺寸和面积
    height, width = image.shape[:2]
    total_area = height * width

    # 转换色彩空间
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 1. 初步检测所有绿色区域（包含树和可能的草）
    lower_green = np.array([30, 20, 20])
    upper_green = np.array([90, 255, 255])
    green_mask = cv2.inRange(image_hsv, lower_green, upper_green)

    # 去除小噪声区域
    kernel = np.ones((3, 3), np.uint8)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)

    # 2. 识别绿色区域中的连通组件
    labeled_mask, num_features = label(green_mask)
    objects = find_objects(labeled_mask)

    # 分析每个连通组件
    green_regions = []
    for i in range(1, num_features + 1):
        if objects[i - 1] is None:
            continue

        slice_y, slice_x = objects[i - 1]
        region_height = slice_y.stop - slice_y.start
        region_width = slice_x.stop - slice_x.start
        region_area = region_height * region_width

        # 过滤太小的区域
        if region_area / total_area < min_tree_size_ratio:
            continue

        # 计算区域的中心坐标
        center_y = (slice_y.start + slice_y.stop) / 2
        center_x = (slice_x.start + slice_x.stop) / 2

        # 计算区域中心与图像中心的距离
        image_center_y, image_center_x = height / 2, width / 2
        distance = np.sqrt((center_y - image_center_y) ** 2 + (center_x - image_center_x) ** 2)

        green_regions.append({
            'area': region_area,
            'center': (center_y, center_x),
            'distance': distance,
            'bbox': (slice_y, slice_x),
            'label': i
        })

    if not green_regions:
        raise ValueError("未检测到足够大的绿色区域")

    # 3. 确定最可能是树的区域（中心附近的大区域）
    green_regions.sort(key=lambda x: (x['distance'], -x['area']))
    tree_region = green_regions[0]
    target_label = tree_region['label']  # 【修改点2】提取确切的连通域标签

    # 4. 获取树的核心掩码
    # 抛弃原来基于Bounding Box的范围截图，直接使用连通域的精准形状
    tree_mask = (labeled_mask == target_label)

    # 5. 保留中间非绿色部分
    # 从掩码边缘向内检测，填充被绿色像素完全包围的“孔洞”（如非绿色的树枝、阴影等）
    tree_mask = binary_fill_holes(tree_mask)

    # 6. 优化掩码
    kernel = np.ones((5, 5), np.uint8)
    tree_mask = tree_mask.astype(np.uint8) * 255

    tree_mask = cv2.morphologyEx(tree_mask, cv2.MORPH_OPEN, kernel)
    tree_mask = tree_mask.astype(bool)

    # 7. 生成结果图像
    tree_only = image_rgb.copy()
    tree_only[~tree_mask] = [0, 0, 0]

    # 创建分割可视化结果
    segmented_vis = image_rgb.copy()
    tree_overlay = np.zeros_like(image_rgb)
    tree_overlay[tree_mask] = [0, 255, 0]
    segmented_vis = cv2.addWeighted(segmented_vis, 0.7, tree_overlay, 0.3, 0)

    return image_rgb, segmented_vis, tree_only, tree_mask, transform, crs, meta


def save_with_original_format(image, input_path, output_path, transform, crs, meta):
    """根据输入图像的格式保存输出图像，保留地理坐标信息"""
    _, ext = os.path.splitext(input_path)
    format = ext.lower()[1:] if ext else 'tiff'

    if format in ['tif', 'tiff', 'geotiff']:
        meta.update(
            driver='GTiff',
            count=3,
            dtype=image.dtype,
            transform=transform,
            crs=crs
        )
        image_raster = image.transpose(2, 0, 1)
        with rasterio.open(output_path, 'w', **meta) as dst:
            dst.write(image_raster)
    else:
        format_mapping = {
            'jpg': 'JPEG',
            'jpeg': 'JPEG',
            'png': 'PNG',
            'bmp': 'BMP',
            'gif': 'GIF'
        }
        save_format = format_mapping.get(format, format.upper())
        img = Image.fromarray(image)
        img.save(output_path, format=save_format)

def save_mask_with_geo(mask, output_path, transform, crs, meta):
    """保存掩码并保留地理坐标信息"""
    meta.update(
        driver='GTiff',
        count=1,
        dtype=np.uint8,
        transform=transform,
        crs=crs
    )
    mask_raster = mask.reshape(1, mask.shape[0], mask.shape[1])
    with rasterio.open(output_path, 'w', **meta) as dst:
        dst.write(mask_raster)


def process_batch(input_paths, output_dir, min_tree_size_ratio=0.05, visualize=False):
    """批量处理图像，保留地理坐标信息"""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "tree_only"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "segmented_vis"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "masks"), exist_ok=True)

    success_count = 0
    fail_count = 0
    failed_files = []

    for img_path in tqdm(input_paths, desc="处理进度"):
        try:
            filename = os.path.basename(img_path)
            name_without_ext = os.path.splitext(filename)[0]

            vis_output_path = os.path.join(output_dir, "segmented_vis", filename)
            tree_output_path = os.path.join(output_dir, "tree_only", filename)
            mask_output_path = os.path.join(output_dir, "masks", f"{name_without_ext}.png")

            original, segmented_vis, tree_only, tree_mask, transform, crs, meta = segment_tree_adaptive_center(
                img_path, min_tree_size_ratio
            )

            # 1. 保存可视化图片
            save_with_original_format(segmented_vis, img_path, vis_output_path, transform, crs, meta)

            # 2. 直接保存原分辨率的 tree_only，背景已经是纯黑 [0,0,0]
            save_with_original_format(tree_only, img_path, tree_output_path, transform, crs, meta)

            # 3. 直接保存原分辨率的掩码 mask
            mask_8bit = (tree_mask * 255).astype(np.uint8)
            save_mask_with_geo(mask_8bit, mask_output_path, transform, crs, meta)

            if visualize:
                plt.figure(figsize=(18, 6))
                plt.subplot(131)
                plt.imshow(original)
                plt.title('原始图像')
                plt.axis('off')

                plt.subplot(132)
                plt.imshow(segmented_vis)
                plt.title('树区域标记')
                plt.axis('off')

                plt.subplot(133)
                # 可视化调整为显示完整原尺寸图像
                plt.imshow(tree_only)
                plt.title('仅显示树（原分辨率背景填充黑）')
                plt.axis('off')

                plt.tight_layout()
                plt.show()

            success_count += 1

        except Exception as e:
            print(f"\n处理文件 {img_path} 时出错: {str(e)}")
            fail_count += 1
            failed_files.append(img_path)

    print("\n处理完成!")
    print(f"成功处理: {success_count} 个文件")
    print(f"处理失败: {fail_count} 个文件")

    if failed_files:
        print("失败的文件列表:")
        for f in failed_files:
            print(f"- {f}")

if __name__ == "__main__":
    input_dir = r"I:\baisha\2025.10.27\train\sam3_result\Ablation_Experiment\caijian\caijian_result\single\output_mask1"
    output_dir = r"I:\baisha\2025.10.27\train\sam3_result\Ablation_Experiment\caijian\caijian_result\single\Kmeansdivide_3"
    min_tree_size_ratio = 0.05
    visualize = False

    image_files = []
    for ext in ['*.tif', '*.tiff', '*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif']:
        image_files.extend(glob.glob(os.path.join(input_dir, ext)))

    if not image_files:
        print(f"在目录 {input_dir} 中未找到图像文件")
    else:
        print(f"找到 {len(image_files)} 个图像文件，开始处理...")
        process_batch(image_files, output_dir, min_tree_size_ratio, visualize)