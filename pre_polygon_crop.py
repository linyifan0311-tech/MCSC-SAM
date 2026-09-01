import geopandas as gpd
import rasterio
from rasterio.mask import mask
import laspy  # 适配1.7.0版本
import numpy as np
from shapely.geometry import Point, Polygon, mapping
import matplotlib.pyplot as plt
import os

plt.rcParams["font.sans-serif"] = ["SimHei"] # 设置字体为黑体
plt.rcParams["axes.unicode_minus"] = False # 正常显示负号

def crop_ortho_and_las(shapefile_path, ortho_path, las_path, output_dir):
    """
    使用shapefile裁剪正射影像和点云，严格遵守原始shapefile的点顺序生成边，输出多边形/矩形裁剪结果

    参数:
        shapefile_path: 多边形shapefile路径（需确保内部存储单个多边形，点序已按预期排列）
        ortho_path: 正射影像tif文件路径
        las_path: 点云las文件路径
        output_dir: 输出文件目录
    """
    try:
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)

        # --------------------------
        # 步骤1: 读取形状文件，严格保留原始点顺序
        # --------------------------
        print("读取形状文件并保留原始点顺序...")
        gdf = gpd.read_file(shapefile_path)

        if len(gdf) != 1:
            raise ValueError(f"形状文件包含{len(gdf)}个几何对象，需确保仅含1个多边形以保留点序！")

        # 从GeoDataFrame中获取原始Polygon对象，其exterior.coords包含原始点序
        original_poly = gdf.geometry.iloc[0]
        if not isinstance(original_poly, Polygon):
            raise TypeError(f"形状文件几何类型为{type(original_poly)}，需为单个Polygon类型！")

        # 提取原始点序列（验证点序是否正确）
        original_points = list(original_poly.exterior.coords)  # 格式：[(x1,y1), (x2,y2), ..., (x1,y1)]（闭合点）
        print(f"原始多边形点数量（含闭合点）: {len(original_points)}")
        print(f"原始点顺序（前5个）: {original_points[:5]}")  # 打印前5个点验证顺序

        # 计算原始多边形及其最小外接矩形（矩形点序不影响，因矩形是规则几何）
        poly_area = original_poly.area
        bbox_geom = original_poly.envelope  # 最小外接矩形（轴对齐）
        bbox_area = bbox_geom.area
        print(f"原始多边形面积: {poly_area:.2f} 平方单位")
        print(f"最小矩形区域面积: {bbox_area:.2f} 平方单位")

        # --------------------------
        # 步骤2: 读取正射影像，按原始点序多边形裁剪
        # --------------------------
        print("读取并裁剪正射影像（严格按原始点序）...")
        with rasterio.open(ortho_path) as src:
            ortho_crs = src.crs
            print(f"正射影像坐标系: {ortho_crs}")
            print(f"形状文件坐标系: {gdf.crs}")

            # 坐标系统一（若不一致，转换原始多边形而非整个GeoDataFrame，避免点序意外变化）
            original_poly_proj = original_poly
            bbox_geom_proj = bbox_geom
            if gdf.crs != ortho_crs:
                print(f"转换多边形坐标系从 {gdf.crs} 到 {ortho_crs}（保持点序）")
                original_poly_proj = original_poly.to_crs(ortho_crs)
                bbox_geom_proj = bbox_geom.to_crs(ortho_crs)
                # 验证转换后的点序（应与原始一致）
                proj_points = list(original_poly_proj.exterior.coords)
                print(f"转换后点顺序（前5个）: {proj_points[:5]}")

            # shapely的mapping函数会严格保留Polygon的exterior点序
            geoms_poly = [mapping(original_poly_proj)]  # 单个原始点序多边形
            geoms_bbox = [mapping(bbox_geom_proj)]  # 矩形（不影响点序）

            # 裁剪多边形区域（按原始点序边生成的区域）
            out_image_poly, out_transform_poly = mask(
                src, geoms_poly, crop=True, all_touched=False  # all_touched=False确保裁剪边界严格
            )
            # 更新影像元数据（保持原始影像的波段、数据类型等）
            out_meta_poly = src.meta.copy()
            out_meta_poly.update({
                "driver": "GTiff",
                "height": out_image_poly.shape[1],  # 裁剪后高度（波段数, 高, 宽）
                "width": out_image_poly.shape[2],  # 裁剪后宽度
                "transform": out_transform_poly,  # 裁剪后的地理变换
                "crs": ortho_crs
            })

            # 裁剪矩形区域（对比用，点序不影响）
            out_image_bbox, out_transform_bbox = mask(
                src, geoms_bbox, crop=True, all_touched=False
            )
            out_meta_bbox = src.meta.copy()
            out_meta_bbox.update({
                "driver": "GTiff",
                "height": out_image_bbox.shape[1],
                "width": out_image_bbox.shape[2],
                "transform": out_transform_bbox,
                "crs": ortho_crs
            })

        # 保存裁剪后的正射影像
        poly_ortho_path = os.path.join(output_dir, "cropped_polygon_image.tif")
        with rasterio.open(poly_ortho_path, "w", **out_meta_poly) as dest:
            dest.write(out_image_poly)
        print(f"多边形裁剪影像（原始点序）已保存: {poly_ortho_path}")

        bbox_ortho_path = os.path.join(output_dir, "cropped_bbox_image.tif")
        with rasterio.open(bbox_ortho_path, "w", **out_meta_bbox) as dest:
            dest.write(out_image_bbox)
        print(f"矩形裁剪影像已保存: {bbox_ortho_path}")

        # --------------------------
        # 步骤3: 裁剪点云，按原始点序多边形筛选
        # --------------------------
        print("读取并裁剪点云（严格按原始点序多边形）...")
        # 读取LAS文件（laspy 1.7.0语法）
        with laspy.file.File(las_path, mode="r") as las:
            # 获取点云坐标和原始属性
            x_coords = las.x
            y_coords = las.y
            z_coords = las.z
            original_count = len(x_coords)
            print(f"原始点云数量: {original_count}")

            # 生成点云的GeoSeries（按原始点序多边形筛选）
            points_geoseries = gpd.GeoSeries(
                [Point(x, y) for x, y in zip(x_coords, y_coords)],
                crs=ortho_crs  # 确保点云坐标系与投影后的多边形一致
            )

            inside_poly = points_geoseries.within(original_poly_proj)
            mask_poly = np.array(inside_poly)  # 多边形内点的掩码（True=保留）
            poly_count = np.sum(mask_poly)

            # 筛选矩形内的点（对比用）
            inside_bbox = points_geoseries.within(bbox_geom_proj)
            mask_bbox = np.array(inside_bbox)
            bbox_count = np.sum(mask_bbox)

            print(f"多边形内点数量（原始点序边界）: {poly_count}")
            print(f"矩形内点数量: {bbox_count}")
            print(f"裁剪保留率（多边形）: {poly_count / original_count * 100:.2f}%")

            # --------------------------
            # 保存多边形裁剪点云（保留原始属性）
            # --------------------------
            poly_las_path = os.path.join(output_dir, "cropped_polygon_point_cloud.las")
            with laspy.file.File(poly_las_path, mode="w", header=las.header) as out_las:
                # 按掩码保留点坐标（严格筛选原始点序多边形内的点）
                out_las.x = x_coords[mask_poly]
                out_las.y = y_coords[mask_poly]
                out_las.z = z_coords[mask_poly]

                # 复制所有非坐标属性（确保点云信息完整）
                for attr in las.point_format:
                    attr_name = attr.name
                    if attr_name not in ["x", "y", "z"]:
                        # 按掩码复制属性值
                        setattr(out_las, attr_name, getattr(las, attr_name)[mask_poly])
            print(f"多边形裁剪点云已保存: {poly_las_path}")

            # --------------------------
            # 保存矩形裁剪点云
            # --------------------------
            bbox_las_path = os.path.join(output_dir, "cropped_bbox_point_cloud.las")
            with laspy.file.File(bbox_las_path, mode="w", header=las.header) as out_las:
                out_las.x = x_coords[mask_bbox]
                out_las.y = y_coords[mask_bbox]
                out_las.z = z_coords[mask_bbox]

                # 复制所有非坐标属性
                for attr in las.point_format:
                    attr_name = attr.name
                    if attr_name not in ["x", "y", "z"]:
                        setattr(out_las, attr_name, getattr(las, attr_name)[mask_bbox])
            print(f"矩形裁剪点云已保存: {bbox_las_path}")

        # --------------------------
        # 步骤4: 可视化验证（显示原始点序、裁剪结果）
        # --------------------------
        print("生成可视化结果（验证点序和裁剪边界）...")
        fig, axes = plt.subplots(1, 2, figsize=(20, 10))

        # 子图1：多边形裁剪结果 + 原始点序标记
        ax1 = axes[0]
        # 显示裁剪后的影像（若为多波段，取第一波段；若为RGB，需调整维度）
        if out_image_poly.shape[0] == 1:
            ax1.imshow(out_image_poly[0], cmap="gray")  # 单波段灰度显示
        else:
            ax1.imshow(np.transpose(out_image_poly, (1, 2, 0)))  # 多波段RGB显示（需确认波段顺序）

        # 将投影后的多边形点转换为影像像素坐标（用于叠加显示）
        proj_points = list(original_poly_proj.exterior.coords)[:-1]  # 去除最后一个闭合点
        # 地理坐标转影像像素坐标（rasterio.transform.rowcol）
        pixel_coords = [
            rasterio.transform.rowcol(out_transform_poly, x, y)
            for x, y in proj_points
        ]
        pixel_x = [p[1] for p in pixel_coords]  # 像素列号（对应x轴）
        pixel_y = [p[0] for p in pixel_coords]  # 像素行号（对应y轴，影像y轴向下）

        # 绘制多边形边界（按原始点序连接）
        ax1.plot(pixel_x, pixel_y, color="red", linewidth=2, label="原始点序边界")
        # 绘制点标记（按原始顺序编号）
        for i, (px, py) in enumerate(zip(pixel_x, pixel_y)):
            ax1.scatter(px, py, color="blue", s=50, zorder=3)  # 点标记
            ax1.text(px + 2, py + 2, str(i + 1), color="white", fontsize=10, zorder=4)  # 点序号

        ax1.set_title("多边形裁剪结果（严格按原始点序）", fontsize=14)
        ax1.axis("off")
        ax1.legend()

        # 子图2：矩形裁剪结果（对比用）
        ax2 = axes[1]
        if out_image_bbox.shape[0] == 1:
            ax2.imshow(out_image_bbox[0], cmap="gray")
        else:
            ax2.imshow(np.transpose(out_image_bbox, (1, 2, 0)))

        # 叠加矩形边界（绿色）
        bbox_points = list(bbox_geom_proj.exterior.coords)[:-1]
        bbox_pixel_coords = [
            rasterio.transform.rowcol(out_transform_bbox, x, y)
            for x, y in bbox_points
        ]
        bbox_px = [p[1] for p in bbox_pixel_coords]
        bbox_py = [p[0] for p in bbox_pixel_coords]
        ax2.plot(bbox_px, bbox_py, color="green", linewidth=2, label="最小矩形边界")

        ax2.set_title("最小矩形裁剪结果", fontsize=14)
        ax2.axis("off")
        ax2.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "crop_result_comparison.png"), dpi=300, bbox_inches="tight")
        # plt.show()
        print(f"可视化结果已保存: {os.path.join(output_dir, 'crop_result_comparison.png')}")

        return True

    except Exception as e:
        print(f"处理过程中发生错误: {str(e)}")
        return False


# 主程序
if __name__ == "__main__":
    # 文件路径设置
    shapefile_path = r"I:\baisha\gcp0509\ordered_polygon.shp"  # 原始点序多边形
    ortho_path = r"I:\baisha\2025.8.30\baisha_kjg_20m_0830\3_dsm_ortho\2_mosaic\baisha_kjg_20m_0830_transparent_mosaic_group1.tif"
    las_path = r"I:\baisha\2025.8.30\baisha_kjg_20m_0830\2_densification\point_cloud\baisha_kjg_20m_0830_group1_densified_point_cloud.las"
    output_dir = r"I:\baisha\2025.8.30\cailiao\Visible\caijian_1"

    # 执行裁剪（严格遵守点序）
    success = crop_ortho_and_las(shapefile_path, ortho_path, las_path, output_dir)
    if success:
        print("所有裁剪任务完成，结果已保存至输出目录！")
    else:
        print("裁剪任务失败，请检查错误信息！")