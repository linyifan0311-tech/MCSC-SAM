# import library
import shapefile
import geopandas
import laspy
from scipy import spatial
import whitebox
import osgeo.ogr as ogr
import osgeo.osr as osr
import rasterio
import rasterio.plot as rplt
import open3d as o3d

from skimage import io
import numpy as np
from matplotlib import pyplot as plt
import os
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimHei"] # 设置字体为黑体
plt.rcParams["axes.unicode_minus"] = False # 正常显示负号

# 从给定的图像路径和RTK信息路径中提取目标区域，按点的顺序生成多边形形状文件（.shp）
def make_ordered_polygon(image_path, RTK_information_path):
    ## 获取坐标参考系统（CRS）的 EPSG 代码
    with rasterio.open(image_path) as src:
        EPSG = src.crs.to_epsg()

    ## 读取 RTK 信息文件并保持点的顺序
    sf = shapefile.Reader(RTK_information_path)
    shapes = sf.shapes()

    list_p = []
    # 按RTK文件中的顺序收集所有点的坐标
    for i, shape in enumerate(shapes):
        # 取每个形状的第一个点，并记录原始索引
        point = shape.points[0]
        list_p.append((i, point[0], point[1]))  # (原始索引, x, y)
        print(f"点 {i + 1}: 原始索引 {i}, 坐标 ({point[0]}, {point[1]})")

    # 检查是否有足够的点来创建多边形（至少需要3个点）
    if len(list_p) < 3:
        raise ValueError(f"生成多边形至少需要3个点，但只找到了{len(list_p)}个点")

    rtk_dir, rtk_name = os.path.split(RTK_information_path)

    ## 创建多边形形状文件（.shp）
    driver = ogr.GetDriverByName("ESRI Shapefile")
    poly_path = os.path.join(rtk_dir, 'ordered_polygon.shp')
    # 如果文件已存在则删除
    if os.path.exists(poly_path):
        driver.DeleteDataSource(poly_path)

    data_source = driver.CreateDataSource(poly_path)

    # 创建空间参考
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(EPSG)

    # 创建多边形图层
    layer = data_source.CreateLayer("ordered_polygon", srs, ogr.wkbMultiPolygon)

    # 添加字段存储点的顺序信息（可选）
    order_field = ogr.FieldDefn("PointOrder", ogr.OFTString)
    order_field.SetWidth(100)
    layer.CreateField(order_field)

    feature = ogr.Feature(layer.GetLayerDefn())

    ## 严格按照点的原始顺序构建WKT字符串
    # 提取点坐标（保持原始顺序）
    points_str = []
    point_order = []
    for idx, x, y in list_p:
        points_str.append(f"{x} {y}")
        point_order.append(str(idx))  # 记录点的原始顺序

    # 闭合多边形 - 添加第一个点作为最后一个点，完成闭合
    points_str.append(f"{list_p[0][1]} {list_p[0][2]}")

    # 组合成完整的WKT
    wkt = f"POLYGON(({','.join(points_str)}))"
    print(f"\n生成的多边形WKT (按点顺序连接): {wkt}")
    print(f"点连接顺序: {', '.join(point_order)} -> {point_order[0]} (闭合)")

    # 创建多边形几何对象
    polygon = ogr.CreateGeometryFromWkt(wkt)

    # 设置要素几何并添加到图层
    feature.SetGeometry(polygon)
    # 存储点的顺序信息
    feature.SetField("PointOrder", ", ".join(point_order))
    layer.CreateFeature(feature)

    # 清理资源
    feature = None
    data_source = None

    ## 可视化绘制 - 显示点的顺序和连接关系
    sf_m = geopandas.read_file(RTK_information_path)
    sf_po = geopandas.read_file(poly_path)

    fig, axes = plt.subplots(1, 2, figsize=(12, 10))

    ax = axes.ravel()

    # 第一个子图：显示点及其顺序编号
    ax[0] = sf_m.plot(color='red', markersize=50, ax=ax[0])
    ax[0].set_title('ROI Points with Order')
    ax[0].set_aspect('equal')

    # 在点旁标注顺序编号
    for i, (orig_idx, x, y) in enumerate(list_p):
        ax[0].text(x, y, f"{i + 1}", fontsize=12, color='blue')

    # 第二个子图：显示多边形及连接顺序
    ax[1] = sf_po.boundary.plot(color="red", zorder=10, ax=ax[1])
    ax[1] = sf_m.plot(color='red', markersize=30, ax=ax[1], alpha=0.7)  # 叠加显示点
    ax[1].set_title('ROI Polygon (按点顺序连接)')
    ax[1].set_aspect('equal')

    # 在多边形上标注点的顺序
    for i, (orig_idx, x, y) in enumerate(list_p):
        ax[1].text(x, y, f"{i + 1}", fontsize=12, color='blue')

    # # 绘制箭头表示连接方向
    # for i in range(len(list_p)):
    #     x1, y1 = list_p[i][1], list_p[i][2]
    #     if i < len(list_p) - 1:
    #         x2, y2 = list_p[i + 1][1], list_p[i + 1][2]
    #     else:
    #         # 最后一个点连接回第一个点
    #         x2, y2 = list_p[0][1], list_p[0][2]
    #
    #     # 绘制箭头表示连接方向
    #     ax[1].arrow(x1, y1, x2 - x1, y2 - y1,
    #                 head_width=5, head_length=10,
    #                 fc='green', ec='green',
    #                 length_includes_head=True,
    #                 alpha=0.6)

    plt.tight_layout()
    plt.show()

    return poly_path, list_p


# 冠层高度模型（CHM）数据可视化
def CHM_Visualization(chm_path):
    img = io.imread(chm_path)

    fig, axes = plt.subplots(1, 2, figsize=(10, 10))

    axes[0].imshow(img, cmap=plt.cm.gray)
    axes[0].set_title('img')
    axes[1].imshow(img, cmap=plt.cm.jet)
    axes[1].set_title('img(jet)')


# 主程序
if __name__ == "__main__":
    # 设置图像和RTK文件路径
    image_path = r"I:\baisha\2025.8.30\ground2\白沙830二号地可见光-20m\3_dsm_ortho\2_mosaic\白沙830二号地可见光-20m_transparent_mosaic_group1.tif"
    RTK_information_path = r"I:\baisha\2025.8.30\ground2\RTK\ground2_0830_gcp_computed.shp"

    try:
        poly_path, list_p = make_ordered_polygon(image_path, RTK_information_path)
        print(f"\n生成的多边形文件路径: {poly_path}")
        print(f"使用的点数量: {len(list_p)}")
        print(f"点连接顺序: 1 -> 2 -> ... -> {len(list_p)} -> 1 (闭合)")
    except Exception as e:
        print(f"发生错误: {e}")
