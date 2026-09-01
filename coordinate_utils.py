import laspy
import numpy as np
import rasterio
from rasterio.transform import rowcol
import cv2
import os
from sklearn.cluster import DBSCAN
from typing import Tuple, Optional


class CoordinateProcessor:
    @staticmethod
    def read_las_points(las_path: str) -> np.ndarray:
        """读取LAS/LAZ点云文件的坐标信息(x, y, z)"""
        if not os.path.exists(las_path):
            raise FileNotFoundError(f"点云文件不存在: {las_path}")

        with laspy.open(las_path) as f:
            points = f.read()
            return np.column_stack((points.x, points.y, points.z))

    @staticmethod
    def cluster_tree_points(points: np.ndarray, eps: float = 1.5, min_samples: int = 5) -> np.ndarray:
        """使用DBSCAN聚类单棵树的点云，返回每类的中心点作为提示点"""
        if len(points) < min_samples:
            return np.array([points.mean(axis=0)])  # 点太少时直接取均值

        # DBSCAN聚类（根据树的间距调整eps参数）
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(points[:, :2])  # 仅用x,y坐标聚类

        # 计算每个聚类的中心点
        cluster_centers = []
        for label in np.unique(labels):
            if label == -1:  # 忽略噪声点
                continue
            cluster_points = points[labels == label]
            cluster_center = cluster_points.mean(axis=0)  # 中心点坐标
            cluster_centers.append(cluster_center)

        return np.array(cluster_centers)

    @staticmethod
    def read_las_folder(folder_path: str) -> np.ndarray:
        """读取文件夹中所有LAS/LAZ文件，每个文件聚类为一个提示点"""
        all_centers = []
        for file in os.listdir(folder_path):
            if file.lower().endswith(('.las', '.laz')):
                file_path = os.path.join(folder_path, file)
                points = CoordinateProcessor.read_las_points(file_path)
                # 单个文件直接取所有点的中心点作为提示点
                center = points.mean(axis=0)
                all_centers.append(center)

        if not all_centers:
            raise ValueError(f"文件夹中未找到点云文件: {folder_path}")
        return np.vstack(all_centers)

    @staticmethod
    def read_las_file_with_clustering(las_path: str) -> np.ndarray:
        """读取单个LAS/LAZ文件，使用DBSCAN聚类提取每棵树的提示点"""
        points = CoordinateProcessor.read_las_points(las_path)
        return CoordinateProcessor.cluster_tree_points(points)

    @staticmethod
    def las_to_ortho_coords(
            las_points: np.ndarray,
            ortho_path: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """将点云地理坐标(x, y)转换为正射图像素坐标(col, row)"""
        with rasterio.open(ortho_path) as src:
            transform = src.transform  # 地理坐标到像素坐标的转换矩阵
            las_xy = las_points[:, :2]  # 提取x, y坐标

            # 转换为像素坐标(row, col)
            rows, cols = rowcol(transform, las_xy[:, 0], las_xy[:, 1])

            # 过滤超出正射图范围的点
            valid_mask = (
                    (rows >= 0) & (rows < src.height) &
                    (cols >= 0) & (cols < src.width)
            )

            # 返回像素坐标(col, row)和有效点云原始坐标
            return np.column_stack((cols[valid_mask], rows[valid_mask])), las_points[valid_mask]

    @staticmethod
    def visualize_prompt_points(
            ortho_img: np.ndarray,  # 正射影像 (BGR, 1024x1024)
            pixel_coords: np.ndarray,  # 像素坐标 (N, 2)
            save_path: str,
            color: Tuple[int, int, int] = (0, 255, 0),  # BGR 格式，默认红色
            point_size: int = 15
    ):
        """
        在图像上绘制提示点并保存。
        pixel_coords 是 (col, row) 格式。
        """
        vis_img = ortho_img.copy()

        # 确保坐标是整数
        pixel_coords = pixel_coords.astype(int)

        for (col, row) in pixel_coords:
            # 检查边界
            if 0 <= row < vis_img.shape[0] and 0 <= col < vis_img.shape[1]:
                # 在 (col, row) 位置绘制圆点
                cv2.circle(vis_img, (int(col), int(row)), point_size, color, -1)

        # 确保保存目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, vis_img)
        print(f"提示点可视化结果已保存至: {save_path}")

    # 修正方法名：确保方法名正确定义为ortho_to_png
    @staticmethod
    def ortho_to_png(
            ortho_path: str,
            save_png_path: str,
            scale: float = 1.0
    ) -> Tuple[np.ndarray, float]:
        """将正射图(TIF)转换为PNG并按比例缩放"""
        with rasterio.open(ortho_path) as src:
            # 读取正射图数据
            ortho_data = src.read()  # (C, H, W)
            # 转换为(H, W, C)并转成BGR格式（OpenCV默认）
            ortho_img = np.transpose(ortho_data, (1, 2, 0))
            if ortho_img.shape[-1] == 4:  # 处理RGBA
                ortho_img = cv2.cvtColor(ortho_img, cv2.COLOR_RGBA2BGR)
            else:  # 处理RGB
                ortho_img = cv2.cvtColor(ortho_img, cv2.COLOR_RGB2BGR)

        # 缩放图像
        h, w = ortho_img.shape[:2]
        if scale != 1.0:
            new_w, new_h = int(w * scale), int(h * scale)
            ortho_img_scaled = cv2.resize(ortho_img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        else:
            ortho_img_scaled = ortho_img

        cv2.imwrite(save_png_path, ortho_img_scaled)
        print(f"正射影像已缩放并保存至: {save_png_path}")

        return ortho_img_scaled, w / ortho_img_scaled.shape[1]  # 返回缩放后的图像和缩放比例

    @staticmethod
    def scale_points(points: np.ndarray, scale: float) -> np.ndarray:
        """
        将像素坐标按比例缩放。用于将原始大图坐标缩放到SAM输入的图像尺寸。
        :param points: (N, 2) 的像素坐标数组 (x, y) / (col, row)。
        :param scale: 缩放比例。
        :return: 缩放并四舍五入后的整数坐标数组。
        """
        if points.size == 0:
            return np.array([], dtype=np.int64)

        # 确保是浮点数才能精确乘法
        scaled_coords = points.astype(np.float64) * scale

        # 四舍五入并转为整数
        return np.round(scaled_coords).astype(np.int64)

    @staticmethod
    def visualize_global_points(
            ortho_ref_path: str,  # <-- 直接输入原始 TIF 文件的路径
            las_root_dir: str,
            full_H: int,
            full_W: int,
            save_path: str
    ):
        """
        [Stage 1 & 2 验证]：将点云地理坐标转换为 full_H x full_W 全局像素坐标，并在图像上可视化。
        直接读取 TIF 文件并将其缩放到目标分辨率。
        """
        import glob
        from tqdm import tqdm

        # 0. 读取原始 TIF 图像并缩放到目标尺寸 (Stages 1 & 2 Image Processing)
        print(f"  Reading and scaling TIF from {ortho_ref_path} to {full_H}x{full_W}...")
        try:
            # 使用 rasterio 读取 TIF 文件
            with rasterio.open(ortho_ref_path) as src:
                orig_H_tif, orig_W_tif = src.height, src.width
                # 读取正射图数据 (C, H, W)
                ortho_data = src.read()

            # 转换为 (H, W, C) 并转成BGR格式（OpenCV默认）
            ortho_img = np.transpose(ortho_data, (1, 2, 0))
            if ortho_img.shape[-1] == 4:
                img_bgr = cv2.cvtColor(ortho_img, cv2.COLOR_RGBA2BGR)
            elif ortho_img.shape[-1] == 3:
                img_bgr = cv2.cvtColor(ortho_img, cv2.COLOR_RGB2BGR)
            else:
                raise ValueError("TIF image must have 3 or 4 channels (RGB/RGBA).")

            # 缩放图像到目标尺寸 (Stage 2)
            img_bgr = cv2.resize(img_bgr, (full_W, full_H), interpolation=cv2.INTER_LINEAR)

        except FileNotFoundError:
            raise FileNotFoundError(f"TIF image not found at {ortho_ref_path}.")
        except Exception as e:
            raise RuntimeError(f"Error processing TIF file: {e}")

        # 1. 获取所有点云的地理中心点 (Geo to Geo Centers)
        all_geo_centers = []
        las_files = glob.glob(os.path.join(las_root_dir, '*.las'))
        if not las_files:
            las_files = glob.glob(os.path.join(las_root_dir, '*.laz'))
        if not las_files:
            print("No LAS/LAZ files found.")
            return

        for las_path in tqdm(las_files, desc="Processing LAS files for Global Vis"):
            try:
                points = CoordinateProcessor.read_las_points(las_path)
                centers = CoordinateProcessor.cluster_tree_points(points)
                if centers.size > 0:
                    all_geo_centers.extend(centers[:, :3])
            except Exception as e:
                # 忽略单个文件错误
                # print(f"Error processing {las_path}: {e}")
                pass

        if not all_geo_centers:
            print("No valid tree centers extracted from point cloud folder.")
            return

        geo_centers_np = np.array(all_geo_centers)

        # 2. Geo to Original TIF Pixel (Stage 1)
        # ortho_pixel_coords_orig (N, 2) 是 (col, row) 坐标
        ortho_pixel_coords_orig, _ = CoordinateProcessor.las_to_ortho_coords(
            geo_centers_np,
            ortho_ref_path  # 这里的 las_to_ortho_coords 仍然需要 TIF 文件来获取地理参考信息
        )

        if ortho_pixel_coords_orig.size == 0:
            print("No valid points within the original TIF extent.")
            return

        # 3. 坐标缩放 (Stage 2)
        # TIF 像素坐标 (orig_W_tif, orig_H_tif) 缩放到目标图像 (full_W, full_H)
        scale_x = full_W / orig_W_tif
        scale_y = full_H / orig_H_tif

        global_pixel_coords = ortho_pixel_coords_orig.astype(float)
        global_pixel_coords[:, 0] *= scale_x  # X (列) 缩放
        global_pixel_coords[:, 1] *= scale_y  # Y (行) 缩放

        # 裁剪到 full_H x full_W 范围内并转为整数
        global_pixel_coords = global_pixel_coords.astype(np.int64)
        global_pixel_coords[:, 0] = np.clip(global_pixel_coords[:, 0], 0, full_W - 1)
        global_pixel_coords[:, 1] = np.clip(global_pixel_coords[:, 1], 0, full_H - 1)

        # 4. Visualization
        vis_img = img_bgr.copy()
        point_radius = max(5, int(full_H * 0.001))

        for pt in global_pixel_coords:
            pt_x = pt[0]
            pt_y = pt[1]
            point_color = (0, 0, 255)  # Red (BGR)
            cv2.circle(vis_img, (pt_x, pt_y), point_radius, point_color, -1)

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, vis_img)
        print(f"✅ 全局点云可视化结果已保存至: {save_path}")