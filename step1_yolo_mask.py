import sys
import cv2
import torch
import numpy as np
import os
from ultralytics import YOLO


# --- 核心逻辑：生成全图分辨率的单目标掩码 ---
def save_mask(shape, box, output_path):
    """
    shape: (height, width) 原图尺寸
    box: [x1, y1, x2, y2] 坐标
    output_path: 保存路径
    """
    mask = np.zeros(shape[:2], dtype=np.uint8)
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
    cv2.imwrite(output_path, mask)


def cluster_merge(detections, dist_threshold=30):
    if len(detections) <= 1: return detections
    detections = detections[detections[:, 4].argsort()[::-1]]
    centers = np.stack([(detections[:, 0] + detections[:, 2]) / 2, (detections[:, 1] + detections[:, 3]) / 2], axis=1)
    keep, removed = [], set()
    for i in range(len(detections)):
        if i in removed: continue
        keep.append(i)
        dists = np.linalg.norm(centers[i] - centers[i + 1:], axis=1)
        close_indices = np.where(dists < dist_threshold)[0] + i + 1
        for idx in close_indices: removed.add(idx)
    return detections[keep]


def run_tiled_inference_with_masks(img_path, model_path, output_dir):
    # --- 参数配置 ---
    tile_size = 1280
    overlap = 320
    conf_thres = 0.45
    merge_dist = 100

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = YOLO(model_path)
    img = cv2.imread(img_path)
    if img is None:
        print(f"无法读取图片: {img_path}")
        return

    h, w, _ = img.shape
    stride = tile_size - overlap
    all_detections = []

    # --- 切片推理逻辑 ---
    y_starts = np.append(np.arange(0, h - tile_size, stride), h - tile_size)
    x_starts = np.append(np.arange(0, w - tile_size, stride), w - tile_size)
    y_starts = np.unique(y_starts[y_starts >= 0])
    x_starts = np.unique(x_starts[x_starts >= 0])

    print("开始切片推理...")
    for y1 in y_starts:
        for x1 in x_starts:
            y1, x1 = int(y1), int(x1)
            tile = img[y1:y1 + tile_size, x1:x1 + tile_size]
            results = model.predict(tile, imgsz=tile_size, conf=conf_thres, device=device, half=True, verbose=False)

            for r in results:
                if len(r.boxes) > 0:
                    temp_det = r.boxes.data.cpu().numpy().copy()
                    temp_det[:, 0] += x1
                    temp_det[:, 1] += y1
                    temp_det[:, 2] += x1
                    temp_det[:, 3] += y1
                    all_detections.append(temp_det)

    if not all_detections:
        print("未检测到任何目标。")
        return

    # --- 执行中心点合并 ---
    all_dets_np = np.concatenate(all_detections, axis=0)
    final_dets = cluster_merge(all_dets_np, dist_threshold=merge_dist)
    print(f"最终检测到 {len(final_dets)} 棵树。")

    os.makedirs(output_dir, exist_ok=True)

    # ==============================================================================
    # 在原图上绘制检测框（红色，加粗）
    # ==============================================================================
    print("开始绘制可视化检测结果...")
    vis_img = img.copy()  # 复制原图

    for d in final_dets:
        x1, y1, x2, y2, conf = d[:5]

        # 将浮点数坐标转换为整数
        pt1 = (int(x1), int(y1))
        pt2 = (int(x2), int(y2))

        cv2.rectangle(vis_img, pt1, pt2, (0, 0, 255), 5)

        label = f"{conf:.2f}"
        cv2.putText(vis_img, label, (pt1[0], pt1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    # 保存整张带框的图片
    vis_save_path = os.path.join(output_dir, "yolo_preview.png")
    cv2.imwrite(vis_save_path, vis_img)
    print(f"可视化检测图已保存至: {vis_save_path}")
    # ==============================================================================

    # --- 掩码保存目录 ---
    mask_dir = os.path.join(output_dir, "masks")
    os.makedirs(mask_dir, exist_ok=True)

    print("开始生成独立掩码...")
    # --- 逐个生成掩码 ---
    for i, d in enumerate(final_dets):
        box = d[:4]
        mask_name = f"tree_{i:04d}.png"
        mask_path = os.path.join(mask_dir, mask_name)

        save_mask((h, w), box, mask_path)

        if (i + 1) % 100 == 0 or (i + 1) == len(final_dets):
            print(f"进度: 已生成 {i + 1}/{len(final_dets)} 个掩码")

    print(f"所有掩码已保存至: {mask_dir}")


if __name__ == "__main__":
    # 请确保以下路径正确
    run_tiled_inference_with_masks(
        r"I:\pinghemiyou\PH.png",
        r"I:\experiment\ultralytics-yolo11-main\runs\train-shumu\exp2\weights\best.pt",
        r"I:\pinghemiyou\result"
    )