import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import csv
import random
from tqdm import tqdm
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
import torch.nn.functional as F

# --- 0. 全局常量 ---
MIN_AREA_THRESHOLD = 50


# --- 1. 工具函数：加载配置、Loss、指标计算 ---
def load_config(config_path="config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


class DiceCELoss(nn.Module):
    def __init__(self, sigmoid=True, squared_pred=True, reduction="mean"):
        super().__init__()
        self.sigmoid = sigmoid
        self.reduction = reduction
        self.squared_pred = squared_pred

    def forward(self, inputs, targets):
        if self.sigmoid:
            inputs = torch.sigmoid(inputs)
        targets = targets.float()
        bce = F.binary_cross_entropy(inputs, targets, reduction=self.reduction)
        smooth = 1e-5
        intersection = (inputs * targets).sum(dim=(2, 3))
        if self.squared_pred:
            union = (inputs ** 2).sum(dim=(2, 3)) + (targets ** 2).sum(dim=(2, 3))
        else:
            union = inputs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
        dice = 1 - (2 * intersection + smooth) / (union + smooth)
        return bce + dice.mean()


def compute_metrics(preds, targets):
    preds = (preds > 0.5).long()
    targets = targets.long()
    TP = (preds * targets).sum(dim=(1, 2))
    FP = (preds * (1 - targets)).sum(dim=(1, 2))
    FN = ((1 - preds) * targets).sum(dim=(1, 2))
    epsilon = 1e-6
    iou = (TP + epsilon) / (TP + FP + FN + epsilon)
    precision = (TP + epsilon) / (TP + FP + epsilon)
    recall = (TP + epsilon) / (TP + FN + epsilon)
    return iou.mean().item(), precision.mean().item(), recall.mean().item()


# --- 2. 数据集类 ---
class SAMDataset(Dataset):
    def __init__(self, root_dir, mode='train', target_size=1024):
        self.root_dir = root_dir
        self.mode = mode
        self.target_size = target_size
        self.images_dir = os.path.join(root_dir, mode, 'images')
        self.masks_dir = os.path.join(root_dir, mode, 'masks')
        self.image_files = sorted([f for f in os.listdir(self.images_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        self.transform = ResizeLongestSide(target_size)
        self.pixel_mean = torch.tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
        self.pixel_std = torch.tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)

        # 预先扫描所有掩码，展开所有实例
        self.samples = []
        print(f"正在扫描 {mode} 数据集，提取所有图像中的所有目标实例...")
        for img_name in tqdm(self.image_files, desc=f"Scanning {mode} instances"):
            mask_path = os.path.join(self.masks_dir, img_name)
            mask = cv2.imread(mask_path, 0)
            if mask is None:
                continue
            _, mask = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)

            mask_1024 = cv2.resize(mask, (self.target_size, self.target_size), interpolation=cv2.INTER_NEAREST)
            num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask_1024, connectivity=8)

            for i in range(1, num_labels):
                if stats[i, cv2.CC_STAT_AREA] >= MIN_AREA_THRESHOLD:
                    self.samples.append((img_name, i))

        print(f"[{mode} 数据集] 扫描完毕！共 {len(self.image_files)} 张图像，提取到 {len(self.samples)} 个独立训练目标。")

    def __len__(self):
        # 长度是所有目标的总数
        return len(self.samples)

    def preprocess_image(self, img_tensor):
        x = (img_tensor - self.pixel_mean) / self.pixel_std
        h, w = x.shape[-2:]
        padh = self.target_size - h
        padw = self.target_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    def __getitem__(self, idx):
        # 通过 idx 获取具体的图像和指定的连通域 ID
        img_name, target_idx = self.samples[idx]

        image = cv2.imread(os.path.join(self.images_dir, img_name))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(os.path.join(self.masks_dir, img_name), 0)
        _, mask = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)

        orig_h, orig_w = image.shape[:2]
        image_1024 = self.transform.apply_image(image)
        mask_1024 = cv2.resize(mask, (image_1024.shape[1], image_1024.shape[0]), interpolation=cv2.INTER_NEAREST)

        image_tensor = torch.as_tensor(image_1024.transpose(2, 0, 1)).float()
        mask_1024_np = mask_1024.astype(np.uint8)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_1024_np, connectivity=8)

        # 仅针对当前选定的目标生成掩码
        instance_mask = (labels == target_idx).astype(np.float32)

        x = stats[target_idx, cv2.CC_STAT_LEFT]
        y = stats[target_idx, cv2.CC_STAT_TOP]
        w = stats[target_idx, cv2.CC_STAT_WIDTH]
        h = stats[target_idx, cv2.CC_STAT_HEIGHT]

        jitter = 5
        x1 = max(0, x - random.randint(0, jitter))
        y1 = max(0, y - random.randint(0, jitter))
        x2 = min(1023, x + w + random.randint(0, jitter))
        y2 = min(1023, y + h + random.randint(0, jitter))
        box = np.array([x1, y1, x2, y2])

        # 生成内部正向点
        cx = np.clip(int(centroids[target_idx, 0]), 0, 1023)
        cy = np.clip(int(centroids[target_idx, 1]), 0, 1023)

        dynamic_offset = int(min(w, h) * 0.1)
        offset = max(5, min(dynamic_offset, 20))

        neg_pts = np.array([
            [max(0, x1 - offset), max(0, y1 - offset)],
            [min(1023, x2 + offset), max(0, y1 - offset)],
            [max(0, x1 - offset), min(1023, y2 + offset)],
            [min(1023, x2 + offset), min(1023, y2 + offset)]
        ])

        point_coords = np.vstack([[cx, cy], neg_pts])
        point_labels = np.array([1, 0, 0, 0, 0])  # 1个正向点，4个负向点

        image_tensor = self.preprocess_image(image_tensor)

        return {
            "image": image_tensor,
            "mask": torch.as_tensor(instance_mask).unsqueeze(0).float(),
            "box": torch.tensor(box).float(),
            "point_coords": torch.tensor(point_coords).float(),
            "point_labels": torch.tensor(point_labels).long(),
            "original_size": torch.tensor([orig_h, orig_w]).long()
        }


# --- 3. 可视化工具函数 ---
def denormalize_and_depad(tensor, h_orig, w_orig):
    pixel_mean = torch.tensor([123.675, 116.28, 103.53]).view(-1, 1, 1).to(tensor.device)
    pixel_std = torch.tensor([58.395, 57.12, 57.375]).view(-1, 1, 1).to(tensor.device)
    denorm_tensor = tensor * pixel_std + pixel_mean
    denorm_tensor = denorm_tensor[:, :h_orig, :w_orig]
    return denorm_tensor.permute(1, 2, 0).cpu().numpy().astype(np.uint8)


def save_visualization_sample(image_tensor, gt_mask_tensor, pred_mask_tensor, box_tensor, point_coords_tensor,
                              point_labels_tensor, h_orig, w_orig, vis_dir, epoch, sample_idx):
    os.makedirs(vis_dir, exist_ok=True)
    img_np = denormalize_and_depad(image_tensor, h_orig, w_orig)
    gt_mask_np = (gt_mask_tensor[:h_orig, :w_orig].cpu().numpy() * 255).astype(np.uint8)
    pred_mask_np = (pred_mask_tensor[:h_orig, :w_orig].cpu().numpy() * 255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    gt_overlay = np.zeros_like(img_bgr)
    gt_overlay[gt_mask_np == 255] = [255, 255, 0]
    img_gt_vis = cv2.addWeighted(img_bgr, 0.5, gt_overlay, 0.5, 0)

    pred_overlay = np.zeros_like(img_bgr)
    pred_overlay[pred_mask_np == 255] = [0, 0, 255]
    img_pred_vis = cv2.addWeighted(img_bgr, 0.5, pred_overlay, 0.5, 0)

    scale_factor_x = w_orig / 1024.0
    scale_factor_y = h_orig / 1024.0

    # 绘制边界框 (蓝色)
    box = box_tensor.cpu().numpy().astype(int)
    bx1, by1 = int(box[0] * scale_factor_x), int(box[1] * scale_factor_y)
    bx2, by2 = int(box[2] * scale_factor_x), int(box[3] * scale_factor_y)
    cv2.rectangle(img_pred_vis, (bx1, by1), (bx2, by2), (255, 0, 0), 2)

    # 绘制提示点
    points = point_coords_tensor.cpu().numpy()
    labels = point_labels_tensor.cpu().numpy()
    for pt, lbl in zip(points, labels):
        pt_x, pt_y = int(pt[0] * scale_factor_x), int(pt[1] * scale_factor_y)
        color = (0, 255, 0) if lbl == 1 else (0, 0, 255)  # 正向点绿色，负向点红色
        cv2.circle(img_pred_vis, (pt_x, pt_y), 5, color, -1)

    composite_image = np.hstack([img_bgr, img_gt_vis, img_pred_vis])
    cv2.imwrite(os.path.join(vis_dir, f"epoch_{epoch:02d}_sample_{sample_idx}.png"), composite_image)


# --- 4. 验证函数 ---
def validate(sam, val_loader, device, criterion, cfg, epoch):
    sam.eval()
    total_loss, total_iou, total_p, total_r, num_batches = 0, 0, 0, 0, 0

    # 在验证循环开始前，随机挑选用于可视化的批次索引
    vis_dir = cfg['other'].get('visualization_dir')
    vis_interval = cfg['other'].get('visualization_interval', 5)
    vis_count = cfg['other'].get('visualization_samples', 3)

    do_visualization = vis_dir and (epoch % vis_interval == 0)
    vis_indices = set()

    if do_visualization:
        num_val_samples = len(val_loader)
        # 从 0 到 num_val_samples-1 中随机抽取 vis_count 个不重复的索引
        vis_indices = set(random.sample(range(num_val_samples), min(vis_count, num_val_samples)))
        print(f"  - 随机挑选了 {len(vis_indices)} 个目标进行可视化展示...")

    with torch.no_grad():
        for i, batch in enumerate(tqdm(val_loader, desc="Validating")):
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            boxes = batch['box'].to(device)
            point_coords = batch['point_coords'].to(device)
            point_labels = batch['point_labels'].to(device)

            # 1. 批量提取图像特征
            image_embeddings = sam.image_encoder(images)

            pred_masks_list = []

            # 2. 遍历当前批次单独解码
            for b_idx in range(images.shape[0]):
                curr_embedding = image_embeddings[b_idx].unsqueeze(0)
                curr_points = (
                    point_coords[b_idx].unsqueeze(0),
                    point_labels[b_idx].unsqueeze(0)
                )
                curr_boxes = boxes[b_idx].unsqueeze(0)

                sparse_embeddings, dense_embeddings = sam.prompt_encoder(
                    points=curr_points,
                    boxes=curr_boxes,
                    masks=None
                )
                image_pe_input = sam.prompt_encoder.get_dense_pe().to(device)

                low_res_masks, _ = sam.mask_decoder(
                    image_embeddings=curr_embedding,
                    image_pe=image_pe_input,
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False,
                )
                pred_masks_list.append(low_res_masks)

            # 3. 重新拼装
            low_res_masks_batch = torch.cat(pred_masks_list, dim=0)
            upscaled_masks = F.interpolate(low_res_masks_batch, size=(1024, 1024), mode="bilinear", align_corners=False)

            loss = criterion(upscaled_masks, masks)
            total_loss += loss.item()

            iou, precision, recall = compute_metrics(upscaled_masks[0:1], masks[0:1])
            total_iou += iou
            total_p += precision
            total_r += recall
            num_batches += 1

            # 只对选中的随机索引执行可视化
            if do_visualization and (i in vis_indices):
                h_orig, w_orig = batch['original_size'][0].tolist()
                save_visualization_sample(
                    images[0], masks[0, 0], (upscaled_masks[0, 0] > 0.5).float(),
                    boxes[0], point_coords[0], point_labels[0], h_orig, w_orig, vis_dir, epoch, i
                )

    return total_loss / num_batches, total_iou / num_batches, total_p / num_batches, total_r / num_batches


# --- 5. 训练主逻辑 ---
def train():
    cfg = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    TARGET_EFFECTIVE_BATCH_SIZE = cfg['training']['target_effective_batch_size']
    PHYSICAL_BATCH_SIZE = cfg['training']['batch_size']
    ACCUMULATION_STEPS = TARGET_EFFECTIVE_BATCH_SIZE // PHYSICAL_BATCH_SIZE

    os.makedirs(cfg['other']['output_dir'], exist_ok=True)
    sam = sam_model_registry[cfg['model']['type']](checkpoint=cfg['model']['checkpoint_path']).to(device)

    if cfg['model']['freeze_image_encoder']:
        for param in sam.image_encoder.parameters(): param.requires_grad = False
        for param in sam.prompt_encoder.parameters(): param.requires_grad = False

    optimizer = optim.AdamW(sam.mask_decoder.parameters(), lr=cfg['training']['learning_rate'],
                            weight_decay=cfg['training']['weight_decay'])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=cfg['training']['lr_scheduler_step_size'],
                                          gamma=cfg['training']['lr_scheduler_gamma'])
    criterion = DiceCELoss()

    train_loader = DataLoader(SAMDataset(cfg['data']['root_dir'], 'train', cfg['data']['image_size']),
                              batch_size=PHYSICAL_BATCH_SIZE, shuffle=True, num_workers=cfg['training']['num_workers'])
    val_loader = DataLoader(SAMDataset(cfg['data']['root_dir'], 'val', cfg['data']['image_size']), batch_size=1,
                            shuffle=False, num_workers=cfg['training']['num_workers'])

    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(1, cfg['training']['epochs'] + 1):
        sam.train()
        if cfg['model']['freeze_image_encoder']:
            sam.image_encoder.eval()
            sam.prompt_encoder.eval()

        epoch_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg['training']['epochs']}")

        for i, batch in enumerate(pbar):
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            boxes = batch['box'].to(device)
            point_coords = batch['point_coords'].to(device)
            point_labels = batch['point_labels'].to(device)

            with torch.no_grad():
                # 1. 批量提取图像特征 (保持高效)
                image_embeddings = sam.image_encoder(images)

            # 初始化一个列表，用于收集当前批次内独立解码的结果
            pred_masks_list = []

            # 2. 遍历当前批次中的每一张图像单独进行解码
            for b_idx in range(images.shape[0]):
                # 提取单张图像的数据并增加 batch 维度保持格式 (1, ...)
                curr_embedding = image_embeddings[b_idx].unsqueeze(0)
                curr_points = (
                    point_coords[b_idx].unsqueeze(0),
                    point_labels[b_idx].unsqueeze(0)
                )
                curr_boxes = boxes[b_idx].unsqueeze(0)

                with torch.no_grad():
                    sparse_embeddings, dense_embeddings = sam.prompt_encoder(
                        points=curr_points,
                        boxes=curr_boxes,
                        masks=None
                    )
                    image_pe_input = sam.prompt_encoder.get_dense_pe().to(device)

                low_res_masks, _ = sam.mask_decoder(
                    image_embeddings=curr_embedding,
                    image_pe=image_pe_input,
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False,
                )
                pred_masks_list.append(low_res_masks)

            # 3. 将单张解码的结果重新拼装成 batch 维度 (B, 1, 256, 256)
            low_res_masks_batch = torch.cat(pred_masks_list, dim=0)
            upscaled_masks = F.interpolate(low_res_masks_batch, size=(1024, 1024), mode="bilinear", align_corners=False)

            loss = criterion(upscaled_masks, masks) / ACCUMULATION_STEPS
            loss.backward()

            if (i + 1) % ACCUMULATION_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()

            epoch_loss += (loss.item() * ACCUMULATION_STEPS)
            pbar.set_postfix(loss=(loss.item() * ACCUMULATION_STEPS))

        if (i + 1) % ACCUMULATION_STEPS != 0:
            optimizer.step()
            optimizer.zero_grad()

        scheduler.step()

        sam.eval()
        val_loss, val_iou, val_p, val_r = validate(sam, val_loader, device, criterion, cfg, epoch)
        print(
            f"Epoch {epoch} | Train Loss: {epoch_loss / len(train_loader):.4f} | Val Loss: {val_loss:.4f} | mIoU: {val_iou:.4f}")

        if val_loss < best_loss - cfg['other'].get('early_stopping_min_delta', 0.0001):
            best_loss = val_loss
            patience_counter = 0
            torch.save(sam.mask_decoder.state_dict(), os.path.join(cfg['other']['output_dir'], "best_sam_decoder.pth"))
        else:
            patience_counter += 1
            if patience_counter >= cfg['other'].get('early_stopping_patience', 10):
                print("Early stopping triggered!")
                break

        save_interval = cfg['other'].get('save_checkpoint_interval', 0)
        # 确保配置了大于0的间隔才执行定期保存
        if save_interval > 0 and epoch % save_interval == 0:
            checkpoint_filename = f"sam_decoder_epoch_{epoch:03d}.pth"
            checkpoint_path = os.path.join(cfg['other']['output_dir'], checkpoint_filename)
            torch.save(sam.mask_decoder.state_dict(), checkpoint_path)
            print(f"  - Checkpoint saved: {checkpoint_path}")


if __name__ == "__main__":
    train()