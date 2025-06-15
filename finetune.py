# finetune.py
# =============================================================================
# 该脚本完成：
#   1) 从 Kaggle 原始数据：train/ + train_labels.csv → 构建 YOLOv8n/DETR 用的 yolo_dataset/
#   2) 在 yolo_dataset 上对 YOLOv8n 做“较强增强的微调” → 输出 ./yolo_weights_finetune/yolov8n/weights/best.pt
#   3) 在 yolo_dataset 上对 DETR(ResNet-50) 做“较强增强的微调” → 输出 ./detr_weights_finetune/checkpoint-last/
#
# 运行：
#   python finetune.py
#
# 输出示例：
#   ./yolo_dataset/             （生成的数据集目录）
#   ./yolo_weights_finetune/
#       └─ yolov8n/
#            └─ weights/best.pt
#   ./detr_weights_finetune/
#       └─ checkpoint-last/      （HuggingFace DETR 微调权重）
# =============================================================================

import os
import random
import shutil
import yaml
import json
import torch
import numpy as np
from ultralytics import YOLO
from PIL import Image
import pandas as pd
import sys
import timm
from tqdm import tqdm
import warnings
warnings.filterwarnings(
    "ignore",
    message="for .*: copying from a non-meta parameter",
    category=UserWarning
)

# HuggingFace DETR 相关
from transformers import DetrForObjectDetection, DetrImageProcessor
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.transforms import functional as F
import torchvision.transforms as T
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import glob
from torch.optim.lr_scheduler import LambdaLR
from sklearn.model_selection import KFold

# ------------- 1. 全局配置 -------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[DEVICE] Using device: {device}")

# Kaggle 原始目录（仅含 train/, test/, train_labels.csv, sample_submission.csv）
RAW_DATA_DIR     = "./data"
RAW_TRAIN_DIR    = os.path.join(RAW_DATA_DIR, "train")       # 含 tomo_id 子目录
RAW_TEST_DIR     = os.path.join(RAW_DATA_DIR, "test")
RAW_LABELS_CSV   = os.path.join(RAW_DATA_DIR, "train_labels.csv")

# 我们要生成的 yolo_dataset 目录
YOLO_DATASET_DIR = os.path.join(os.getcwd(), "yolo_dataset")

# 微调输出目录
YOLO_FINETUNE_DIR = os.path.join(os.getcwd(), "yolo_weights_finetune")
DETR_FINETUNE_DIR = os.path.join(os.getcwd(), "detr_weights_finetune")

# YOLOv8n 微调用预训练权重 (需提前把 yolov8n.pt 放在工作目录下)
YOLO_PRETRAIN_WEIGHTS = "yolov8n.pt"

# —— YOLOv8n 微调 超参 —— #
YOLO_FINE_EPOCHS    = 50
YOLO_FINE_BATCH     = 16
YOLO_FINE_IMGSZ     = 640
YOLO_FINE_PATIENCE  = 7
YOLO_FINE_LR0       = 0.01

YOLO_FINE_MOSAIC    = 0.7
YOLO_FINE_MIXUP     = 0.3
YOLO_FINE_FLIPLR    = 0.5
YOLO_FINE_FLIPUD    = 0.1
YOLO_FINE_DEGREES   = 10
YOLO_FINE_TRANSLATE = 0.1
YOLO_FINE_SCALE     = 0.3
YOLO_FINE_SHEAR     = 0.0
YOLO_FINE_PERSPECT  = 0.001
YOLO_FINE_HSV_H     = 0.015
YOLO_FINE_HSV_S     = 0.7
YOLO_FINE_HSV_V     = 0.4

# —— DETR 微调 超参 —— #
DETR_FINE_EPOCHS       = 150
DETR_FINE_BATCH        = 16      # 调整为 16
DETR_FINE_LR_BACKBONE  = 1e-5
DETR_FINE_LR_DECODER   = 5e-5
DETR_FINE_LR_HEAD      = 1e-4
DETR_FINE_WEIGHT_DECAY = 1e-4
DETR_FINE_EVAL_EVERY   = 1000
DETR_FINE_PATIENCE     = 30     # 30 轮后开始监控 mAP@0.5

# ------------- 2. 从原始数据构建 yolo_dataset -------------
def build_yolo_dataset(raw_train_dir, labels_csv, out_dir, val_split=0.2, trust=4, box_size=24):
    """
    从 Kaggle 原始的 train/<tomo_id>/<slice>.jpg + train_labels.csv
    生成 yolo_dataset/{images,labels}/{train,val} 结构，以及 dataset.yaml。
    """
    # 1) 读取 CSV，按 tomo_id 分组
    df = pd.read_csv(labels_csv)
    df = df.dropna(subset=["Motor axis 0", "Motor axis 1", "Motor axis 2"])
    tomo_with_motor = df["tomo_id"].unique().tolist()
    random.shuffle(tomo_with_motor)
    split = int(len(tomo_with_motor) * (1 - val_split))
    train_tomos = set(tomo_with_motor[:split])
    val_tomos   = set(tomo_with_motor[split:])

    # 2) 创建目录
    img_tr_dir = os.path.join(out_dir, "images", "train")
    lbl_tr_dir = os.path.join(out_dir, "labels", "train")
    img_va_dir = os.path.join(out_dir, "images", "val")
    lbl_va_dir = os.path.join(out_dir, "labels", "val")
    for d in [img_tr_dir, lbl_tr_dir, img_va_dir, lbl_va_dir]:
        os.makedirs(d, exist_ok=True)

    # 3) 遍历每个 tomo_id 的所有 motor
    counts = {"train": 0, "val": 0}
    for idx, row in df.iterrows():
        tomo_id = row["tomo_id"]
        zc = int(round(row["Motor axis 0"]))
        yc = int(round(row["Motor axis 1"]))
        xc = int(round(row["Motor axis 2"]))
        tomo_dir = os.path.join(raw_train_dir, tomo_id)
        if not os.path.isdir(tomo_dir):
            continue

        for z in range(zc - trust, zc + trust + 1):
            if z < 0:
                continue
            slice_name = f"slice_{z:04d}.jpg"
            src_img = os.path.join(tomo_dir, slice_name)
            if not os.path.exists(src_img):
                continue

            img = Image.open(src_img)
            w, h = img.size
            x_center_norm = xc / w
            y_center_norm = yc / h
            bw_norm = box_size / w
            bh_norm = box_size / h

            new_fn = f"{tomo_id}_z{z:04d}_y{yc:04d}_x{xc:04d}.jpg"
            if tomo_id in train_tomos:
                dst_img = os.path.join(img_tr_dir, new_fn)
                dst_lbl = os.path.join(lbl_tr_dir, new_fn.replace(".jpg", ".txt"))
                counts["train"] += 1
            else:
                dst_img = os.path.join(img_va_dir, new_fn)
                dst_lbl = os.path.join(lbl_va_dir, new_fn.replace(".jpg", ".txt"))
                counts["val"] += 1

            shutil.copy(src_img, dst_img)
            with open(dst_lbl, "w") as f:
                f.write(f"0 {x_center_norm:.6f} {y_center_norm:.6f} {bw_norm:.6f} {bh_norm:.6f}\n")

    print(f"[BUILD] YOLO Dataset built: train_count={counts['train']}, val_count={counts['val']}")

    yaml_dict = {
        "path": out_dir,
        "train": "images/train",
        "val":   "images/val",
        "names": {0: "motor"}
    }
    with open(os.path.join(out_dir, "dataset.yaml"), "w") as f:
        yaml.dump(yaml_dict, f)
    print(f"[BUILD] dataset.yaml saved ← {out_dir}/dataset.yaml")

# ------------- 3. YOLOv8n 微调 -------------
def fine_tune_yolov8n(yaml_path, weights, save_dir):
    """
    从 COCO 预训练权重加载 YOLOv8n，然后在 yolo_dataset 上用较强增强做 50 轮微调
    """
    print("\n[YOLO-FINE] =====> Start YOLOv8n Fine-tuning <=====")
    model = YOLO(weights)
    os.makedirs(os.path.join(save_dir, "yolov8n"), exist_ok=True)

    model.train(
        data=yaml_path,
        epochs=YOLO_FINE_EPOCHS,
        batch=YOLO_FINE_BATCH,
        imgsz=YOLO_FINE_IMGSZ,
        project=save_dir,
        name="yolov8n",
        exist_ok=True,
        patience=YOLO_FINE_PATIENCE,
        lr0=YOLO_FINE_LR0,
        mosaic=YOLO_FINE_MOSAIC,
        mixup=YOLO_FINE_MIXUP,
        flipud=YOLO_FINE_FLIPUD,
        fliplr=YOLO_FINE_FLIPLR,
        degrees=YOLO_FINE_DEGREES,
        translate=YOLO_FINE_TRANSLATE,
        scale=YOLO_FINE_SCALE,
        shear=YOLO_FINE_SHEAR,
        perspective=YOLO_FINE_PERSPECT,
        hsv_h=YOLO_FINE_HSV_H,
        hsv_s=YOLO_FINE_HSV_S,
        hsv_v=YOLO_FINE_HSV_V,
        verbose=True
    )

    best = os.path.join(save_dir, "yolov8n", "weights", "best.pt")
    print(f"[YOLO-FINE] YOLOv8n fine-tuned best.pt → {best}")
    return best

# ============================================
# 1) get_num_labels
# ============================================
def get_num_labels(label_dir: str) -> int:
    """
    遍历 label_dir 下所有子目录及 .txt 文件，找出最大 cls_id，然后返回 max_id + 1。
    如果根本没有任何 .txt，就返回 0。
    """
    max_id = -1
    pattern = os.path.join(label_dir, "**", "*.txt")
    all_txts = glob.glob(pattern, recursive=True)
    if not all_txts:
        return 0

    for txt_path in all_txts:
        with open(txt_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                cid = int(parts[0])
                if cid > max_id:
                    max_id = cid

    return max_id + 1 if max_id >= 0 else 0

# ============================================
# 2) Helper: 把 YOLO (x_center_norm, y_center_norm, w_norm, h_norm) 转为 [x_min, y_min, w, h]（像素）
# ============================================
def yolo_to_detr_boxes(yolo_txt_path: str, img_width: int, img_height: int):
    boxes = []
    labels = []
    with open(yolo_txt_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            cls_id = int(parts[0])
            x_center_norm = float(parts[1])
            y_center_norm = float(parts[2])
            w_norm = float(parts[3])
            h_norm = float(parts[4])

            x_center = x_center_norm * img_width
            y_center = y_center_norm * img_height
            w_box = w_norm * img_width
            h_box = h_norm * img_height

            x_min = x_center - w_box / 2
            y_min = y_center - h_box / 2

            x_min = max(0, x_min)
            y_min = max(0, y_min)
            w_box = min(w_box, img_width - x_min)
            h_box = min(h_box, img_height - y_min)

            boxes.append([x_min, y_min, w_box, h_box])
            labels.append(cls_id)
    return boxes, labels

# ============================================
# 3) Dataset: 递归读取 yolo_dataset 下的 images 和 labels, 增加数据增强
# ============================================
class YoloToDetrDataset(Dataset):
    def __init__(self, img_dir: str, label_dir: str, transforms=None):
        super().__init__()
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transforms = transforms

        # 收集所有图片路径
        self.image_paths = []
        for ext in (".jpg", ".jpeg", ".png", ".bmp"):
            pattern = os.path.join(self.img_dir, "**", f"*{ext}")
            self.image_paths.extend(glob.glob(pattern, recursive=True))
        self.image_paths.sort()

        if len(self.image_paths) == 0:
            raise ValueError(f"[ERROR] 在 {img_dir} 下没有找到任何图片文件，请检查路径是否正确。")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        img_width, img_height = image.size

        # 数据增强（PIL Image），包括随机裁剪、色彩抖动、水平翻转、旋转
        if self.transforms is not None:
            image = self.transforms(image)

        # 生成对应的 YOLO .txt 路径
        rel = os.path.relpath(img_path, self.img_dir)
        base = os.path.splitext(rel)[0]
        yolo_txt = os.path.join(self.label_dir, base + ".txt")

        if os.path.exists(yolo_txt):
            boxes, labels = yolo_to_detr_boxes(yolo_txt, img_width, img_height)
            boxes_tensor = torch.as_tensor(boxes, dtype=torch.float32)   # [num_obj,4]
            labels_tensor = torch.as_tensor(labels, dtype=torch.int64)   # [num_obj]
        else:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.int64)

        target = {"boxes": boxes_tensor, "labels": labels_tensor}
        return image, target

# ============================================
# 4) Collate 函数：把一个 batch 的 PIL.Image + YOLO label 转为 DETR 所需的格式
# ============================================
def detr_collate_fn(batch, processor: DetrImageProcessor):
    images = []
    coco_annotations = []

    for idx, (image, annot) in enumerate(batch):
        images.append(image)
        boxes = annot["boxes"].tolist()
        labels = annot["labels"].tolist()

        objs = []
        for (x_min, y_min, w_box, h_box), lb in zip(boxes, labels):
            objs.append({
                "bbox": [x_min, y_min, w_box, h_box],
                "category_id": lb,
                "area": float(w_box * h_box),
                "iscrowd": 0
            })
        coco_annotations.append({
            "image_id": idx,      # batch 内 index 即可
            "annotations": objs
        })

    encoding = processor(images=images, annotations=coco_annotations, return_tensors="pt")
    pixel_values = encoding["pixel_values"]    # Tensor[B,3,H,W]
    labels_for_detr = encoding["labels"]       # list of dicts

    return pixel_values, labels_for_detr

# ============================================
# 5) 构造 COCO-格式的 GT annotations（供 COCOeval 用）
# ============================================
def build_coco_gt_annotations(dataset: Dataset):
    """
    兼容 Subset 和 YoloToDetrDataset：
    - 如果传入的是 Subset，则从它的 .dataset 和 .indices 提取正确的 image_paths 子集。
    - 返回 COCO 格式的 dict，包含 images, annotations, categories。
    """
    # 判断是否是 Subset
    if isinstance(dataset, Subset):
        base_dataset = dataset.dataset
        indices = dataset.indices
    else:
        base_dataset = dataset
        indices = list(range(len(base_dataset)))

    coco_gt = {"images": [], "annotations": [], "categories": []}
    ann_id = 1
    cat_ids = set()

    # 先收集所有类别 ID（仅在验证集上）
    for idx in indices:
        img_path = base_dataset.image_paths[idx]
        rel = os.path.relpath(img_path, base_dataset.img_dir)
        base = os.path.splitext(rel)[0]
        txt_path = os.path.join(base_dataset.label_dir, base + ".txt")
        if not os.path.exists(txt_path):
            continue
        with open(txt_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                cid = int(parts[0])
                cat_ids.add(cid)

    for cid in sorted(cat_ids):
        coco_gt["categories"].append({"id": cid, "name": str(cid)})

    img_id_counter = 1
    for idx in indices:
        img_path = base_dataset.image_paths[idx]
        rel = os.path.relpath(img_path, base_dataset.img_dir).replace("\\", "/")
        base = os.path.splitext(rel)[0]
        txt_path = os.path.join(base_dataset.label_dir, base + ".txt")

        image = Image.open(img_path)
        width, height = image.size
        coco_gt["images"].append({
            "id": img_id_counter,
            "file_name": rel,
            "width": width,
            "height": height
        })

        if os.path.exists(txt_path):
            with open(txt_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    cid = int(parts[0])
                    x_center_norm = float(parts[1])
                    y_center_norm = float(parts[2])
                    w_norm = float(parts[3])
                    h_norm = float(parts[4])

                    x_center = x_center_norm * width
                    y_center = y_center_norm * height
                    w_box = w_norm * width
                    h_box = h_norm * height
                    x_min = x_center - w_box / 2
                    y_min = y_center - h_box / 2

                    x_min = max(0, x_min)
                    y_min = max(0, y_min)
                    w_box = min(w_box, width - x_min)
                    h_box = min(h_box, height - y_min)

                    coco_gt["annotations"].append({
                        "id": ann_id,
                        "image_id": img_id_counter,
                        "category_id": cid,
                        "bbox": [x_min, y_min, w_box, h_box],
                        "area": w_box * h_box,
                        "iscrowd": 0
                    })
                    ann_id += 1

        img_id_counter += 1

    return coco_gt

# ============================================
# 6) 后处理：把 DETR 输出转为 COCOeval 需要的 List[dict]
# ============================================
def postprocess_predictions(outputs, image_ids, orig_sizes, processor: DetrImageProcessor):
    """
    outputs: DetrForObjectDetectionOutput
    image_ids: List[int], 与 orig_sizes 顺序一一对应
    orig_sizes: List[(height, width)], 后处理需要按 (H, W) 格式
    processor: DetrImageProcessor
    返回：
      List[{
        "image_id": int,
        "category_id": int,
        "bbox": [x_min, y_min, w, h],  # 像素坐标
        "score": float
      }, …]
    """
    batch_size = outputs.logits.shape[0]
    results = []

    processed = processor.post_process_object_detection(
        outputs,
        target_sizes=orig_sizes,   # **必须是 (height, width)**
        threshold=0.0
    )
    for i in range(batch_size):
        img_id = image_ids[i]
        scores = processed[i]["scores"].cpu().tolist()
        labels = processed[i]["labels"].cpu().tolist()
        boxes = processed[i]["boxes"].cpu().tolist()  # [ [x_min,y_min,x_max,y_max], … ]

        for score, label_id, box in zip(scores, labels, boxes):
            x_min, y_min, x_max, y_max = box
            w_box = x_max - x_min
            h_box = y_max - y_min
            results.append({
                "image_id": img_id,
                "category_id": label_id,
                "bbox": [x_min, y_min, w_box, h_box],
                "score": score
            })

    return results

# ============================================
# 7) 训练函数：finetune_detr_fold（单 Fold 训练）
# ============================================
def finetune_detr_fold(
    train_dataset: Dataset,
    val_dataset: Dataset,
    output_dir: str,
    num_labels: int,
    num_epochs: int = DETR_FINE_EPOCHS,
    batch_size: int = DETR_FINE_BATCH,
    num_workers: int = 4,
    save_every: int = 5,
    eval_every: int = DETR_FINE_EVAL_EVERY,
    patience: int = DETR_FINE_PATIENCE,
    device: str = "cuda"
):
    """
    train_dataset, val_dataset: 可以是 Subset 或 YoloToDetrDataset
    output_dir: 本 fold 的保存目录
    num_labels: 类别数
    """
    os.makedirs(output_dir, exist_ok=True)

    # ---------- Step A: 初始化 DETR 模型 & Processor ----------
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

    # 替换分类头为 (num_labels + 1)
    model.config.num_labels = num_labels
    in_feat = model.class_labels_classifier.in_features
    model.class_labels_classifier = torch.nn.Linear(in_feat, num_labels + 1)
    torch.nn.init.xavier_uniform_(model.class_labels_classifier.weight)
    model.class_labels_classifier.bias.data.zero_()

    # 调整 loss 权重：增加 bbox_loss_coefficient 到 6.0
    model.config.bbox_loss_coef = 6.0
    model.config.giou_loss_coef = 2.0
    model.config.cls_loss_coef  = 1.0

    # 冻结 backbone
    for name, param in model.named_parameters():
        if "backbone" in name:
            param.requires_grad = False

    model.to(device)

    # ---------- Step B: DataLoader ----------
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=lambda b: detr_collate_fn(b, processor)
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=lambda b: detr_collate_fn(b, processor)
    )

    # ---------- Step C: 构造 COCO GT for 验证集 ----------
    coco_gt_dict = build_coco_gt_annotations(val_dataset)
    coco_gt_dict["info"] = {}
    coco_gt_dict["licenses"] = []
    coco_gt = COCO()
    coco_gt.dataset = coco_gt_dict
    coco_gt.createIndex()

    # ---------- Step D: 参数分组 & 优化器 ----------
    param_groups = [
        {"params": [p for n, p in model.named_parameters() if "backbone" in n], "lr": DETR_FINE_LR_BACKBONE},
        {"params": [p for n, p in model.named_parameters() if ("backbone" not in n and "class_labels_classifier" not in n)], "lr": DETR_FINE_LR_DECODER},
        {"params": model.class_labels_classifier.parameters(), "lr": DETR_FINE_LR_HEAD}
    ]
    optimizer = torch.optim.AdamW(param_groups, weight_decay=DETR_FINE_WEIGHT_DECAY)

    # ---------- Step E: Warm-up + Scheduler ----------
    total_steps = num_epochs * len(train_loader)
    warmup_steps = min(1000, total_steps // 10)

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(
            0.0, float(total_steps - current_step) / float(max(1, total_steps - warmup_steps))
        )

    scheduler = LambdaLR(optimizer, lr_lambda)

    # ---------- Step F: 训练 + 验证 循环，含 Early Stopping 但只在第 50 轮后开始监控 mAP@0.5 ----------
    # 构造 val_img_id_list 与 val_img_size_list，注意 val_img_size_list 存 (height, width)
    # 由于 val_dataset 可能是 Subset，需要先提取 indices
    if isinstance(val_dataset, Subset):
        base_val = val_dataset.dataset
        idxs = val_dataset.indices
    else:
        base_val = val_dataset
        idxs = list(range(len(base_val)))

    val_img_id_list = []
    val_img_size_list = []
    for counter, idx in enumerate(idxs, 1):
        img_path = base_val.image_paths[idx]
        image = Image.open(img_path)
        w, h = image.size
        val_img_id_list.append(counter)
        val_img_size_list.append((h, w))

    best_mAP50 = 0.0
    epochs_without_improve = 0
    global_step = 0

    START_MONITOR_EPOCH = 50  # 只在第 50 轮之后，才开始计“mAP@0.5 是否提升”
    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0

        # 到第 21 轮时解冻 backbone
        if epoch == 21:
            for name, param in model.named_parameters():
                param.requires_grad = True

        train_pbar = tqdm(train_loader, desc=f"Fold Train Epoch {epoch}/{num_epochs}", unit="batch")
        for step, (pixel_values, labels) in enumerate(train_pbar):
            pixel_values = pixel_values.to(device)
            labels_on_dev = [{k: v.to(device) for k, v in t.items()} for t in labels]

            outputs = model(pixel_values=pixel_values, labels=labels_on_dev)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            running_loss += loss.item()
            global_step += 1

            # —— Quick‐Eval：每隔 eval_every 个 batch 做一次 mAP@0.5 估计 —— 
            if (step + 1) % eval_every == 0:
                model.eval()
                quick_preds = []
                offset = 0

                for val_pixel_values, _ in val_loader:
                    val_pixel_values = val_pixel_values.to(device)
                    with torch.no_grad():
                        outputs_pred = model(pixel_values=val_pixel_values)
                    bsize = val_pixel_values.shape[0]

                    batch_image_ids = val_img_id_list[offset: offset + bsize]
                    batch_sizes     = val_img_size_list[offset: offset + bsize]  # (h, w)
                    preds = postprocess_predictions(outputs_pred, batch_image_ids, batch_sizes, processor)
                    quick_preds.extend(preds)
                    offset += bsize

                coco_dt_quick = coco_gt.loadRes(quick_preds)
                coco_eval_quick = COCOeval(coco_gt, coco_dt_quick, iouType="bbox")
                coco_eval_quick.params.imgIds = val_img_id_list
                coco_eval_quick.evaluate()
                coco_eval_quick.accumulate()
                coco_eval_quick.summarize()
                model.train()

                quick_mAP50 = coco_eval_quick.stats[1]  # AP@0.5
                train_pbar.set_postfix({"loss": f"{loss.item():.4f}", "AP@0.5": f"{quick_mAP50:.3f}"})

        avg_train_loss = running_loss / len(train_loader)

        # —— Epoch 末尾完整验证 —— 
        model.eval()
        val_loss_sum = 0.0
        all_preds = []
        offset = 0

        for val_pixel_values, labels in tqdm(val_loader, desc=f"Fold Val Epoch {epoch}/{num_epochs}", unit="batch"):
            val_pixel_values = val_pixel_values.to(device)
            labels_on_dev = [{k: v.to(device) for k, v in t.items()} for t in labels]

            with torch.no_grad():
                outputs = model(pixel_values=val_pixel_values, labels=labels_on_dev)
                val_loss_sum += outputs.loss.item()
                outputs_pred = model(pixel_values=val_pixel_values)

                bsize = val_pixel_values.shape[0]
                batch_image_ids = val_img_id_list[offset: offset + bsize]
                batch_sizes     = val_img_size_list[offset: offset + bsize]  # (h, w)
                preds = postprocess_predictions(outputs_pred, batch_image_ids, batch_sizes, processor)
                all_preds.extend(preds)
                offset += bsize

        avg_val_loss = val_loss_sum / len(val_loader)

        # 完整评估写入 JSON 并调用 COCOeval
        preds_json = os.path.join(output_dir, f"fold_predictions_epoch{epoch}.json")
        with open(preds_json, "w") as f:
            json.dump(all_preds, f)

        coco_dt = coco_gt.loadRes(preds_json)
        coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
        coco_eval.params.imgIds = val_img_id_list
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        mAP_all = coco_eval.stats[0]
        mAP_50  = coco_eval.stats[1]
        mAP_75  = coco_eval.stats[2]

        print(
            f"[Fold Epoch {epoch}/{num_epochs}] "
            f"train_loss={avg_train_loss:.4f}  "
            f"val_loss={avg_val_loss:.4f}  "
            f"mAP@0.50:0.95={mAP_all:.4f}  "
            f"mAP@0.50={mAP_50:.4f}  "
            f"mAP@0.75={mAP_75:.4f}"
        )

        # Early Stopping：只有当 epoch >= START_MONITOR_EPOCH 时，才开始监控 mAP@0.5
        if epoch >= START_MONITOR_EPOCH:
            if mAP_50 > best_mAP50:
                best_mAP50 = mAP_50
                epochs_without_improve = 0
            else:
                epochs_without_improve += 1
        else:
            # 前 50 轮不计入 patience 范围
            epochs_without_improve = 0

        # 保存 checkpoint
        if (epoch % save_every == 0) or (epoch == num_epochs) or (mAP_50 >= best_mAP50):
            ckpt_path = os.path.join(output_dir, f"fold_checkpoint-epoch{epoch}.pt")
            torch.save(model.state_dict(), ckpt_path)
            print(f"[INFO] Saved checkpoint → {ckpt_path}")

        # 仅在监控期才触发 Early Stopping
        if epoch >= START_MONITOR_EPOCH and epochs_without_improve >= patience:
            print(f"[EarlyStopping] 自第 {START_MONITOR_EPOCH} 轮起，mAP@0.5 已连续 {patience} 轮未提升，提前终止。")
            break

        model.train()

    # 保存最后一个最优模型
    last_ckpt = os.path.join(output_dir, f"fold_best_checkpoint.pt")
    torch.save(model.state_dict(), last_ckpt)
    print(f"[INFO] Fold training finished. Best checkpoint → {last_ckpt}")

# ============================================
# 脚本入口：只需配置下面这几行即可开始微调或进行 5-fold CV
# ============================================
if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # # 1) 构建 yolo_dataset
    # if os.path.exists(YOLO_DATASET_DIR):
    #     shutil.rmtree(YOLO_DATASET_DIR)
    # os.makedirs(YOLO_DATASET_DIR, exist_ok=True)
    # build_yolo_dataset(RAW_TRAIN_DIR, RAW_LABELS_CSV, YOLO_DATASET_DIR, val_split=0.2, trust=4, box_size=24)

    # # 2) YOLOv8n 微调
    # os.makedirs(YOLO_FINETUNE_DIR, exist_ok=True)
    # yolo_yaml = os.path.join(YOLO_DATASET_DIR, "dataset.yaml")
    # yolobest = fine_tune_yolov8n(yolo_yaml, YOLO_PRETRAIN_WEIGHTS, YOLO_FINETUNE_DIR)

    # 3) DETR 微调（单 Fold 或 5-fold）
    USE_FIVE_FOLD = True  # 如果需要做 5-fold，请设置为 True；否则仅做一次训练

    IMG_DIR    = "./yolo_dataset/images/train"   # 仅用训练目录进行 5-fold
    LABEL_DIR  = "./yolo_dataset/labels/train"   # 仅用训练标签目录进行 5-fold
    BASE_OUTPUT_DIR = "./detr_weights_finetune"

    # 构造基础 Dataset（仅 train 部分）
    det_transforms = T.Compose([
        T.RandomResizedCrop(size=800, scale=(0.8, 1.0), ratio=(0.95, 1.05)),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=10)
    ])
    full_dataset = YoloToDetrDataset(img_dir=IMG_DIR, label_dir=LABEL_DIR, transforms=det_transforms)
    num_labels = get_num_labels(LABEL_DIR)
    print(f"[MAIN] 总样本数 (train) = {len(full_dataset)}, 类别数 = {num_labels}")

    if USE_FIVE_FOLD:
        print("[MAIN] 开始 5-fold 交叉验证 ...")
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        for fold, (train_idx, val_idx) in enumerate(kf.split(full_dataset), 1):
            print(f"\n[MAIN] Fold {fold}/5")
            train_subset = Subset(full_dataset, train_idx)
            # 验证集不做随机增强，只做 Resize→ToTensor
            val_base_dataset = YoloToDetrDataset(
                img_dir=IMG_DIR,
                label_dir=LABEL_DIR,
                transforms=T.Compose([
                    T.Resize(size=(800, 800)),
                ])
            )
            val_subset = Subset(val_base_dataset, val_idx)

            fold_output = os.path.join(BASE_OUTPUT_DIR, f"fold_{fold}")
            finetune_detr_fold(
                train_dataset=train_subset,
                val_dataset=val_subset,
                output_dir=fold_output,
                num_labels=num_labels,
                num_epochs=DETR_FINE_EPOCHS,
                batch_size=DETR_FINE_BATCH,
                num_workers=4,
                save_every=5,
                eval_every=DETR_FINE_EVAL_EVERY,
                patience=DETR_FINE_PATIENCE,
                device="cuda" if torch.cuda.is_available() else "cpu",
            )
        print("[MAIN] 5-fold 训练完成。")
    else:
        # 单 Fold 训练时，把原有 train/val 同时利用
        train_img_dir = os.path.join("./yolo_dataset/images", "train")
        train_lbl_dir = os.path.join("./yolo_dataset/labels", "train")
        val_img_dir   = os.path.join("./yolo_dataset/images", "val")
        val_lbl_dir   = os.path.join("./yolo_dataset/labels", "val")

        train_dataset = YoloToDetrDataset(img_dir=train_img_dir, label_dir=train_lbl_dir, transforms=det_transforms)
        val_dataset   = YoloToDetrDataset(img_dir=val_img_dir, label_dir=val_lbl_dir, transforms=T.Compose([
            T.Resize(size=(800, 800)),
        ]))

        single_output = os.path.join(BASE_OUTPUT_DIR, "single_fold")
        finetune_detr_fold(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            output_dir=single_output,
            num_labels=num_labels,
            num_epochs=DETR_FINE_EPOCHS,
            batch_size=DETR_FINE_BATCH,
            num_workers=4,
            save_every=5,
            eval_every=DETR_FINE_EVAL_EVERY,
            patience=DETR_FINE_PATIENCE,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        print("[MAIN] 单次训练完成。")





# # finetune.py
# # =============================================================================
# # 该脚本完成：
# #   1) 从 Kaggle 原始数据：train/ + train_labels.csv → 构建 YOLOv8n/DETR 用的 yolo_dataset/
# #   2) 在 yolo_dataset 上对 YOLOv8n 做“较强增强的微调” → 输出 ./yolo_weights_finetune/yolov8n/weights/best.pt
# #   3) 在 yolo_dataset 上对 DETR(ResNet-50) 做“较强增强的微调” → 输出 ./detr_weights_finetune/checkpoint-last/
# #
# # 运行：
# #   python finetune.py
# #
# # 输出示例：
# #   ./yolo_dataset/             （生成的数据集目录）
# #   ./yolo_weights_finetune/
# #       └─ yolov8n/
# #            └─ weights/best.pt
# #   ./detr_weights_finetune/
# #       └─ checkpoint-last/      （HuggingFace DETR 微调权重）
# # =============================================================================

# import os
# import random
# import shutil
# import yaml
# import json
# import torch
# import numpy as np
# from ultralytics import YOLO
# from PIL import Image
# import pandas as pd
# import sys
# import timm
# from tqdm import tqdm
# import warnings
# warnings.filterwarnings(
#     "ignore",
#     message="for .*: copying from a non-meta parameter",
#     category=UserWarning
# )

# # HuggingFace DETR 相关
# from transformers import DetrForObjectDetection, DetrImageProcessor
# from torch.utils.data import Dataset, DataLoader, Subset
# from torchvision.transforms import functional as F
# import torchvision.transforms as T
# from pycocotools.coco import COCO
# from pycocotools.cocoeval import COCOeval
# import glob
# from torch.optim.lr_scheduler import LambdaLR

# # ------------- 1. 全局配置 -------------
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"[DEVICE] Using device: {device}")

# # Kaggle 原始目录（仅含 train/, test/, train_labels.csv, sample_submission.csv）
# RAW_DATA_DIR     = "./data"
# RAW_TRAIN_DIR    = os.path.join(RAW_DATA_DIR, "train")       # 含 tomo_id 子目录
# RAW_TEST_DIR     = os.path.join(RAW_DATA_DIR, "test")
# RAW_LABELS_CSV   = os.path.join(RAW_DATA_DIR, "train_labels.csv")

# # 我们要生成的 yolo_dataset 目录
# YOLO_DATASET_DIR = os.path.join(os.getcwd(), "yolo_dataset")

# # 微调输出目录
# YOLO_FINETUNE_DIR = os.path.join(os.getcwd(), "yolo_weights_finetune")
# DETR_FINETUNE_DIR = os.path.join(os.getcwd(), "detr_weights_finetune")

# # YOLOv8n 微调用预训练权重 (需提前把 yolov8n.pt 放在工作目录下)
# YOLO_PRETRAIN_WEIGHTS = "yolov8n.pt"

# # —— YOLOv8n 微调 超参 —— #
# YOLO_FINE_EPOCHS    = 50
# YOLO_FINE_BATCH     = 16
# YOLO_FINE_IMGSZ     = 640
# YOLO_FINE_PATIENCE  = 7
# YOLO_FINE_LR0       = 0.01

# YOLO_FINE_MOSAIC    = 0.7
# YOLO_FINE_MIXUP     = 0.3
# YOLO_FINE_FLIPLR    = 0.5
# YOLO_FINE_FLIPUD    = 0.1
# YOLO_FINE_DEGREES   = 10
# YOLO_FINE_TRANSLATE = 0.1
# YOLO_FINE_SCALE     = 0.3
# YOLO_FINE_SHEAR     = 0.0
# YOLO_FINE_PERSPECT  = 0.001
# YOLO_FINE_HSV_H     = 0.015
# YOLO_FINE_HSV_S     = 0.7
# YOLO_FINE_HSV_V     = 0.4

# # —— DETR 微调 超参 —— #
# DETR_FINE_EPOCHS    = 50
# DETR_FINE_BATCH     = 16     # Batch Size 调整为 8
# DETR_FINE_LR_BACKBONE = 1e-5
# DETR_FINE_LR_DECODER  = 5e-5
# DETR_FINE_LR_HEAD     = 1e-4
# DETR_FINE_WEIGHT_DECAY= 1e-4
# DETR_FINE_EVAL_EVERY  = 1000

# # ------------- 2. 从原始数据构建 yolo_dataset -------------
# def build_yolo_dataset(raw_train_dir, labels_csv, out_dir, val_split=0.2, trust=4, box_size=24):
#     """
#     从 Kaggle 原始的 train/<tomo_id>/<slice>.jpg + train_labels.csv
#     生成 yolo_dataset/{images,labels}/{train,val} 结构，以及 dataset.yaml。
#     """
#     # 1) 读取 CSV，按 tomo_id 分组
#     df = pd.read_csv(labels_csv)
#     df = df.dropna(subset=["Motor axis 0", "Motor axis 1", "Motor axis 2"])
#     tomo_with_motor = df["tomo_id"].unique().tolist()
#     random.shuffle(tomo_with_motor)
#     split = int(len(tomo_with_motor) * (1 - val_split))
#     train_tomos = set(tomo_with_motor[:split])
#     val_tomos   = set(tomo_with_motor[split:])

#     # 2) 创建目录
#     img_tr_dir = os.path.join(out_dir, "images", "train")
#     lbl_tr_dir = os.path.join(out_dir, "labels", "train")
#     img_va_dir = os.path.join(out_dir, "images", "val")
#     lbl_va_dir = os.path.join(out_dir, "labels", "val")
#     for d in [img_tr_dir, lbl_tr_dir, img_va_dir, lbl_va_dir]:
#         os.makedirs(d, exist_ok=True)

#     # 3) 遍历每个 tomo_id 的所有 motor
#     counts = {"train": 0, "val": 0}
#     for idx, row in df.iterrows():
#         tomo_id = row["tomo_id"]
#         zc = int(round(row["Motor axis 0"]))
#         yc = int(round(row["Motor axis 1"]))
#         xc = int(round(row["Motor axis 2"]))
#         tomo_dir = os.path.join(raw_train_dir, tomo_id)
#         if not os.path.isdir(tomo_dir):
#             continue

#         for z in range(zc - trust, zc + trust + 1):
#             if z < 0:
#                 continue
#             slice_name = f"slice_{z:04d}.jpg"
#             src_img = os.path.join(tomo_dir, slice_name)
#             if not os.path.exists(src_img):
#                 continue

#             img = Image.open(src_img)
#             w, h = img.size
#             x_center_norm = xc / w
#             y_center_norm = yc / h
#             bw_norm = box_size / w
#             bh_norm = box_size / h

#             new_fn = f"{tomo_id}_z{z:04d}_y{yc:04d}_x{xc:04d}.jpg"
#             if tomo_id in train_tomos:
#                 dst_img = os.path.join(img_tr_dir, new_fn)
#                 dst_lbl = os.path.join(lbl_tr_dir, new_fn.replace(".jpg", ".txt"))
#                 counts["train"] += 1
#             else:
#                 dst_img = os.path.join(img_va_dir, new_fn)
#                 dst_lbl = os.path.join(lbl_va_dir, new_fn.replace(".jpg", ".txt"))
#                 counts["val"] += 1

#             shutil.copy(src_img, dst_img)
#             with open(dst_lbl, "w") as f:
#                 f.write(f"0 {x_center_norm:.6f} {y_center_norm:.6f} {bw_norm:.6f} {bh_norm:.6f}\n")

#     print(f"[BUILD] YOLO Dataset built: train_count={counts['train']}, val_count={counts['val']}")

#     yaml_dict = {
#         "path": out_dir,
#         "train": "images/train",
#         "val":   "images/val",
#         "names": {0: "motor"}
#     }
#     with open(os.path.join(out_dir, "dataset.yaml"), "w") as f:
#         yaml.dump(yaml_dict, f)
#     print(f"[BUILD] dataset.yaml saved ← {out_dir}/dataset.yaml")

# # ------------- 3. YOLOv8n 微调 -------------
# def fine_tune_yolov8n(yaml_path, weights, save_dir):
#     """
#     从 COCO 预训练权重加载 YOLOv8n，然后在 yolo_dataset 上用较强增强做 50 轮微调
#     """
#     print("\n[YOLO-FINE] =====> Start YOLOv8n Fine-tuning <=====")
#     model = YOLO(weights)
#     os.makedirs(os.path.join(save_dir, "yolov8n"), exist_ok=True)

#     model.train(
#         data=yaml_path,
#         epochs=YOLO_FINE_EPOCHS,
#         batch=YOLO_FINE_BATCH,
#         imgsz=YOLO_FINE_IMGSZ,
#         project=save_dir,
#         name="yolov8n",
#         exist_ok=True,
#         patience=YOLO_FINE_PATIENCE,
#         lr0=YOLO_FINE_LR0,
#         mosaic=YOLO_FINE_MOSAIC,
#         mixup=YOLO_FINE_MIXUP,
#         flipud=YOLO_FINE_FLIPUD,
#         fliplr=YOLO_FINE_FLIPLR,
#         degrees=YOLO_FINE_DEGREES,
#         translate=YOLO_FINE_TRANSLATE,
#         scale=YOLO_FINE_SCALE,
#         shear=YOLO_FINE_SHEAR,
#         perspective=YOLO_FINE_PERSPECT,
#         hsv_h=YOLO_FINE_HSV_H,
#         hsv_s=YOLO_FINE_HSV_S,
#         hsv_v=YOLO_FINE_HSV_V,
#         verbose=True
#     )

#     best = os.path.join(save_dir, "yolov8n", "weights", "best.pt")
#     print(f"[YOLO-FINE] YOLOv8n fine-tuned best.pt → {best}")
#     return best

# # ============================================
# # 1) get_num_labels
# # ============================================
# def get_num_labels(label_dir: str) -> int:
#     """
#     遍历 label_dir 下所有子目录及 .txt 文件，找出最大 cls_id，然后返回 max_id + 1。
#     如果根本没有任何 .txt，就返回 0。
#     """
#     max_id = -1
#     pattern = os.path.join(label_dir, "**", "*.txt")
#     all_txts = glob.glob(pattern, recursive=True)
#     if not all_txts:
#         return 0

#     for txt_path in all_txts:
#         with open(txt_path, "r") as f:
#             for line in f:
#                 parts = line.strip().split()
#                 if not parts:
#                     continue
#                 cid = int(parts[0])
#                 if cid > max_id:
#                     max_id = cid

#     return max_id + 1 if max_id >= 0 else 0

# # ============================================
# # 2) Helper: 把 YOLO (x_center_norm, y_center_norm, w_norm, h_norm) 转为 [x_min, y_min, w, h]（像素）
# # ============================================
# def yolo_to_detr_boxes(yolo_txt_path: str, img_width: int, img_height: int):
#     boxes = []
#     labels = []
#     with open(yolo_txt_path, "r") as f:
#         for line in f:
#             parts = line.strip().split()
#             if not parts:
#                 continue
#             cls_id = int(parts[0])
#             x_center_norm = float(parts[1])
#             y_center_norm = float(parts[2])
#             w_norm = float(parts[3])
#             h_norm = float(parts[4])

#             x_center = x_center_norm * img_width
#             y_center = y_center_norm * img_height
#             w_box = w_norm * img_width
#             h_box = h_norm * img_height

#             x_min = x_center - w_box / 2
#             y_min = y_center - h_box / 2

#             x_min = max(0, x_min)
#             y_min = max(0, y_min)
#             w_box = min(w_box, img_width - x_min)
#             h_box = min(h_box, img_height - y_min)

#             boxes.append([x_min, y_min, w_box, h_box])
#             labels.append(cls_id)
#     return boxes, labels

# # ============================================
# # 3) Dataset: 递归读取 yolo_dataset 下的 images 和 labels, 增加数据增强
# # ============================================
# class YoloToDetrDataset(Dataset):
#     def __init__(self, img_dir: str, label_dir: str, transforms=None):
#         super().__init__()
#         self.img_dir = img_dir
#         self.label_dir = label_dir
#         self.transforms = transforms

#         # 收集所有图片路径
#         self.image_paths = []
#         for ext in (".jpg", ".jpeg", ".png", ".bmp"):
#             pattern = os.path.join(self.img_dir, "**", f"*{ext}")
#             self.image_paths.extend(glob.glob(pattern, recursive=True))
#         self.image_paths.sort()

#         if len(self.image_paths) == 0:
#             raise ValueError(f"[ERROR] 在 {img_dir} 下没有找到任何图片文件，请检查路径是否正确。")

#     def __len__(self):
#         return len(self.image_paths)

#     def __getitem__(self, idx):
#         img_path = self.image_paths[idx]
#         image = Image.open(img_path).convert("RGB")
#         img_width, img_height = image.size

#         # 数据增强（PIL Image）
#         if self.transforms is not None:
#             image = self.transforms(image)

#         # 生成对应的 YOLO .txt 路径
#         rel = os.path.relpath(img_path, self.img_dir)
#         base = os.path.splitext(rel)[0]
#         yolo_txt = os.path.join(self.label_dir, base + ".txt")

#         if os.path.exists(yolo_txt):
#             boxes, labels = yolo_to_detr_boxes(yolo_txt, img_width, img_height)
#             boxes_tensor = torch.as_tensor(boxes, dtype=torch.float32)   # [num_obj,4]
#             labels_tensor = torch.as_tensor(labels, dtype=torch.int64)   # [num_obj]
#         else:
#             boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
#             labels_tensor = torch.zeros((0,), dtype=torch.int64)

#         target = {"boxes": boxes_tensor, "labels": labels_tensor}
#         return image, target

# # ============================================
# # 4) Collate 函数：把一个 batch 的 PIL.Image + YOLO label 转为 DETR 所需的格式
# # ============================================
# def detr_collate_fn(batch, processor: DetrImageProcessor):
#     images = []
#     coco_annotations = []

#     for idx, (image, annot) in enumerate(batch):
#         images.append(image)
#         boxes = annot["boxes"].tolist()
#         labels = annot["labels"].tolist()

#         objs = []
#         for (x_min, y_min, w_box, h_box), lb in zip(boxes, labels):
#             objs.append({
#                 "bbox": [x_min, y_min, w_box, h_box],
#                 "category_id": lb,
#                 "area": float(w_box * h_box),
#                 "iscrowd": 0
#             })
#         coco_annotations.append({
#             "image_id": idx,      # batch 内 index 即可
#             "annotations": objs
#         })

#     encoding = processor(images=images, annotations=coco_annotations, return_tensors="pt")
#     pixel_values = encoding["pixel_values"]    # Tensor[B,3,H,W]
#     labels_for_detr = encoding["labels"]       # list of dicts

#     return pixel_values, labels_for_detr

# # ============================================
# # 5) 构造 COCO-格式的 GT annotations（供 COCOeval 用）
# # ============================================
# def build_coco_gt_annotations(dataset: YoloToDetrDataset):
#     coco_gt = {"images": [], "annotations": [], "categories": []}
#     ann_id = 1
#     cat_ids = set()

#     for img_path in dataset.image_paths:
#         rel = os.path.relpath(img_path, dataset.img_dir)
#         base = os.path.splitext(rel)[0]
#         txt_path = os.path.join(dataset.label_dir, base + ".txt")
#         if not os.path.exists(txt_path):
#             continue
#         with open(txt_path, "r") as f:
#             for line in f:
#                 parts = line.strip().split()
#                 if not parts:
#                     continue
#                 cid = int(parts[0])
#                 cat_ids.add(cid)

#     for cid in sorted(cat_ids):
#         coco_gt["categories"].append({"id": cid, "name": str(cid)})

#     img_id_counter = 1
#     for img_path in dataset.image_paths:
#         rel = os.path.relpath(img_path, dataset.img_dir).replace("\\", "/")
#         base = os.path.splitext(rel)[0]
#         txt_path = os.path.join(dataset.label_dir, base + ".txt")

#         image = Image.open(img_path)
#         width, height = image.size
#         coco_gt["images"].append({
#             "id": img_id_counter,
#             "file_name": rel,
#             "width": width,
#             "height": height
#         })

#         if os.path.exists(txt_path):
#             with open(txt_path, "r") as f:
#                 for line in f:
#                     parts = line.strip().split()
#                     if not parts:
#                         continue
#                     cid = int(parts[0])
#                     x_center_norm = float(parts[1])
#                     y_center_norm = float(parts[2])
#                     w_norm = float(parts[3])
#                     h_norm = float(parts[4])

#                     x_center = x_center_norm * width
#                     y_center = y_center_norm * height
#                     w_box = w_norm * width
#                     h_box = h_norm * height
#                     x_min = x_center - w_box / 2
#                     y_min = y_center - h_box / 2

#                     x_min = max(0, x_min)
#                     y_min = max(0, y_min)
#                     w_box = min(w_box, width - x_min)
#                     h_box = min(h_box, height - y_min)

#                     coco_gt["annotations"].append({
#                         "id": ann_id,
#                         "image_id": img_id_counter,
#                         "category_id": cid,
#                         "bbox": [x_min, y_min, w_box, h_box],
#                         "area": w_box * h_box,
#                         "iscrowd": 0
#                     })
#                     ann_id += 1

#         img_id_counter += 1

#     return coco_gt

# # ============================================
# # 6) 后处理：把 DETR 输出转为 COCOeval 需要的 List[dict]
# # ============================================ 
# def postprocess_predictions(outputs, image_ids, orig_sizes, processor: DetrImageProcessor):
#     """
#     outputs: DetrForObjectDetectionOutput
#     image_ids: List[int], 与 orig_sizes 顺序一一对应
#     orig_sizes: List[(height, width)], 后处理需要按 (H, W) 格式
#     processor: DetrImageProcessor
#     返回：
#       List[{
#         "image_id": int,
#         "category_id": int,
#         "bbox": [x_min, y_min, w, h],  # 像素坐标
#         "score": float
#       }, …]
#     """
#     batch_size = outputs.logits.shape[0]
#     results = []

#     processed = processor.post_process_object_detection(
#         outputs,
#         target_sizes=orig_sizes,   # **必须是 (height, width)**
#         threshold=0.0
#     )
#     for i in range(batch_size):
#         img_id = image_ids[i]
#         scores = processed[i]["scores"].cpu().tolist()
#         labels = processed[i]["labels"].cpu().tolist()
#         boxes = processed[i]["boxes"].cpu().tolist()  # [ [x_min,y_min,x_max,y_max], … ]

#         for score, label_id, box in zip(scores, labels, boxes):
#             x_min, y_min, x_max, y_max = box
#             w_box = x_max - x_min
#             h_box = y_max - y_min
#             results.append({
#                 "image_id": img_id,
#                 "category_id": label_id,
#                 "bbox": [x_min, y_min, w_box, h_box],
#                 "score": score
#             })

#     return results

# # ============================================
# # 7) 训练函数：finetune_detr（含实时 mAP 以及 (H,W) 修正）
# # ============================================
# def finetune_detr(
#     img_dir: str,
#     label_dir: str,
#     output_dir: str,
#     num_epochs: int = DETR_FINE_EPOCHS,
#     batch_size: int = DETR_FINE_BATCH,
#     num_workers: int = 4,
#     save_every: int = 5,
#     eval_every: int = DETR_FINE_EVAL_EVERY,
#     device: str = "cuda" if torch.cuda.is_available() else "cpu",
# ):
#     """
#     img_dir:     "./yolo_dataset/images"
#     label_dir:   "./yolo_dataset/labels"
#     output_dir:  "./detr_weights_finetune/checkpoints"
#     eval_every:  每隔多少个 train batch 触发一次 Quick‐Eval
#     """
#     os.makedirs(output_dir, exist_ok=True)

#     # ---------- Step A: 计算类别数 ----------
#     num_labels = get_num_labels(label_dir)
#     if num_labels <= 0:
#         raise ValueError(f"[ERROR] 在 {label_dir} 下没有任何 .txt 标注，无法计算 num_labels。")
#     print(f"[INFO] Detected {num_labels} class IDs (0-based).")

#     # ---------- Step B: 初始化 DETR 模型 & Processor ----------
#     print("[INFO] Loading DetrImageProcessor and DetrForObjectDetection...")
#     processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
#     model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

#     # 替换分类头为 (num_labels + 1)
#     model.config.num_labels = num_labels
#     in_feat = model.class_labels_classifier.in_features
#     model.class_labels_classifier = torch.nn.Linear(in_feat, num_labels + 1)
#     torch.nn.init.xavier_uniform_(model.class_labels_classifier.weight)
#     model.class_labels_classifier.bias.data.zero_()

#     # 调整 loss 权重
#     model.config.bbox_loss_coef = 5.0
#     model.config.giou_loss_coef = 2.0
#     model.config.cls_loss_coef  = 1.0

#     model.to(device)
#     print(f"[INFO] Model moved to device = {device}.")

#     # ---------- Step C: 构造 Dataset & DataLoader ----------
#     # 构造数据增强：随机水平翻转、随机旋转 ±10°
#     det_transforms = T.Compose([
#         T.RandomHorizontalFlip(p=0.5),
#         T.RandomRotation(degrees=10)
#     ])

#     train_img_dir = os.path.join(img_dir, "train")
#     train_lbl_dir = os.path.join(label_dir, "train")
#     val_img_dir = os.path.join(img_dir, "val")
#     val_lbl_dir = os.path.join(label_dir, "val")

#     train_dataset = YoloToDetrDataset(img_dir=train_img_dir, label_dir=train_lbl_dir, transforms=det_transforms)
#     val_dataset   = YoloToDetrDataset(img_dir=val_img_dir,   label_dir=val_lbl_dir,   transforms=None)

#     train_loader = DataLoader(
#         train_dataset,
#         batch_size=batch_size,
#         shuffle=True,
#         num_workers=num_workers,
#         collate_fn=lambda b: detr_collate_fn(b, processor)
#     )
#     val_loader = DataLoader(
#         val_dataset,
#         batch_size=batch_size,
#         shuffle=False,
#         num_workers=num_workers,
#         collate_fn=lambda b: detr_collate_fn(b, processor)
#     )
#     print(f"[INFO] Created DataLoaders. #train = {len(train_dataset)}, #val = {len(val_dataset)}, batch_size = {batch_size}")

#     # ---------- Step D: 构造 COCO GT for 验证集 ----------
#     print("[INFO] Building COCO-style GT for validation set ...")
#     coco_gt_dict = build_coco_gt_annotations(val_dataset)
#     coco_gt_dict["info"] = {}
#     coco_gt_dict["licenses"] = []
#     coco_gt = COCO()
#     coco_gt.dataset = coco_gt_dict
#     coco_gt.createIndex()

#     # ---------- Step E: 参数分组 & 优化器 ----------
#     param_groups = [
#         {"params": [p for n, p in model.named_parameters() if "backbone" in n], "lr": DETR_FINE_LR_BACKBONE},
#         {"params": [p for n, p in model.named_parameters() if ("backbone" not in n and "class_labels_classifier" not in n)], "lr": DETR_FINE_LR_DECODER},
#         {"params": model.class_labels_classifier.parameters(), "lr": DETR_FINE_LR_HEAD}
#     ]
#     optimizer = torch.optim.AdamW(param_groups, weight_decay=DETR_FINE_WEIGHT_DECAY)
#     print(f"[INFO] Optimizer: AdamW with grouped LR")

#     # ---------- Step F: Warm-up + Scheduler ----------
#     total_steps = num_epochs * len(train_loader)
#     warmup_steps = min(1000, total_steps // 10)

#     def lr_lambda(current_step):
#         if current_step < warmup_steps:
#             return float(current_step) / float(max(1, warmup_steps))
#         return max(
#             0.0, float(total_steps - current_step) / float(max(1, total_steps - warmup_steps))
#         )

#     scheduler = LambdaLR(optimizer, lr_lambda)

#     # ---------- Step G: 训练 + 验证 循环（含实时 mAP） ----------
#     # 先构造 val_img_id_list 与 val_img_size_list，注意 val_img_size_list 存 (height, width)
#     val_img_id_list = []
#     val_img_size_list = []
#     for idx, img_path in enumerate(val_dataset.image_paths):
#         image = Image.open(img_path)
#         w, h = image.size
#         val_img_id_list.append(idx + 1)      # image_id 从 1 开始
#         val_img_size_list.append((h, w))    # **(height, width)**

#     global_step = 0
#     for epoch in range(1, num_epochs + 1):
#         model.train()
#         running_loss = 0.0

#         train_pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}", unit="batch")
#         for step, (pixel_values, labels) in enumerate(train_pbar):
#             pixel_values = pixel_values.to(device)
#             labels_on_dev = [{k: v.to(device) for k, v in t.items()} for t in labels]

#             outputs = model(pixel_values=pixel_values, labels=labels_on_dev)
#             loss = outputs.loss
#             loss.backward()
#             optimizer.step()
#             scheduler.step()
#             optimizer.zero_grad()

#             running_loss += loss.item()
#             global_step += 1

#             # —— Quick‐Eval：每隔 eval_every 个 batch 做一次 mAP 估计 —— 
#             if (step + 1) % eval_every == 0:
#                 model.eval()
#                 quick_preds = []
#                 offset = 0

#                 for val_pixel_values, _ in val_loader:
#                     val_pixel_values = val_pixel_values.to(device)
#                     with torch.no_grad():
#                         outputs_pred = model(pixel_values=val_pixel_values)
#                     bsize = val_pixel_values.shape[0]

#                     batch_image_ids = val_img_id_list[offset: offset + bsize]
#                     batch_sizes     = val_img_size_list[offset: offset + bsize]  # 已是 (h, w)
#                     preds = postprocess_predictions(outputs_pred, batch_image_ids, batch_sizes, processor)
#                     quick_preds.extend(preds)
#                     offset += bsize

#                 coco_dt_quick = coco_gt.loadRes(quick_preds)
#                 coco_eval_quick = COCOeval(coco_gt, coco_dt_quick, iouType="bbox")
#                 coco_eval_quick.params.imgIds = val_img_id_list
#                 coco_eval_quick.evaluate()
#                 coco_eval_quick.accumulate()
#                 coco_eval_quick.summarize()
#                 model.train()

#                 quick_mAP50 = coco_eval_quick.stats[1]  # AP@0.5
#                 train_pbar.set_postfix({"loss": f"{loss.item():.4f}", "AP@0.5": f"{quick_mAP50:.3f}"})

#         avg_train_loss = running_loss / len(train_loader)

#         # —— Epoch 末尾完整验证 —— 
#         model.eval()
#         val_loss_sum = 0.0
#         all_preds = []
#         offset = 0

#         for val_pixel_values, labels in tqdm(val_loader, desc=f"Epoch {epoch}/{num_epochs} [Val]", unit="batch"):
#             val_pixel_values = val_pixel_values.to(device)
#             labels_on_dev = [{k: v.to(device) for k, v in t.items()} for t in labels]

#             with torch.no_grad():
#                 outputs = model(pixel_values=val_pixel_values, labels=labels_on_dev)
#                 val_loss_sum += outputs.loss.item()
#                 outputs_pred = model(pixel_values=val_pixel_values)

#                 bsize = val_pixel_values.shape[0]
#                 batch_image_ids = val_img_id_list[offset: offset + bsize]
#                 batch_sizes     = val_img_size_list[offset: offset + bsize]  # (h, w)
#                 preds = postprocess_predictions(outputs_pred, batch_image_ids, batch_sizes, processor)
#                 all_preds.extend(preds)
#                 offset += bsize

#         avg_val_loss = val_loss_sum / len(val_loader)

#         # 写入 JSON 供 COCOeval 完整评估
#         preds_json = os.path.join(output_dir, f"predictions_epoch{epoch}.json")
#         with open(preds_json, "w") as f:
#             json.dump(all_preds, f)

#         coco_dt = coco_gt.loadRes(preds_json)
#         coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
#         coco_eval.params.imgIds = val_img_id_list
#         coco_eval.evaluate()
#         coco_eval.accumulate()
#         coco_eval.summarize()

#         mAP_all = coco_eval.stats[0]
#         mAP_50  = coco_eval.stats[1]
#         mAP_75  = coco_eval.stats[2]

#         print(
#             f"[Epoch {epoch}/{num_epochs}] "
#             f"train_loss={avg_train_loss:.4f}  "
#             f"val_loss={avg_val_loss:.4f}  "
#             f"mAP@0.50:0.95={mAP_all:.4f}  "
#             f"mAP@0.50={mAP_50:.4f}  "
#             f"mAP@0.75={mAP_75:.4f}"
#         )

#         if (epoch % save_every == 0) or (epoch == num_epochs):
#             ckpt_path = os.path.join(output_dir, f"checkpoint-epoch{epoch}.pt")
#             torch.save(model.state_dict(), ckpt_path)
#             print(f"[INFO] Saved checkpoint → {ckpt_path}")

#         model.train()

#     last_ckpt = os.path.join(output_dir, "checkpoint-last.pt")
#     torch.save(model.state_dict(), last_ckpt)
#     print(f"[INFO] Training finished. Last checkpoint → {last_ckpt}")

# # ============================================
# # 8) Overfit 测试代码（可选，供调试用）
# # ============================================
# def run_overfit_debug():
#     """
#     这段代码和上面 Overfit 训练部分合并示例，方便你跑一次 Overfit，
#     可视化 GT vs 预测框，确认 (H,W) 传参已经正确。
#     """
#     train_img_dir   = "./yolo_dataset/images/train"
#     train_label_dir = "./yolo_dataset/labels/train"
#     dataset = YoloToDetrDataset(img_dir=train_img_dir, label_dir=train_label_dir, transforms=None)
#     processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

#     pos_indices = []
#     for idx, img_path in enumerate(dataset.image_paths):
#         rel = os.path.relpath(img_path, dataset.img_dir)
#         txt_path = os.path.join(dataset.label_dir, os.path.splitext(rel)[0] + ".txt")
#         if os.path.exists(txt_path):
#             pos_indices.append(idx)
#     subset_idx = pos_indices[:2]
#     print(f"[Overfit] positive sample indices = {subset_idx}")

#     repeat_times = 50
#     subset_indices = subset_idx * repeat_times
#     mini_loader = DataLoader(
#         Subset(dataset, subset_indices),
#         batch_size=2,
#         shuffle=True,
#         collate_fn=lambda b: detr_collate_fn(b, processor)
#     )

#     model_overfit = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
#     num_labels = 1
#     model_overfit.config.num_labels = num_labels
#     in_feat = model_overfit.class_labels_classifier.in_features
#     model_overfit.class_labels_classifier = torch.nn.Linear(in_feat, num_labels + 1)
#     torch.nn.init.xavier_uniform_(model_overfit.class_labels_classifier.weight)
#     model_overfit.class_labels_classifier.bias.data.zero_()
#     model_overfit.to(device)

#     optimizer = torch.optim.AdamW(model_overfit.parameters(), lr=1e-5)

#     def debug_print_preds(model, processor, dataset, subset_idx, epoch, device):
#         model.eval()
#         for idx in subset_idx:
#             img, target = dataset[idx]
#             orig_w, orig_h = img.size
#             img_np = np.array(img)
#             fig, ax = plt.subplots(figsize=(4, 4))
#             ax.imshow(img_np, cmap="gray")
#             gt_boxes = target["boxes"].cpu().numpy()
#             for (x_min, y_min, w_box, h_box) in gt_boxes:
#                 rect = patches.Rectangle((x_min, y_min), w_box, h_box,
#                                          linewidth=1, edgecolor="g", facecolor="none")
#                 ax.add_patch(rect)
#             ax.set_title(f"Epoch {epoch}  idx={idx}  (green=GT)")

#             inputs = processor(images=img, return_tensors="pt").pixel_values.to(device)
#             with torch.no_grad():
#                 outputs_pred = model(pixel_values=inputs)
#             preds = postprocess_predictions(outputs_pred, [idx + 1], [(orig_h, orig_w)], processor)

#             for p in preds:
#                 if p["score"] < 0.3:
#                     continue
#                 x, y, w_box, h_box = p["bbox"]
#                 rect = patches.Rectangle((x, y), w_box, h_box,
#                                          linewidth=1, edgecolor="r", facecolor="none")
#                 ax.add_patch(rect)
#                 ax.text(x, y - 2, f"{p['score']:.2f}", color="r", fontsize=6)

#             ax.axis("off")
#             save_path = f"overfit_debug_epoch{epoch}_idx{idx}.png"
#             fig.savefig(save_path, bbox_inches="tight", pad_inches=0)
#             plt.close(fig)
#             print(f"[Overfit DEBUG] Saved → {save_path}")

#     model_overfit.train()
#     for epoch in range(1, 51):
#         total_loss = 0.0
#         for pixel_values, labels in mini_loader:
#             pixel_values = pixel_values.to(device)
#             labels_on_dev = [{k: v.to(device) for k, v in t.items()} for t in labels]

#             outputs = model_overfit(pixel_values=pixel_values, labels=labels_on_dev)
#             loss = outputs.loss
#             loss.backward()
#             optimizer.step()
#             optimizer.zero_grad()
#             total_loss += loss.item()

#         avg_loss = total_loss / len(mini_loader)
#         print(f"[Overfit] Epoch {epoch:02d}/50 — loss={avg_loss:.5f}")

#         if epoch % 10 == 0:
#             debug_print_preds(model_overfit, processor, dataset, subset_idx, epoch, device)

#     print("[Overfit] Training complete.")

# # ============================================
# # 脚本入口：只需配置下面这几行即可开始微调
# # ============================================
# if __name__ == "__main__":
#     random.seed(42)
#     np.random.seed(42)
#     torch.manual_seed(42)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(42)

#     # # 1) 构建 yolo_dataset
#     # if os.path.exists(YOLO_DATASET_DIR):
#     #     shutil.rmtree(YOLO_DATASET_DIR)
#     # os.makedirs(YOLO_DATASET_DIR, exist_ok=True)
#     # build_yolo_dataset(RAW_TRAIN_DIR, RAW_LABELS_CSV, YOLO_DATASET_DIR, val_split=0.2, trust=4, box_size=24)

#     # # 2) YOLOv8n 微调
#     # os.makedirs(YOLO_FINETUNE_DIR, exist_ok=True)
#     # yolo_yaml = os.path.join(YOLO_DATASET_DIR, "dataset.yaml")
#     # yolobest = fine_tune_yolov8n(yolo_yaml, YOLO_PRETRAIN_WEIGHTS, YOLO_FINETUNE_DIR)

#     # 3) DETR 微调
#     os.makedirs(DETR_FINETUNE_DIR, exist_ok=True)

#     # —— 先做 Overfit 调试 —— 
#     # run_overfit_debug()

#     # ------------- 只要改这三行就可以了 -------------
#     IMG_DIR    = "./yolo_dataset/images"            # 根目录：会自动递归查找子目录下的 *.jpg/*.png
#     LABEL_DIR  = "./yolo_dataset/labels"            # 根目录：会自动递归查找子目录下的 *.txt
#     OUTPUT_DIR = "./detr_weights_finetune/checkpoints"  # 保存 DETR checkpoint 的目录
#     # -----------------------------------------------

#     finetune_detr(
#         img_dir=IMG_DIR,
#         label_dir=LABEL_DIR,
#         output_dir=OUTPUT_DIR,
#         num_epochs=DETR_FINE_EPOCHS,
#         batch_size=DETR_FINE_BATCH,
#         num_workers=4,
#         save_every=5,
#         eval_every=DETR_FINE_EVAL_EVERY,
#         device="cuda" if torch.cuda.is_available() else "cpu",
#     )
