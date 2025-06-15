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

#HuggingFace DETR相关
from transformers import DetrForObjectDetection, DetrImageProcessor
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.transforms import functional as F
import torchvision.transforms as T
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import glob
from torch.optim.lr_scheduler import LambdaLR
from sklearn.model_selection import KFold


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[DEVICE] Using device: {device}")


RAW_DATA_DIR     = "./data"
RAW_TRAIN_DIR    = os.path.join(RAW_DATA_DIR, "train")       # 含 tomo_id 子目录
RAW_TEST_DIR     = os.path.join(RAW_DATA_DIR, "test")
RAW_LABELS_CSV   = os.path.join(RAW_DATA_DIR, "train_labels.csv")

#要生成的 yolo_dataset 目录
YOLO_DATASET_DIR = os.path.join(os.getcwd(), "yolo_dataset")

#微调输出目录
YOLO_FINETUNE_DIR = os.path.join(os.getcwd(), "yolo_weights_finetune")
DETR_FINETUNE_DIR = os.path.join(os.getcwd(), "detr_weights_finetune")

#YOLOv8n微调用预训练权重 (需提前把 yolov8n.pt 放在工作目录下)
YOLO_PRETRAIN_WEIGHTS = "yolov8n.pt"

#YOLOv8n微调超参
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

#DETR微调超参
DETR_FINE_EPOCHS       = 150
DETR_FINE_BATCH        = 16 #调整为16
DETR_FINE_LR_BACKBONE  = 1e-5
DETR_FINE_LR_DECODER   = 5e-5
DETR_FINE_LR_HEAD      = 1e-4
DETR_FINE_WEIGHT_DECAY = 1e-4
DETR_FINE_EVAL_EVERY   = 1000
DETR_FINE_PATIENCE     = 30     #30轮后开始监控mAP@0.5

#从原始数据构建 yolo_dataset
def build_yolo_dataset(raw_train_dir, labels_csv, out_dir, val_split=0.2, trust=4, box_size=24):
    """
     Kaggle原始的 train/<tomo_id>/<slice>.jpg + train_labels.csv
    生成yolo_dataset/{images,labels}/{train,val} 结构，以及 dataset.yaml。
    """
    #读取 CSV，按tomo_id分组
    df = pd.read_csv(labels_csv)
    df = df.dropna(subset=["Motor axis 0", "Motor axis 1", "Motor axis 2"])
    tomo_with_motor = df["tomo_id"].unique().tolist()
    random.shuffle(tomo_with_motor)
    split = int(len(tomo_with_motor) * (1 - val_split))
    train_tomos = set(tomo_with_motor[:split])
    val_tomos   = set(tomo_with_motor[split:])

    #创建目录
    img_tr_dir = os.path.join(out_dir, "images", "train")
    lbl_tr_dir = os.path.join(out_dir, "labels", "train")
    img_va_dir = os.path.join(out_dir, "images", "val")
    lbl_va_dir = os.path.join(out_dir, "labels", "val")
    for d in [img_tr_dir, lbl_tr_dir, img_va_dir, lbl_va_dir]:
        os.makedirs(d, exist_ok=True)

    #遍历每个tomo_id的所有motor
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

#YOLOv8n微调
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

def get_num_labels(label_dir: str) -> int:
    """
    遍历label_dir下所有子目录及 .txt 文件，找出最大 cls_id，然后返回 max_id + 1。
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


class YoloToDetrDataset(Dataset):
    def __init__(self, img_dir: str, label_dir: str, transforms=None):
        super().__init__()
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transforms = transforms

        #收集所有图片路径
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

        #数据增强.随机裁剪、色彩抖动、水平翻转、旋转
        if self.transforms is not None:
            image = self.transforms(image)

        #生成对应的YOLO.txt 路径
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


#Collate函数：把一个 batch 的 PIL.Image + YOLO label 转为 DETR 所需的格式
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
    pixel_values = encoding["pixel_values"]    #Tensor[B,3,H,W]
    labels_for_detr = encoding["labels"]       #list of dicts

    return pixel_values, labels_for_detr

#构造COCO-格式的 GT annotations（供 COCOeval 用）
def build_coco_gt_annotations(dataset: Dataset):
    """
    兼容Subset和YoloToDetrDataset：
    - 如果传入的是 Subset，则从它的 .dataset 和 .indices 提取正确的 image_paths 子集。
    - 返回 COCO 格式的 dict，包含 images, annotations, categories。
    """
    #判断是否是 Subset
    if isinstance(dataset, Subset):
        base_dataset = dataset.dataset
        indices = dataset.indices
    else:
        base_dataset = dataset
        indices = list(range(len(base_dataset)))

    coco_gt = {"images": [], "annotations": [], "categories": []}
    ann_id = 1
    cat_ids = set()

    #先收集所有类别 ID（仅在验证集上）
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


#后处理：把DETR输出转为COCOeval需要的List[dict]
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

    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

    #替换分类头为 (num_labels + 1)
    model.config.num_labels = num_labels
    in_feat = model.class_labels_classifier.in_features
    model.class_labels_classifier = torch.nn.Linear(in_feat, num_labels + 1)
    torch.nn.init.xavier_uniform_(model.class_labels_classifier.weight)
    model.class_labels_classifier.bias.data.zero_()

    #调整 loss 权重：增加 bbox_loss_coefficient 到 6.0
    model.config.bbox_loss_coef = 6.0
    model.config.giou_loss_coef = 2.0
    model.config.cls_loss_coef  = 1.0

    #冻结 backbone
    for name, param in model.named_parameters():
        if "backbone" in name:
            param.requires_grad = False

    model.to(device)

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

    coco_gt_dict = build_coco_gt_annotations(val_dataset)
    coco_gt_dict["info"] = {}
    coco_gt_dict["licenses"] = []
    coco_gt = COCO()
    coco_gt.dataset = coco_gt_dict
    coco_gt.createIndex()


    param_groups = [
        {"params": [p for n, p in model.named_parameters() if "backbone" in n], "lr": DETR_FINE_LR_BACKBONE},
        {"params": [p for n, p in model.named_parameters() if ("backbone" not in n and "class_labels_classifier" not in n)], "lr": DETR_FINE_LR_DECODER},
        {"params": model.class_labels_classifier.parameters(), "lr": DETR_FINE_LR_HEAD}
    ]
    optimizer = torch.optim.AdamW(param_groups, weight_decay=DETR_FINE_WEIGHT_DECAY)

    total_steps = num_epochs * len(train_loader)
    warmup_steps = min(1000, total_steps // 10)

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(
            0.0, float(total_steps - current_step) / float(max(1, total_steps - warmup_steps))
        )

    scheduler = LambdaLR(optimizer, lr_lambda)

    #构造val_img_id_list与val_img_size_list，注意val_img_size_list存 (height, width)
    #由于val_dataset可能是 Subset，需要先提取 indices
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

    START_MONITOR_EPOCH = 50  #只在第 50 轮之后，才开始计“mAP@0.5 是否提升”
    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0

        #到第21轮时解冻backbone
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

        if epoch >= START_MONITOR_EPOCH:
            if mAP_50 > best_mAP50:
                best_mAP50 = mAP_50
                epochs_without_improve = 0
            else:
                epochs_without_improve += 1
        else:
            epochs_without_improve = 0

        if (epoch % save_every == 0) or (epoch == num_epochs) or (mAP_50 >= best_mAP50):
            ckpt_path = os.path.join(output_dir, f"fold_checkpoint-epoch{epoch}.pt")
            torch.save(model.state_dict(), ckpt_path)
            print(f"[INFO] Saved checkpoint → {ckpt_path}")

        if epoch >= START_MONITOR_EPOCH and epochs_without_improve >= patience:
            print(f"[EarlyStopping] 自第 {START_MONITOR_EPOCH} 轮起，mAP@0.5 已连续 {patience} 轮未提升，提前终止。")
            break

        model.train()

    last_ckpt = os.path.join(output_dir, f"fold_best_checkpoint.pt")
    torch.save(model.state_dict(), last_ckpt)
    print(f"[INFO] Fold training finished. Best checkpoint → {last_ckpt}")


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    #构建 yolo_dataset
    if os.path.exists(YOLO_DATASET_DIR):
        shutil.rmtree(YOLO_DATASET_DIR)
    os.makedirs(YOLO_DATASET_DIR, exist_ok=True)
    build_yolo_dataset(RAW_TRAIN_DIR, RAW_LABELS_CSV, YOLO_DATASET_DIR, val_split=0.2, trust=4, box_size=24)

    #YOLOv8n 微调
    os.makedirs(YOLO_FINETUNE_DIR, exist_ok=True)
    yolo_yaml = os.path.join(YOLO_DATASET_DIR, "dataset.yaml")
    yolobest = fine_tune_yolov8n(yolo_yaml, YOLO_PRETRAIN_WEIGHTS, YOLO_FINETUNE_DIR)

    #DETR 微调
    USE_FIVE_FOLD = True  

    IMG_DIR    = "./yolo_dataset/images/train"   
    LABEL_DIR  = "./yolo_dataset/labels/train"
    BASE_OUTPUT_DIR = "./detr_weights_finetune"

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
            #验证集不做随机增强，只做Resize→ToTensor
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



