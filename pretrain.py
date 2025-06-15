#pretrain.py
'''
从Kaggle原始数据：train/ + train_labels.csv → 构建 YOLOv8n/DETR 用的 yolo_dataset/
在yolo_dataset上对YOLOv8n做“较强增强的预训练” → 输出 ./yolo_weights_pretrain/yolov8n/weights/best.pt
在yolo_dataset上对DETR(ResNet-50)做“较强增强的预训练” → 输出 ./detr_weights_pretrain/checkpoint-last/
运行：
python pretrain.py
输出示例：
   ./yolo_dataset/             
   ./yolo_weights_pretrain/
       └─ yolov8n/
            └─ weights/best.pt
   ./detr_weights_pretrain/
       └─ checkpoint-last/      （HuggingFace DETR 微调权重）
'''


import os
import random
import shutil
import yaml
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
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import to_tensor, resize

#全局配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[DEVICE] Using device: {device}")

#Kaggle原始目录
RAW_DATA_DIR  = "./data"
RAW_TRAIN_DIR = os.path.join(RAW_DATA_DIR, "train")  #含tomo_id子目录
RAW_TEST_DIR = os.path.join(RAW_DATA_DIR, "test")
RAW_LABELS_CSV = os.path.join(RAW_DATA_DIR, "train_labels.csv")

#要生成的yolo_dataset目录
YOLO_DATASET_DIR = os.path.join(os.getcwd(), "yolo_dataset")

#预训练输出目录
YOLO_PRETRAIN_DIR = os.path.join(os.getcwd(), "yolo_weights_pretrain")
DETR_PRETRAIN_DIR = os.path.join(os.getcwd(), "detr_weights_pretrain")

#YOLOv8n预训练用预训练权重 (需提前把 yolov8n.pt 放在工作目录下)
YOLO_PRETRAIN_WEIGHTS = "yolov8n.pt"

#YOLOv8n预训练超参
YOLO_PRE_EPOCHS = 50
YOLO_PRE_BATCH = 16
YOLO_PRE_IMGSZ = 640
YOLO_PRE_PATIENCE = 7
YOLO_PRE_LR0  = 0.01

YOLO_PRE_MOSAIC    = 0.7
YOLO_PRE_MIXUP     = 0.3
YOLO_PRE_FLIPLR    = 0.5
YOLO_PRE_FLIPUD    = 0.1
YOLO_PRE_DEGREES   = 10
YOLO_PRE_TRANSLATE = 0.1
YOLO_PRE_SCALE     = 0.3
YOLO_PRE_SHEAR     = 0.0
YOLO_PRE_PERSPECT  = 0.001
YOLO_PRE_HSV_H     = 0.015
YOLO_PRE_HSV_S     = 0.7
YOLO_PRE_HSV_V     = 0.4

#DETR预训练超参
DETR_PRE_EPOCHS    = 20
DETR_PRE_BATCH     = 4
DETR_PRE_LR        = 1e-4
DETR_IMAGE_SIZE    = 800

#从原始数据构建yolo_dataset
def build_yolo_dataset(raw_train_dir, labels_csv, out_dir, val_split=0.2, trust=4, box_size=24):
    """
    从 Kaggle 原始的 train/<tomo_id>/<slice>.jpg + train_labels.csv
    生成 yolo_dataset/{images,labels}/{train,val} 结构，以及 dataset.yaml。

    核心思路：
      - 读取 train_labels.csv:
         每行一个 motor，对应某个 tomo_id 和 (Motor axis 0,1,2) 绝对坐标 → (z, y, x)。
      - 对每个 motor，取 zc = round(Motor axis 0)，
        然后在 [zc-trust, zc+trust] 范围内挑 slice，生成归一化后YOLO框。
      - train/val 按 tomo_id 先随机打散，再拆分 80/20 (可选 k-fold)。
      - 只生成含 motor 的 slice；未包含 motor 的 slice 不生成。
    """

    #读取 CSV，按 tomo_id 分组
    df = pd.read_csv(labels_csv)
    df = df.dropna(subset=["Motor axis 0", "Motor axis 1", "Motor axis 2"])
    # 先筛出有 motor 的 tomogram
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

    #遍历每个 tomo_id 的所有 motor
    counts = {"train": 0, "val": 0}
    for idx, row in df.iterrows():
        tomo_id = row["tomo_id"]
        zc = int(round(row["Motor axis 0"]))
        yc = int(round(row["Motor axis 1"]))
        xc = int(round(row["Motor axis 2"]))
        # 当前 tomo子目录
        tomo_dir = os.path.join(raw_train_dir, tomo_id)
        if not os.path.isdir(tomo_dir):
            continue

        #把(zc-trust) ~ (zc+trust) 的 slice 都生成 VOYOLO 格式
        for z in range(zc - trust, zc + trust + 1):
            if z < 0:
                continue
            slice_name = f"slice_{z:04d}.jpg"
            src_img = os.path.join(tomo_dir, slice_name)
            if not os.path.exists(src_img):
                continue

            #读取 slice 大小
            img = Image.open(src_img)
            w, h = img.size

            #归一化 YOLO: x_center_norm, y_center_norm, bw_norm, bh_norm
            x_center_norm = xc / w
            y_center_norm = yc / h
            bw_norm = box_size / w
            bh_norm = box_size / h

            #新文件名格式：{tomo_id}_z{z:04d}_y{yc:04d}_x{xc:04d}.jpg
            new_fn = f"{tomo_id}_z{z:04d}_y{yc:04d}_x{xc:04d}.jpg"
            if tomo_id in train_tomos:
                dst_img = os.path.join(img_tr_dir, new_fn)
                dst_lbl = os.path.join(lbl_tr_dir, new_fn.replace(".jpg", ".txt"))
                counts["train"] += 1
            else:
                dst_img = os.path.join(img_va_dir, new_fn)
                dst_lbl = os.path.join(lbl_va_dir, new_fn.replace(".jpg", ".txt"))
                counts["val"] += 1

            #复制图片
            shutil.copy(src_img, dst_img)
            #写 label
            with open(dst_lbl, "w") as f:
                # 0 表示 “motor” 类
                f.write(f"0 {x_center_norm:.6f} {y_center_norm:.6f} {bw_norm:.6f} {bh_norm:.6f}\n")

    print(f"[BUILD] YOLO Dataset built: train_count={counts['train']}, val_count={counts['val']}")

    #生成 dataset.yaml
    yaml_dict = {
        "path": out_dir,
        "train": "images/train",
        "val":   "images/val",
        "names": {0: "motor"}
    }
    with open(os.path.join(out_dir, "dataset.yaml"), "w") as f:
        yaml.dump(yaml_dict, f)
    print(f"[BUILD] dataset.yaml saved ← {out_dir}/dataset.yaml")


#YOLOv8n 预训练
def train_yolov8n_pretrain(yaml_path, weights, save_dir):
    """
    从预训练权重加载 YOLOv8n，然后在 yolo_dataset 上用较强增强做 50 轮预训练
    """
    print("\n[YOLO-PRE] =====> Start YOLOv8n Pretraining <=====")
    model = YOLO(weights)
    os.makedirs(os.path.join(save_dir, "yolov8n"), exist_ok=True)

    model.train(
        data=yaml_path,
        epochs=YOLO_PRE_EPOCHS,
        batch=YOLO_PRE_BATCH,
        imgsz=YOLO_PRE_IMGSZ,
        project=save_dir,
        name="yolov8n",
        exist_ok=True,
        patience=YOLO_PRE_PATIENCE,
        lr0=YOLO_PRE_LR0,
        mosaic=YOLO_PRE_MOSAIC,
        mixup=YOLO_PRE_MIXUP,
        flipud=YOLO_PRE_FLIPUD,
        fliplr=YOLO_PRE_FLIPLR,
        degrees=YOLO_PRE_DEGREES,
        translate=YOLO_PRE_TRANSLATE,
        scale=YOLO_PRE_SCALE,
        shear=YOLO_PRE_SHEAR,
        perspective=YOLO_PRE_PERSPECT,
        hsv_h=YOLO_PRE_HSV_H,
        hsv_s=YOLO_PRE_HSV_S,
        hsv_v=YOLO_PRE_HSV_V,
        verbose=True
    )

    best = os.path.join(save_dir, "yolov8n", "weights", "best.pt")
    print(f"[YOLO-PRE] YOLOv8n pretrained best.pt → {best}")
    return best


import os
import random
import shutil
import glob
import json
import torch
import numpy as np
import yaml
from PIL import Image
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import DetrForObjectDetection, DetrImageProcessor
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from ultralytics import YOLO  # 如果你后面要用 YOLO 预训练的话
from torch.utils.data import Subset
import glob

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


#Helper: 把 YOLO (x_center_norm, y_center_norm, w_norm, h_norm) 转为 [x_min, y_min, w, h]（像素）
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

            #防止越界
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            w_box = min(w_box, img_width - x_min)
            h_box = min(h_box, img_height - y_min)

            boxes.append([x_min, y_min, w_box, h_box])
            labels.append(cls_id)
    return boxes, labels


#Dataset: 递归读取 yolo_dataset 下的images和 labels
class YoloToDetrDataset(Dataset):
    def __init__(self, img_dir: str, label_dir: str, img_extensions=(".jpg", ".jpeg", ".png", ".bmp"), transforms=None):
        super().__init__()
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transforms = transforms

        #收集所有图片路径
        self.image_paths = []
        for ext in img_extensions:
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

        #生成对应的 YOLO .txt 路径
        rel = os.path.relpath(img_path, self.img_dir)    # e.g. "0001.jpg" 或 "subdir/0001.jpg"
        base = os.path.splitext(rel)[0]
        yolo_txt = os.path.join(self.label_dir, base + ".txt")

        if os.path.exists(yolo_txt):
            boxes, labels = yolo_to_detr_boxes(yolo_txt, img_width, img_height)
            boxes_tensor = torch.as_tensor(boxes, dtype=torch.float32)  # [num_obj,4]
            labels_tensor = torch.as_tensor(labels, dtype=torch.int64)   # [num_obj]
        else:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.int64)

        if self.transforms is not None:
            image = self.transforms(image)

        target = {"boxes": boxes_tensor, "labels": labels_tensor}
        return image, target


#Collate 函数：把一个 batch 的 PIL.Image + YOLO label 转为 DETR 所需的格式
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


#构造 COCO-格式的 GT annotations（供 COCOeval 用）
def build_coco_gt_annotations(dataset: YoloToDetrDataset):
    coco_gt = {"images": [], "annotations": [], "categories": []}
    ann_id = 1
    cat_ids = set()

    #收集所有类别 ID
    for img_path in dataset.image_paths:
        rel = os.path.relpath(img_path, dataset.img_dir)
        base = os.path.splitext(rel)[0]
        txt_path = os.path.join(dataset.label_dir, base + ".txt")
        if not os.path.exists(txt_path):
            continue
        with open(txt_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                cid = int(parts[0])
                cat_ids.add(cid)

    #填categories
    for cid in sorted(cat_ids):
        coco_gt["categories"].append({"id": cid, "name": str(cid)})

    #填images + annotations
    img_id_counter = 1
    for img_path in dataset.image_paths:
        rel = os.path.relpath(img_path, dataset.img_dir).replace("\\", "/")
        base = os.path.splitext(rel)[0]
        txt_path = os.path.join(dataset.label_dir, base + ".txt")

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


#后处理：把 DETR 输出转为 COCOeval 需要的 List[dict]
def postprocess_predictions(outputs, image_ids, orig_sizes, processor: DetrImageProcessor):
    """
    outputs: DetrForObjectDetectionOutput
    image_ids: List[int]，与 orig_sizes 顺序一一对应
    orig_sizes: List[(height, width)]，后处理需要按 (H, W) 格式
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


#训练函数：train_detr_pretrain（含实时 mAP 以及 (H,W) 修正）
def train_detr_pretrain(
    img_dir: str,
    label_dir: str,
    output_dir: str,
    num_epochs: int = 20,
    batch_size: int = 4,
    learning_rate: float = 1e-4,
    num_workers: int = 4,
    save_every: int = 5,
    eval_every: int = 500,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    """
    img_dir:     "./yolo_dataset/images"
    label_dir:   "./yolo_dataset/labels"
    output_dir:  "./detr_weights_pretrain/checkpoints"
    eval_every:  每隔多少个 train batch 触发一次 Quick‐Eval
    """
    os.makedirs(output_dir, exist_ok=True)

    num_labels = get_num_labels(label_dir := label_dir)
    if num_labels <= 0:
        raise ValueError(f"[ERROR] 在 {label_dir} 下没有任何 .txt 标注，无法计算 num_labels。")
    print(f"[INFO] Detected {num_labels} class IDs (0-based).")

    print("[INFO] Loading DetrImageProcessor and DetrForObjectDetection...")
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

    #替换分类头为 (num_labels + 1)
    model.config.num_labels = num_labels
    in_feat = model.class_labels_classifier.in_features
    model.class_labels_classifier = torch.nn.Linear(in_feat, num_labels + 1)
    torch.nn.init.xavier_uniform_(model.class_labels_classifier.weight)
    model.class_labels_classifier.bias.data.zero_()

    model.to(device)
    print(f"[INFO] Model moved to device = {device}.")

    pil_transforms = None
    train_img_dir = os.path.join(img_dir, "train")
    train_lbl_dir = os.path.join(label_dir, "train")
    val_img_dir = os.path.join(img_dir, "val")
    val_lbl_dir = os.path.join(label_dir, "val")

    train_dataset = YoloToDetrDataset(img_dir=train_img_dir, label_dir=train_lbl_dir, transforms=pil_transforms)
    val_dataset   = YoloToDetrDataset(img_dir=val_img_dir,   label_dir=val_lbl_dir,   transforms=pil_transforms)

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
    print(f"[INFO] Created DataLoaders. #train = {len(train_dataset)}, #val = {len(val_dataset)}, batch_size = {batch_size}")

    print("[INFO] Building COCO-style GT for validation set ...")
    coco_gt_dict = build_coco_gt_annotations(val_dataset)
    # 补充 'info' 和 'licenses' 字段，避免 loadRes 报 KeyError
    coco_gt_dict["info"] = {}
    coco_gt_dict["licenses"] = []
    coco_gt = COCO()
    coco_gt.dataset = coco_gt_dict
    coco_gt.createIndex()

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    print(f"[INFO] Optimizer: AdamW, lr = {learning_rate}")

    #先构造 val_img_id_list 与 val_img_size_list，注意 val_img_size_list 存 (height, width)
    val_img_id_list = []
    val_img_size_list = []
    for idx, img_path in enumerate(val_dataset.image_paths):
        image = Image.open(img_path)
        w, h = image.size
        val_img_id_list.append(idx + 1)      #image_id 从 1 开始
        val_img_size_list.append((h, w))    #(height, width)

    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0

        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}", unit="batch")
        for step, (pixel_values, labels) in enumerate(train_pbar):
            pixel_values = pixel_values.to(device)
            labels_on_dev = [{k: v.to(device) for k, v in t.items()} for t in labels]

            outputs = model(pixel_values=pixel_values, labels=labels_on_dev)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            running_loss += loss.item()

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
                    batch_sizes     = val_img_size_list[offset: offset + bsize]  # 已是 (h, w)
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

                quick_mAP = coco_eval_quick.stats[0]  # mAP@0.50:0.95
                train_pbar.set_postfix({"loss": f"{loss.item():.4f}", "mAP": f"{quick_mAP:.3f}"})

        avg_train_loss = running_loss / len(train_loader)

        model.eval()
        val_loss_sum = 0.0
        all_preds = []
        offset = 0

        for val_pixel_values, labels in tqdm(val_loader, desc=f"Epoch {epoch}/{num_epochs} [Val]", unit="batch"):
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

        #写入 JSON 供 COCOeval 完整评估
        preds_json = os.path.join(output_dir, f"predictions_epoch{epoch}.json")
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
            f"[Epoch {epoch}/{num_epochs}] "
            f"train_loss={avg_train_loss:.4f}  "
            f"val_loss={avg_val_loss:.4f}  "
            f"mAP@0.50:0.95={mAP_all:.4f}  "
            f"mAP@0.50={mAP_50:.4f}  "
            f"mAP@0.75={mAP_75:.4f}"
        )

        if (epoch % save_every == 0) or (epoch == num_epochs):
            ckpt_path = os.path.join(output_dir, f"checkpoint-epoch{epoch}.pt")
            torch.save(model.state_dict(), ckpt_path)
            print(f"[INFO] Saved checkpoint → {ckpt_path}")

        model.train()

    last_ckpt = os.path.join(output_dir, "checkpoint-last.pt")
    torch.save(model.state_dict(), last_ckpt)
    print(f"[INFO] Training finished. Last checkpoint → {last_ckpt}")


#Overfit测试代码（供调试用）
def run_overfit_debug():
    """
    这段代码和上面 Overfit 训练部分合并示例，方便你跑一次 Overfit，
    可视化 GT vs 预测框，确认 (H,W) 传参已经正确。
    """
    #复用前面定义的 dataset, processor
    train_img_dir   = "./yolo_dataset/images/train"
    train_label_dir = "./yolo_dataset/labels/train"
    dataset = YoloToDetrDataset(img_dir=train_img_dir, label_dir=train_label_dir, transforms=None)
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

    #筛选前两个“带 motor”的索引
    pos_indices = []
    for idx, img_path in enumerate(dataset.image_paths):
        rel = os.path.relpath(img_path, dataset.img_dir)
        txt_path = os.path.join(dataset.label_dir, os.path.splitext(rel)[0] + ".txt")
        if os.path.exists(txt_path):
            pos_indices.append(idx)
    subset_idx = pos_indices[:2]
    print(f"[Overfit] positive sample indices = {subset_idx}")

    #构造 Overfit的小DataLoader
    repeat_times = 50
    subset_indices = subset_idx * repeat_times
    mini_loader = DataLoader(
        Subset(dataset, subset_indices),
        batch_size=2,
        shuffle=True,
        collate_fn=lambda b: detr_collate_fn(b, processor)
    )

    #初始化 Overfit 用的 DETR
    model_overfit = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
    num_labels = 1
    model_overfit.config.num_labels = num_labels
    in_feat = model_overfit.class_labels_classifier.in_features
    model_overfit.class_labels_classifier = torch.nn.Linear(in_feat, num_labels + 1)
    torch.nn.init.xavier_uniform_(model_overfit.class_labels_classifier.weight)
    model_overfit.class_labels_classifier.bias.data.zero_()
    model_overfit.to(device)

    optimizer = torch.optim.AdamW(model_overfit.parameters(), lr=1e-5)

    #辅助：可视化 GT vs 预测
    def debug_print_preds(model, processor, dataset, subset_idx, epoch, device):
        model.eval()
        for idx in subset_idx:
            img, target = dataset[idx]
            orig_w, orig_h = img.size
            img_np = np.array(img)
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.imshow(img_np, cmap="gray")
            gt_boxes = target["boxes"].cpu().numpy()
            for (x_min, y_min, w_box, h_box) in gt_boxes:
                rect = patches.Rectangle((x_min, y_min), w_box, h_box,
                                         linewidth=1, edgecolor="g", facecolor="none")
                ax.add_patch(rect)
            ax.set_title(f"Epoch {epoch}  idx={idx}  (green=GT)")

            inputs = processor(images=img, return_tensors="pt").pixel_values.to(device)
            with torch.no_grad():
                outputs_pred = model(pixel_values=inputs)
            preds = postprocess_predictions(outputs_pred, [idx + 1], [(orig_h, orig_w)], processor)

            for p in preds:
                if p["score"] < 0.3:
                    continue
                x, y, w_box, h_box = p["bbox"]
                rect = patches.Rectangle((x, y), w_box, h_box,
                                         linewidth=1, edgecolor="r", facecolor="none")
                ax.add_patch(rect)
                ax.text(x, y - 2, f"{p['score']:.2f}", color="r", fontsize=6)

            ax.axis("off")
            save_path = f"overfit_debug_epoch{epoch}_idx{idx}.png"
            fig.savefig(save_path, bbox_inches="tight", pad_inches=0)
            plt.close(fig)
            print(f"[Overfit DEBUG] Saved → {save_path}")

    #Overfit训练循环
    model_overfit.train()
    for epoch in range(1, 51):
        total_loss = 0.0
        for pixel_values, labels in mini_loader:
            pixel_values = pixel_values.to(device)
            labels_on_dev = [{k: v.to(device) for k, v in t.items()} for t in labels]

            outputs = model_overfit(pixel_values=pixel_values, labels=labels_on_dev)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()

        avg_loss = total_loss / len(mini_loader)
        print(f"[Overfit] Epoch {epoch:02d}/50 — loss={avg_loss:.5f}")

        #每 10 轮可视化一次
        if epoch % 10 == 0:
            debug_print_preds(model_overfit, processor, dataset, subset_idx, epoch, device)

    print("[Overfit] Training complete.")



if __name__ == "__main__":
    #固定随机
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

    #YOLOv8n 预训练
    os.makedirs(YOLO_PRETRAIN_DIR, exist_ok=True)
    yolo_yaml = os.path.join(YOLO_DATASET_DIR, "dataset.yaml")
    yolobest = train_yolov8n_pretrain(yolo_yaml, YOLO_PRETRAIN_WEIGHTS, YOLO_PRETRAIN_DIR)

    #DETR 预训练
    os.makedirs(DETR_PRETRAIN_DIR, exist_ok=True)

    # run_overfit_debug()
    
    IMG_DIR    = "./yolo_dataset/images"            # 根目录：会自动递归查找子目录下的 *.jpg/*.png
    LABEL_DIR  = "./yolo_dataset/labels"            # 根目录：会自动递归查找子目录下的 *.txt
    OUTPUT_DIR = "./detr_weights_pretrain/checkpoints"  # 保存 DETR checkpoint 的目录

    NUM_EPOCHS    = 20
    BATCH_SIZE    = 4
    LEARNING_RATE = 1e-4
    NUM_WORKERS   = 4
    SAVE_EVERY    = 5    #每隔多少个epoch保存一次checkpoint

    train_detr_pretrain(
        img_dir=IMG_DIR,
        label_dir=LABEL_DIR,
        output_dir=OUTPUT_DIR,
        num_epochs=20,
        batch_size=4,
        learning_rate=1e-4,
        num_workers=4,
        save_every=5,
        eval_every=500,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    
