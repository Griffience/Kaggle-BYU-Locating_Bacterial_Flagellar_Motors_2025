# utils.py
import os
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import yaml
from pathlib import Path
import random
import glob
import matplotlib.pyplot as plt

def normalize_slice(slice_data):
    """
    归一化切片图像：使用2%–98%分位数剪裁并线性拉伸到[0,255]
    """
    p2 = np.percentile(slice_data, 2)
    p98 = np.percentile(slice_data, 98)
    clipped = np.clip(slice_data, p2, p98)
    normed = 255 * (clipped - p2) / (p98 - p2)
    return np.uint8(normed)


def prepare_yolo_dataset(
    data_dir: str,
    out_dir: str,
    trust: int = 4,
    box_size: int = 24,
    train_split: float = 0.8
) -> None:
    """
    预处理函数：
    - 从 data_dir/train 中读取断层图像和 train_labels.csv
    - 按 tomogram 划分 train/val（80/20）
    - 对每个马达中心周围 trust 范围内切片做归一化
    - 保存到 YOLO 格式目录 out_dir/images/{train,val} 和 labels/{train,val}
    - 生成 dataset.yaml
    """
    # 创建输出目录结构
    images_train = Path(out_dir) / 'images' / 'train'
    images_val = Path(out_dir) / 'images' / 'val'
    labels_train = Path(out_dir) / 'labels' / 'train'
    labels_val = Path(out_dir) / 'labels' / 'val'
    for p in [images_train, images_val, labels_train, labels_val]:
        p.mkdir(parents=True, exist_ok=True)

    # 读取标注
    labels_path = Path(data_dir) / 'train_labels.csv'
    df = pd.read_csv(labels_path)
    # 只保留有马达的 tomogram id
    motor_df = df[df['Number of motors'] > 0]
    tomos = motor_df['tomo_id'].unique().tolist()
    random.seed(42)
    random.shuffle(tomos)
    split = int(len(tomos) * train_split)
    train_tomos = set(tomos[:split])
    val_tomos = set(tomos[split:])

    # 处理函数
    def proc_set(tomo_set, img_dir, lbl_dir, name):
        for tomo in tomo_set:
            sub = df[df['tomo_id'] == tomo]
            # tomogram 下所有切片数
            all_slices = sorted((Path(data_dir)/'train'/tomo).glob('*.jpg'))
            max_z = len(all_slices)
            for _, row in sub.iterrows():
                zc, yc, xc = map(int, [row['Motor axis 0'], row['Motor axis 1'], row['Motor axis 2']])
                zmin = max(0, zc - trust)
                zmax = min(max_z-1, zc + trust)
                for z in range(zmin, zmax+1):
                    src = Path(data_dir)/'train'/tomo/f"slice_{z:04d}.jpg"
                    if not src.exists():
                        continue
                    img = Image.open(src)
                    arr = np.array(img)
                    norm = normalize_slice(arr)
                    dest_name = f"{tomo}_z{z:04d}_y{yc:04d}_x{xc:04d}.jpg"
                    dst_img = img_dir / dest_name
                    Image.fromarray(norm).save(dst_img)
                    # 生成标签
                    w, h = img.size
                    x_cn = xc / w
                    y_cn = yc / h
                    bw = box_size / w
                    bh = box_size / h
                    with open(lbl_dir / dest_name.replace('.jpg', '.txt'), 'w') as f:
                        f.write(f"0 {x_cn} {y_cn} {bw} {bh}\n")
    # 分别处理
    proc_set(train_tomos, images_train, labels_train, 'train')
    proc_set(val_tomos, images_val, labels_val, 'val')

    # 写 YAML
    yaml_dict = {
        'path': out_dir,
        'train': 'images/train',
        'val': 'images/val',
        'nc': 1,
        'names': {0: 'motor'}
    }
    with open(Path(out_dir)/'dataset.yaml', 'w') as f:
        yaml.dump(yaml_dict, f, default_flow_style=False)


def visualize_samples(out_dir: str, num: int = 4):
    """
    随机可视化 out_dir/images/train 下的 num 张图及其 YOLO 框
    """
    imgs = glob.glob(os.path.join(out_dir, 'images', 'train', '*.jpg'))
    sel = random.sample(imgs, min(num, len(imgs)))
    cols = 2
    rows = (len(sel)+1)//2
    fig, axs = plt.subplots(rows, cols, figsize=(8,4*rows))
    axs = axs.flatten()
    for ax, path in zip(axs, sel):
        img = Image.open(path)
        arr = np.array(img)
        norm = normalize_slice(arr)
        img_rgb = Image.fromarray(norm).convert('RGB')
        draw = ImageDraw.Draw(img_rgb, 'RGBA')
        txt = path.replace('.jpg','.txt').replace('images','labels')
        if os.path.exists(txt):
            with open(txt) as f:
                for line in f:
                    c,x,y,w,h = line.split()
                    x, y, w, h = float(x)*img.width, float(y)*img.height, float(w)*img.width, float(h)*img.height
                    x1,y1 = x-w/2, y-h/2
                    x2,y2 = x+w/2, y+h/2
                    draw.rectangle([x1,y1,x2,y2], outline=(255,0,0,200), fill=(255,0,0,50))
        ax.imshow(img_rgb)
        ax.axis('off')
    plt.tight_layout()
    plt.show()


