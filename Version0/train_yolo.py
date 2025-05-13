# train_yolo.py
"""
训练 YOLOv8 模型脚本（适配小显存环境）
使用方法: python train_yolo.py
"""

import os
from pathlib import Path
import yaml

from utils import prepare_yolo_dataset
from ultralytics import YOLO

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.resolve()

# 路径配置
DATA_DIR    = PROJECT_ROOT / 'data'
YOLO_DATA   = DATA_DIR / 'yolo_dataset'
WEIGHTS_DIR = PROJECT_ROOT / 'weights'
PRETRAIN    = 'yolov8n.pt'  # 最轻量版

if __name__ == '__main__':
    # 1. 预处理
    print("1️⃣ 预处理数据...")
    prepare_yolo_dataset(str(DATA_DIR), str(YOLO_DATA))

    # 2. 修正 YAML
    print("2️⃣ 修正 dataset.yaml...")
    yaml_file = YOLO_DATA / 'dataset.yaml'
    cfg = yaml.safe_load(open(yaml_file, 'r'))
    cfg['path'] = str(YOLO_DATA)
    cfg['nc']   = 1
    with open(yaml_file, 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False)

    # 3. 创建输出目录
    print("3️⃣ 确保保存路径：", WEIGHTS_DIR)
    WEIGHTS_DIR.mkdir(exist_ok=True)

    # 4. 加载模型并训练
    print("4️⃣ 开始训练（小批量 + 小尺寸 + AMP）...")
    model = YOLO(PRETRAIN)

    abs_yaml = (PROJECT_ROOT / 'data' / 'yolo_dataset' / 'dataset.yaml').resolve()
    print("   使用数据配置：", abs_yaml)

    results = model.train(
        data=str(abs_yaml),
        epochs=50,

        # —— 下面三行是显存友好设置 —— #
        batch=1,      # 极限小批量
        imgsz=320,    # 降低输入分辨率
        amp=True,     # 自动混合精度

        project=str(WEIGHTS_DIR),
        name='motor_detector',
        patience=5,
        save_period=5
    )

    print("✅ 训练结束，权重保存在：", WEIGHTS_DIR / 'motor_detector')
