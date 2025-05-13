# submission.py
"""
生成提交文件脚本
使用方法: python submission.py
需要安装 ultralytics, opencv, numpy, pandas
"""
import os
import numpy as np
import pandas as pd
from ultralytics import YOLO
from tqdm import tqdm
from pathlib import Path

def perform_3d_nms(dets, thr=0.2):
    """简单3D NMS: 按置信度排序，合并近距离检测"""
    if not dets:
        return []
    dets = sorted(dets, key=lambda x: x['conf'], reverse=True)
    keep = []
    while dets:
        b = dets.pop(0)
        keep.append(b)
        dets = [d for d in dets if np.linalg.norm(
            (d['z']-b['z'], d['y']-b['y'], d['x']-b['x'])) > 24*thr]
    return keep

if __name__=='__main__':
    # 本地测试数据
    TEST_DIR = './data/test'
    MODEL_PT = './weights/motor_detector/best.pt'
    SUB = './submission.csv'
    model = YOLO(MODEL_PT)
    rows = []
    for tomo in tqdm(sorted(os.listdir(TEST_DIR))):
        tomopath = Path(TEST_DIR)/tomo
        imgs = sorted([str(p) for p in tomopath.glob('*.jpg')])
        # 全切片推理
        dets = []
        for img in imgs:
            res = model(img, conf=0.45)[0]
            for box, conf in zip(res.boxes.xyxy, res.boxes.conf):
                x1,y1,x2,y2 = box.cpu().numpy()
                dets.append({'z': int(Path(img).stem.split('_')[1]),
                             'y': int((y1+y2)/2),
                             'x': int((x1+x2)/2),
                             'conf': float(conf)})
        # NMS
        final = perform_3d_nms(dets)
        if final:
            best = final[0]
            rows.append([tomo, best['z'], best['y'], best['x']])
        else:
            rows.append([tomo, -1, -1, -1])
    df = pd.DataFrame(rows, columns=['tomo_id','Motor axis 0','Motor axis 1','Motor axis 2'])
    df.to_csv(SUB, index=False)
    print('提交文件已生成:', SUB)
