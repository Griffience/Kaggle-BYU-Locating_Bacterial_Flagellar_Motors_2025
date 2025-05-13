训练 YOLOv8 模型脚本
使用方法: python train_yolo.py
前提：已安装 ultralytics（pip install ultralytics）和其他依赖
项目结构假设在项目根下有：
  ├── data/
  │   ├── train/
  │   ├── test/
  │   ├── train_labels.csv
  │   └── sample_submission.csv
  ├── utils.py
  └── weights/      （训练后自动创建并保存模型）
```bash
#在本地软连接 目标路径-链接路径(创建快捷方式的名称)
ln -s "E:\DataSet\KaggleBYU\byu-locating-bacterial-flagellar-motors-2025" ./input/data
#Windows
mklink /D "F:\E_Academic\Competition\Kaggle\BYU\input\data" "E:\DataSet\KaggleBYU\byu-locating-bacterial-flagellar-motors-2025"
```