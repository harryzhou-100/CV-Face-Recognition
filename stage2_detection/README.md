# Stage 2: WIDER FACE 人脸检测（MMDetection）

基于 **MMDetection 3.3** 在 **WIDER FACE** 上训练 **RetinaNet R50-FPN** 单类人脸检测器，并在验证集上评估 mAP / Precision / Recall，输出分类可视化结果。

## 目录结构

```text
stage2_detection/
├── configs/retinanet_r50_fpn_wider_face.py   # 训练配置
├── scripts/
│   ├── prepare_wider_face.py                 # 下载 + COCO 标注转换
│   ├── train.py                              # 训练入口
│   ├── evaluate.py                           # 验证集评估与报告
│   ├── visualize_results.py                  # 测试图可视化（分类存放）
│   └── run_pipeline.sh                       # 一键流水线
├── data/wider_face/                          # 数据集与 COCO 标注
├── work_dirs/retinanet_r50_wider_face/       # 权重与日志
├── reports/                                  # evaluation_report.md
└── outputs/{easy,medium,hard,...}/           # 可视化结果
```

## 环境

```bash
pip install -r stage2_detection/requirements.txt
pip install https://download.openmmlab.com/mmcv/dist/cu121/torch2.1.0/mmcv-2.1.0-cp310-cp310-manylinux1_x86_64.whl
```

## 运行

**最终模型权重**：`work_dirs/retinanet_r50_wider_face/epoch_6.pth`（验证集 bbox_mAP ≈ 0.281）

```bash
cd /root/CV-Face-Recognition
python3 stage2_detection/scripts/evaluate.py \
  --checkpoint stage2_detection/work_dirs/retinanet_r50_wider_face/epoch_6.pth
python3 stage2_detection/scripts/visualize_results.py \
  --checkpoint stage2_detection/work_dirs/retinanet_r50_wider_face/epoch_6.pth
```

完整流水线（含数据准备与训练）：

```bash
python3 stage2_detection/scripts/prepare_wider_face.py
python3 stage2_detection/scripts/train.py   # 默认 6 epoch
python3 stage2_detection/scripts/evaluate.py
python3 stage2_detection/scripts/visualize_results.py
```

## 输出说明

- **reports/evaluation_report.md**：mAP、Precision、Recall、Easy/Medium/Hard 子集
- **outputs/**：按检测难度与场景分类的带框图片
