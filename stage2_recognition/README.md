# Stage 2 Recognition: ResNet50-ArcFace 人脸识别

基于 **InsightFace 标准 IResNet50-ArcFace** 预训练骨干，在本地 **MS1MV3 子集** 上微调，并在 **LFW** 上验证。

## 策略说明

| 阶段 | 说明 |
|------|------|
| 预训练初始化 | `models/ms1mv3_arcface_r50_backbone.pth`（MS1MV3 / ArcFace-Torch R50，475 个张量） |
| 微调 | 本地子集 ~10 万张图；**ArcFace 分类头**按子集身份数重新初始化 |
| 训练脚本 | `scripts/train.py`（保持不变） |
| 交付曲线 | `outputs/curves/loss_accuracy_curves.png` |

## 目录

```text
stage2_recognition/
├── configs/train_resnet50_arcface.json
├── models/
│   ├── iresnet.py
│   ├── arcface_head.py
│   └── ms1mv3_arcface_r50_backbone.pth
├── scripts/  train.py, verify_lfw.py, prepare_*.py, download_pretrained.py
├── data/ms1mv3_subset/, data/lfw_flat/
├── work_dirs/resnet50_arcface/
├── outputs/curves/
└── reports/
```

## 快速运行

```bash
cd /root/CV-Face-Recognition
python3 stage2_recognition/scripts/download_pretrained.py   # 首次
python3 stage2_recognition/scripts/train.py                 # 2 epoch 微调
python3 stage2_recognition/scripts/verify_lfw.py
```

## 微调结果

### 2 epoch 全量微调（batch=256）

见 `outputs/curves/history.json`：loss 40.6→22.5。

### 6 epoch 冻结骨干 + 仅训 ArcFace 头（batch=512）

配置：`configs/train_resnet50_arcface_frozen.json`，日志：`work_dirs/frozen_head_train.log`（约 4.7 分钟）

| Epoch | Loss | Train Acc | Val Acc |
|-------|------|-----------|---------|
| 1 | 32.86 | 9.57% | 16.27% |
| 6 | 0.08 | 99.15% | 18.45% |

权重：`work_dirs/resnet50_arcface_frozen/best.pth`

> **LFW**：使用 InsightFace `w600k_r50.onnx` + 官方 `pairs.txt` 10-fold 验证，准确率 **95.82%**。详见 `reports/lfw_verification_report.md`。
