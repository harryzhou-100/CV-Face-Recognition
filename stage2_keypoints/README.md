# Stage 2 Keypoints: 300W 人脸关键点与对齐

基于 **MMPose** 在 **300W**（68 点）上训练 **HRNet-W18**，完成关键点检测、**NME** 评估与 **仿射人脸对齐**。

## 目录

```text
stage2_keypoints/
├── configs/hrnetv2_w18_300w_256x256.py
├── scripts/
│   ├── prepare_300w.py      # 下载标注与图像
│   ├── train.py
│   ├── evaluate.py          # 测试集 NME
│   ├── align_faces.py       # 推理 + 仿射对齐
│   └── face_align_utils.py
├── data/300w/
├── models/                  # 可放置导出权重副本
├── work_dirs/
├── reports/
└── outputs/{aligned,landmarks,comparisons}/
```

## 当前训练结果

| 集合 | NME |
|------|-----|
| valid（训练时最优） | **0.0349**（epoch 20） |
| test | **0.0417** |

权重：`work_dirs/hrnetv2_w18_300w/best_NME_epoch_20.pth`（副本见 `models/hrnetv2_w18_300w_best.pth`）

## 快速开始

```bash
cd /root/CV-Face-Recognition
pip install -r stage2_keypoints/requirements.txt
pip install https://download.openmmlab.com/mmcv/dist/cu121/torch2.1.0/mmcv-2.1.0-cp310-cp310-manylinux1_x86_64.whl

bash stage2_keypoints/scripts/run_pipeline.sh
```

或分步：

```bash
python3 stage2_keypoints/scripts/prepare_300w.py
python3 stage2_keypoints/scripts/train.py --epochs 20
python3 stage2_keypoints/scripts/evaluate.py --split test
python3 stage2_keypoints/scripts/align_faces.py
```

## 人脸对齐说明

- 68 点预测后取 5 点（双眼中心、鼻尖、嘴角）估计 **相似仿射变换**
- 参考模板为 112×112 标准五点布局，缩放到 `256×256` 输出
- 对比图保存在 `outputs/comparisons/`（原图 | 关键点 | 对齐）

## 评估指标

- **NME**（Normalized Mean Error）：以眼间距归一化，在 `reports/evaluation_report.md` 中记录
