# LFW 人脸识别验证报告（ResNet50 + ArcFace）

- 推理后端: **PyTorch CUDA (onnx2torch, w600k_r50.onnx) + TTA flip**
- 训练数据: WebFace600K 预训练（InsightFace buffalo_l / w600k_r50）
- 模型: `/root/.insightface/models/buffalo_l/w600k_r50.onnx`
- LFW 目录: `/root/CV-Face-Recognition/stage2_recognition/data/lfw_flat`
- 配对协议: `/root/CV-Face-Recognition/stage2_recognition/data/sklearn_cache/lfw_home/pairs.txt`
- 预处理: InsightFace 官方 `(pixel - 127.5) / 127.5`，BGR→RGB（`blobFromImages`）

## 验证结果

| 指标 | 数值 |
|------|------|
| **准确率 (Accuracy)**（LFW **10-fold** 交叉验证均值） | **95.82%** |
| 最优阈值 | 0.2300 |
| 精确率 | 97.38% |
| 召回率 | 94.33% |
| F1 | 95.83% |
| ROC-AUC | 0.9885 |

| 无约束全局阈值准确率 | 95.90% |
| 10-fold 各折准确率 | 96.00%, 94.67%, 94.83%, 96.50%, 94.83%, 96.17%, 95.33%, 95.83%, 97.83%, 96.17% |

| 图像数 | 13233 |
| 身份数 | 5749 |
| 有效配对数 | 6000 |
| 跳过（读图失败） | 0 |

### 混淆矩阵

| | 预测: 不同人 | 预测: 同一人 |
|---|-------------|-------------|
| 真实: 不同人 | 2924 | 76 |
| 真实: 同一人 | 170 | 2830 |
