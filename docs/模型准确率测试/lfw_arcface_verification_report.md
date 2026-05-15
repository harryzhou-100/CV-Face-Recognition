# LFW 人脸识别验证报告（ArcFace / InsightFace）

- 生成时间: 2026-05-15 18:05:11
- 数据目录: `/Users/apple/Projects/data/lfw/lfw`
- 识别模型: InsightFace `buffalo_l`（ArcFace 特征）
- 验证协议: 随机采样同一人 / 不同人配对（各 1500 对）

## 1. 数据概况

| 指标 | 数值 |
|------|------|
| 图像总数 | 4,857 |
| 身份数（由文件名解析） | 1,680 |
| 有效验证对数 | 2,984 |
| 跳过（未检测到人脸） | 16 |
| 成功提取特征图像数 | 3,018 |

## 2. 验证结果

| 指标 | 数值 |
|------|------|
| **准确率 (Accuracy)** | **98.46%** |
| 最优阈值 (cosine) | 0.2450 |
| 精确率 (Precision) | 100.00% |
| 召回率 (Recall) | 96.92% |
| F1 分数 | 98.43% |
| ROC-AUC | 0.9839 |

### 混淆矩阵（阈值 = 最优阈值）

| | 预测: 不同人 | 预测: 同一人 |
|---|-------------|-------------|
| 真实: 不同人 | 1,492 (TN) | 0 (FP) |
| 真实: 同一人 | 46 (FN) | 1,446 (TP) |

## 3. 说明

- **同一人**：文件名主体相同（如 `Tom_Cruise_0001` 与 `Tom_Cruise_0002`）。
- **不同人**：随机抽取两个不同主体各一张图。
- 相似度为 ArcFace 特征向量余弦相似度；大于阈值判为同一人。
- 当前 LFW 为扁平目录（`*.jpg`），若使用官方完整 LFW 与 `pairs.txt`，
  可将 `--pairs-file` 指向官方协议文件以复现标准 benchmark。

## 4. 复现命令

```bash
conda activate face_cv
python scripts/lfw_arcface_verify.py \
  --data-root /Users/apple/Projects/data/lfw/lfw \
  --report /Users/apple/Projects/Bytedance-CV-Project/docs/outputs/lfw_arcface_verification_report.md
```
