# Stage 2 Optimization：ResNet50 量化与 ONNX 导出

在 `stage2_recognition` 训练权重基础上，对 **IResNet50 嵌入骨干** 做部署优化。

## 目录

```text
stage2_optimization/
├── scripts/
│   ├── model_utils.py          # 加载 FP32 / INT8 骨干
│   ├── quantize.py             # 动态 INT8 量化并保存
│   ├── compare_quantization.py # FP32 vs INT8 对比表
│   └── export_onnx.py          # 导出 ONNX
├── models/                     # 量化权重、ONNX
└── reports/
    └── quantization_comparison.md
```

## 快速运行

```bash
cd /root/CV-Face-Recognition

# 1. 动态 INT8 量化
python3 stage2_optimization/scripts/quantize.py

# 2. FP32 vs INT8 对比（体积 / 速度 / LFW 准确率）
python3 stage2_optimization/scripts/compare_quantization.py

# 3. 导出 ONNX（112×112 → 512-d 归一化嵌入）
python3 stage2_optimization/scripts/export_onnx.py
```

默认 FP32 权重：`stage2_recognition/work_dirs/resnet50_arcface_frozen/best.pth`（仅导出/量化其中的 `backbone.*`）。

## 说明

- **量化对象**：`Linear` 层动态 INT8（PyTorch `quantize_dynamic`）；卷积保持 FP32。
- **ONNX**：输入 `input` `[N,3,112,112]`，输出 `embedding` `[N,512]`（L2 归一化）。
- **准确率**：与 recognition 阶段一致，使用官方 `pairs.txt` + InsightFace 预处理 + L2 距离 10-fold。
