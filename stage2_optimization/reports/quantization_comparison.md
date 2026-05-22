# ResNet50 动态量化对比（交付表）

| 指标 | FP32（原始） | INT8 动态量化 | 变化 |
|------|-------------|---------------|------|
| 权重文件大小 (MB) | 166.58 | 129.86 | ↓ 22.0% |
| 单张推理耗时 (ms/张) | 3.458 | 265.302 | ↑ 7571.7% |
| LFW 10-fold 准确率 (%) | 75.50 ± 0.00 | 75.67 ± 0.00 | +0.17 pp |

- 基准权重: `/root/CV-Face-Recognition/stage2_recognition/work_dirs/resnet50_arcface_frozen/best.pth`
- 量化权重: `/root/CV-Face-Recognition/stage2_optimization/models/resnet50_int8_dynamic.pth`
- LFW: `/root/CV-Face-Recognition/stage2_recognition/data/lfw_flat` (600 pairs)
- 测速样本: 150 张, FP32 device=cuda:0, INT8 device=cpu
