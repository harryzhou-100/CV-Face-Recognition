# Stage 3 Live Filter：实时人脸关键点与动态特效

基于 **MediaPipe Face Landmarker**（478 点 + 52 Blendshapes）的实时人脸滤镜系统，支持动态贴纸、美颜美妆与表情驱动互动特效。

## 功能

| 模块 | 特效 |
|------|------|
| 贴纸 | 眼镜、帽子、猫耳（RGBA 仿射对齐） |
| 美颜 | 磨皮（bilateral）、美白（LAB）、口红（嘴唇 mask） |
| 表情 | 微笑爱心、眨眼 sparkle、张嘴音符 |

## 移动端优化

- `detect_scale`：检测降采样（默认 0.5）
- `detect_every`：跳帧检测，复用上一帧关键点
- ROI 局部美颜，避免全帧滤波
- float16 模型（3.6 MB）

## 快速开始

```bash
cd stage3_livefilter
pip install -r requirements.txt

# 生成贴纸资源
python scripts/generate_stickers.py

# 下载模型（首次自动下载）
# models/face_landmarker.task

# 生成演示视频
python scripts/run_demo_video.py

# 性能测试 + 报告
python scripts/benchmark.py
python scripts/generate_report.py

# 实时摄像头
python scripts/run_live.py --preset full --detect-scale 0.5
python scripts/run_live.py --beauty --glasses --expression
```

## 目录

```text
stage3_livefilter/
├── src/                  # 检测 + 特效 + 管线
├── scripts/              # 运行脚本
├── assets/stickers/      # PNG 贴纸
├── models/               # face_landmarker.task
├── outputs/demo/         # 演示视频
├── outputs/benchmark/    # 性能数据
└── reports/              # 实验报告 + 图表
```

## 交付物

- [x] 实时人脸关键点检测和动态特效代码 (`src/`)
- [x] 特效演示视频 (`outputs/demo/demo_video.mp4`)
- [x] 性能分析 CPU/GPU 帧率 (`outputs/benchmark/benchmark_results.json`)
- [x] 实验报告 (`reports/experiment_report.md`)
