# Stage 3: StarGAN + CelebA 人脸属性编辑

基于 **StarGAN v1** 在 **CelebA** 上训练多属性人脸编辑模型，支持 **黑发 / 金发 / 棕发 / 性别 / 年龄（年轻）** 五维二值属性切换，并完成 **FID、IS** 质量评估与实验报告。

## 目录结构

```text
stage3_stargan/
├── configs/train_celeba_stargan.json   # 超参与路径
├── src/                                # Generator、Discriminator、Solver、DataLoader
├── scripts/
│   ├── prepare_celeba.py               # 下载 CelebA
│   ├── train.py                        # 训练
│   ├── generate_edits.py               # 属性编辑可视化
│   ├── evaluate_quality.py             # FID / IS
│   ├── plot_training.py                # 训练曲线
│   ├── write_experiment_report.py      # 实验报告
│   └── run_pipeline.sh                 # 一键流水线
├── data/celeba/                        # 数据集
├── checkpoints/                        # 模型权重
├── outputs/
│   ├── samples/                        # 训练中采样
│   ├── edited/                         # 编辑结果（按属性分子目录）
│   └── fid_fakes/                      # 评估用生成图
└── reports/
    ├── experiment_report.md            # 实验报告（含代码段与图表）
    ├── quality_evaluation_report.md
    └── figures/
```

## 环境

```bash
pip install -r stage3_stargan/requirements.txt
```

需要 CUDA GPU（推荐 ≥16GB 显存）。默认训练 **3 个 epoch**（可在 `configs/train_celeba_stargan.json` 中修改 `num_epochs`）。

## 运行

```bash
cd /workspace/CV-Face-Recognition   # 项目在 /dev/md0 (150G)，避免根盘 30G 写满
bash stage3_stargan/scripts/run_pipeline.sh
```

或分步：

```bash
python3 stage3_stargan/scripts/prepare_celeba.py
python3 stage3_stargan/scripts/train.py
python3 stage3_stargan/scripts/generate_edits.py
python3 stage3_stargan/scripts/evaluate_quality.py
python3 stage3_stargan/scripts/plot_training.py
python3 stage3_stargan/scripts/write_experiment_report.py
```

## 输出说明

| 路径 | 内容 |
|------|------|
| `reports/experiment_report.md` | 思路、关键代码、训练曲线、编辑图、FID/IS 分析 |
| `reports/quality_evaluation_report.md` | FID、IS 数值表 |
| `outputs/edited/<Attr>/` | 各属性编辑前后对比图 |
