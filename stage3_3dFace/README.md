# Stage 3：单张图像 3D 人脸重建与可视化

基于 **[3DDFA_V2](https://github.com/cleardusk/3DDFA_V2)** 从单张 RGB 图像回归 3DMM 参数并重建稠密 3D 人脸，使用 **PyTorch3D / OpenGL (pyrender) / Matplotlib** 进行渲染与多角度可视化。

## 目录结构

```text
stage3_3dFace/
├── configs/pipeline.json
├── src/                    # 重建封装、渲染调度
├── scripts/
│   ├── setup_third_party.sh
│   ├── prepare_samples.py
│   ├── reconstruct.py
│   ├── analyze_results.py
│   └── run_pipeline.sh
├── third_party/3DDFA_V2/
├── data/samples/
├── outputs/                # 模型与渲染结果
└── reports/                # 思路设计 + 实验报告 + 图表
```

## 环境

```bash
pip install -r stage3_3dFace/requirements.txt
# 可选 GPU 渲染
# pip install "git+https://github.com/facebookresearch/pytorch3d.git@v0.7.6"
# pip install pyrender PyOpenGL
```

## 一键运行

```bash
cd /root/workspace/CV-Face-Recognition
bash stage3_3dFace/scripts/run_pipeline.sh
```

## 分步执行

```bash
bash stage3_3dFace/scripts/setup_third_party.sh
python3 stage3_3dFace/scripts/prepare_samples.py
python3 stage3_3dFace/scripts/reconstruct.py
python3 stage3_3dFace/scripts/analyze_results.py
```

## 交付物

| 类型 | 路径 |
|------|------|
| 重建代码 | `src/`, `scripts/` |
| 思路设计 | `reports/design.md` |
| 3D 模型 | `outputs/meshes/*.obj`, `*.ply` |
| 渲染结果 | `outputs/renders/` |
| 多角度图 | `outputs/multiview/` |
| 分析图表 | `reports/figures/` |
| 实验报告 | `reports/experiment_report.md` |
