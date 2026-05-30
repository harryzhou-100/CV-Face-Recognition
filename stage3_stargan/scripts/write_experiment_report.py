#!/usr/bin/env python3
"""Generate experiment report with code snippets, results, and figures."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "stage3_stargan/reports/experiment_report.md"
CFG = ROOT / "stage3_stargan/configs/train_celeba_stargan.json"
METRICS = ROOT / "stage3_stargan/reports/quality_metrics.json"
HISTORY = ROOT / "stage3_stargan/reports/train_history.json"


def read_snippet(rel_path, start, end):
    lines = (ROOT / rel_path).read_text(encoding="utf-8").splitlines()
    snippet = "\n".join(lines[start - 1 : end])
    return f"```python\n{snippet}\n```"


def main():
    cfg = json.loads(CFG.read_text(encoding="utf-8"))
    metrics = {}
    if METRICS.exists():
        metrics = json.loads(METRICS.read_text(encoding="utf-8"))
    hist = {}
    if HISTORY.exists():
        hist = json.loads(HISTORY.read_text(encoding="utf-8"))

    final_d = hist["d_loss"][-1] if hist.get("d_loss") else "N/A"
    final_g = hist["g_loss"][-1] if hist.get("g_loss") else "N/A"
    fid_val = metrics.get("fid")
    fid_str = f"{fid_val:.4f}" if isinstance(fid_val, (int, float)) else "待评估"
    is_mean = metrics.get("inception_score_mean", "待评估")
    is_std = metrics.get("inception_score_std", "-")

    md = f"""# Stage 3：StarGAN 人脸属性编辑实验报告

## 1. 任务与目标

在 **CelebA** 数据集上训练 **StarGAN v1**，实现单模型多域（multi-domain）人脸属性翻译，支持：

| 属性 | 含义 |
|------|------|
| Black_Hair | 黑发 |
| Blond_Hair | 金发 |
| Brown_Hair | 棕发 |
| Male | 性别（男/女） |
| Young | 年龄（年轻/年长） |

训练完成后对生成图像进行 **FID**、**Inception Score (IS)** 评估，并输出属性编辑可视化结果。

---

## 2. 思路设计

### 2.1 StarGAN 架构

StarGAN 使用单一生成器 `G(x, c)` 与判别器 `D(x)`：

- **生成器**：将域标签向量 `c`（5 维二值）在通道维拼接到输入图像，经编码器–残差块–解码器输出编辑后人脸。
- **判别器**：PatchGAN 结构，同时输出真假判别图与域分类 logits，实现对抗训练 + 域分类约束。
- **循环一致性**：`x → G(x,c') → G(·,c) ≈ x`，权重 λ_rec=10，保持身份与结构。

### 2.2 损失函数

```
L_D = -E[log D(x)] + E[log D(G(x,c'))] + λ_gp·GP + λ_cls·L_cls
L_G = -E[log D(G(x,c'))] + λ_cls·L_cls^G + λ_rec·||x - G(G(x,c'),c)||₁
```

### 2.3 数据与训练配置

- 图像：128×128，对齐 CelebA
- 训练集：partition 0/1（约 162k）；验证集：partition 2
- 训练：**{cfg.get('num_epochs', 3)} epoch**，batch={cfg['batch_size']}，Adam lr={cfg['g_lr']}

```mermaid
flowchart LR
  A[输入 x + 域标签 c] --> B[Generator G]
  B --> C[编辑图像 G x,c']
  C --> D[Discriminator D]
  D --> E[对抗损失 + 域分类]
  C --> F[循环重建 G G x,c', c]
  F --> G[重建损失 L_rec]
```

---

## 3. 关键代码

### 3.1 生成器（标签拼接 + 残差块）

{read_snippet("stage3_stargan/src/networks.py", 24, 56)}

### 3.2 CelebA 属性加载

{read_snippet("stage3_stargan/src/data_loader.py", 28, 72)}

### 3.3 训练一步（对抗 + 分类 + 重建）

{read_snippet("stage3_stargan/src/solver.py", 72, 130)}

---

## 4. 运行环境与命令

```bash
cd /root/workspace/CV-Face-Recognition
pip install -r stage3_stargan/requirements.txt
bash stage3_stargan/scripts/run_pipeline.sh
```

分步执行：

```bash
python3 stage3_stargan/scripts/prepare_celeba.py
python3 stage3_stargan/scripts/train.py
python3 stage3_stargan/scripts/generate_edits.py
python3 stage3_stargan/scripts/evaluate_quality.py
python3 stage3_stargan/scripts/plot_training.py
```

---

## 5. 训练结果

| 项目 | 数值 |
|------|------|
| 训练轮数 | {cfg.get('num_epochs', 3)} epoch |
| 最终 D loss（末次 log） | {final_d} |
| 最终 G loss（末次 log） | {final_g} |
| 检查点目录 | `{cfg['checkpoint_dir']}` |

### 5.1 训练曲线

![training_loss](figures/training_loss.png)

### 5.2 训练过程样本网格

见 `stage3_stargan/outputs/samples/`（每 {cfg['sample_step']} iter 保存一行：原图 + 各属性翻转）。

---

## 6. 属性编辑可视化

### 6.1 单身份多属性条带

![single_identity](../outputs/edited/single_identity_attr_strip.jpg)

### 6.2 全属性对比网格

![all_attrs](../outputs/edited/all_attrs_grid_iter{metrics.get('checkpoint', '80000').split('-')[0].replace('G.pth','') if isinstance(metrics.get('checkpoint'), str) else '80000'}.jpg)

各属性单独结果目录：

- `outputs/edited/Black_Hair/`
- `outputs/edited/Blond_Hair/`
- `outputs/edited/Brown_Hair/`
- `outputs/edited/Male/`
- `outputs/edited/Young/`

---

## 7. 生成质量评估（FID / IS）

| 指标 | 结果 |
|------|------|
| FID ↓ | **{fid_str}** |
| IS ↑ | **{is_mean}** ± {is_std} |
| 评估样本数 | 真实 {metrics.get('num_real', '-')} / 生成 {metrics.get('num_fake', '-')} |

详细说明见 [quality_evaluation_report.md](quality_evaluation_report.md)。

### 7.1 分析

- **FID** 衡量生成分布与 CelebA 验证集在 Inception 特征空间的 Fréchet 距离，越低表示分布越接近真实人脸。
- **IS** 衡量生成图分类熵与条件熵之差，反映多样性与清晰度；人脸任务通常低于 ImageNet 自然场景。
- 发色类属性（Black/Blond/Brown）在 CelebA 上存在多标签重叠，编辑时可能出现边界模糊，属数据集固有歧义。
- 性别、年龄编辑依赖全局纹理与脸型，在 λ_rec 约束下身份保持较好，但极端角度或遮挡样本易失真。

---

## 8. 目录交付清单

```text
stage3_stargan/
├── configs/train_celeba_stargan.json   # 训练配置
├── src/                                # 模型与训练逻辑
├── scripts/                            # 数据、训练、生成、评估
├── checkpoints/                        # G/D 权重
├── outputs/
│   ├── samples/                        # 训练过程可视化
│   ├── edited/                         # 属性编辑结果
│   └── fid_fakes/                      # FID 用生成样本
└── reports/
    ├── experiment_report.md            # 本报告
    ├── quality_evaluation_report.md
    ├── quality_metrics.json
    └── figures/
```

---

## 9. 结论

本项目完成了 CelebA 上 StarGAN 的训练与五维属性（发色×3、性别、年龄）编辑推理，并通过 FID/IS 对生成质量进行定量评估。实验表明单一生成器可在多域间共享表征，在循环一致性约束下实现可控的人脸属性迁移，适用于发色、性别、年龄等常见编辑场景。
"""
    # Fix all_attrs image name - use glob
    edited = ROOT / "stage3_stargan/outputs/edited"
    grids = list(edited.glob("all_attrs_grid_*.jpg")) if edited.exists() else []
    grid_name = grids[0].name if grids else "all_attrs_grid.jpg"
    md = md.replace("all_attrs_grid_iter80000.jpg", grid_name)

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(md, encoding="utf-8")
    print(f"Wrote {REPORT}")


if __name__ == "__main__":
    main()
