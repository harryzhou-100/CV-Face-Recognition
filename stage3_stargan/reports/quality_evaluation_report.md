# StarGAN 生成图像质量评估

## 指标

| 指标 | 数值 | 说明 |
|------|------|------|
| FID ↓ | **361.60** | 相对 CelebA 验证集真实分布（Inception 特征，越低越好） |
| IS ↑ | **1.82 ± 0.06** | 生成图 Inception Score（越高越好） |

- 评估样本数：真实 1000 / 生成 1000
- 权重：`2400-G.pth`

## 参考

- 原始 StarGAN 论文在 CelebA 上 FID 约 15–25（视训练迭代与实现而定）
- IS 对人脸域通常低于 ImageNet 自然图像（论文 IS 约 2–3）

## 生成假样本目录

`stage3_stargan/outputs/fid_fakes/`
