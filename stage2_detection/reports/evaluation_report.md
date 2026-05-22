# WIDER FACE 人脸检测评估报告

- 模型: RetinaNet R50-FPN
- 权重: `stage2_detection/work_dirs/retinanet_r50_wider_face/epoch_6.pth`
- 验证集: WIDER FACE val (3226 张图)

## 核心指标

| 指标 | 数值 |
|------|------|
| mAP (COCO bbox, IoU 0.5:0.95) | 0.281 |
| mAP@0.5 (COCO) | 0.557 |
| Average Recall (AR@0.5:0.95, maxDets=100) | 0.315 |
| mAP@0.5 可视为 IoU=0.5 下的检测精度参考 | 0.557 |

## 按尺度 mAP（COCO 标准）

| 尺度 | mAP |
|------|-----|
| small | 0.169 |
| medium | 0.561 |
| large | 0.654 |

## COCO 详细指标

```json
{
  "coco/face_precision": 0.293,
  "coco/bbox_mAP": 0.281,
  "coco/bbox_mAP_50": 0.557,
  "coco/bbox_mAP_75": 0.281,
  "coco/bbox_mAP_s": 0.169,
  "coco/bbox_mAP_m": 0.561,
  "coco/bbox_mAP_l": 0.654
}
```