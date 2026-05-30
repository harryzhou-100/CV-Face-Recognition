#!/usr/bin/env python3
"""Plot training curves for experiment report."""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
hist_path = ROOT / "stage3_stargan/reports/train_history.json"
out_dir = ROOT / "stage3_stargan/reports/figures"
out_dir.mkdir(parents=True, exist_ok=True)

if not hist_path.exists():
    print(f"No history at {hist_path}")
    sys.exit(0)

with open(hist_path, "r", encoding="utf-8") as f:
    h = json.load(f)

fig, ax = plt.subplots(1, 1, figsize=(8, 4))
steps = range(len(h["d_loss"]))
ax.plot(steps, h["d_loss"], label="D loss", alpha=0.8)
ax.plot(steps, h["g_loss"], label="G loss", alpha=0.8)
ax.set_xlabel("Log interval")
ax.set_ylabel("Loss")
ax.set_title("StarGAN Training Loss")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(out_dir / "training_loss.png", dpi=150)
plt.close()
print(f"Saved {out_dir / 'training_loss.png'}")
