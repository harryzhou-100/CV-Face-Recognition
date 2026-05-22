#!/usr/bin/env python3
"""Train ResNet50 + ArcFace on MS-Celeb-1M subset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STAGE2_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STAGE2_ROOT / "scripts"))
sys.path.insert(0, str(STAGE2_ROOT / "models"))

from dataset import MS1MLabelTxtDataset  # noqa: E402
from arcface_head import ResNet50ArcFace  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=STAGE2_ROOT / "configs" / "train_resnet50_arcface.json",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=STAGE2_ROOT / "work_dirs" / "resnet50_arcface",
    )
    args = parser.parse_args()

    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    data_root = (PROJECT_ROOT / cfg["data_root"]).resolve()
    work_dir = args.work_dir
    work_dir.mkdir(parents=True, exist_ok=True)
    curves_dir = STAGE2_ROOT / "outputs" / "curves"
    curves_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset = MS1MLabelTxtDataset(
        data_root, image_size=cfg["image_size"], augment=True
    )
    n_val = max(2000, int(0.02 * len(dataset)))
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )
    train_loader = DataLoader(
        train_set,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=True,
    )

    model = ResNet50ArcFace(
        num_classes=dataset.num_classes,
        embedding_size=cfg.get("embedding_size", 512),
        s=cfg.get("arcface_s", 64.0),
        m=cfg.get("arcface_m", 0.5),
        pretrained=cfg.get("pretrained", True),
        pretrained_path=cfg.get("pretrained_checkpoint"),
        freeze_backbone=cfg.get("freeze_backbone", False),
    ).to(device)
    trainable = model.trainable_parameters()
    n_trainable = sum(p.numel() for p in trainable)
    print(f"[train] Trainable parameters: {n_trainable:,}")
    optimizer = torch.optim.SGD(
        trainable,
        lr=cfg["lr"],
        momentum=0.9,
        weight_decay=cfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["epochs"]
    )
    criterion = nn.CrossEntropyLoss()

    history = {"epoch": [], "train_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0.0

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        if model.backbone_frozen:
            model.backbone.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            if model.backbone_frozen:
                with torch.no_grad():
                    emb = model.forward_features(images)
                logits = model.arcface(emb, labels)
            else:
                logits = model(images, labels)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += images.size(0)
        scheduler.step()
        train_loss = running_loss / total
        train_acc = correct / total

        model.eval()
        if model.backbone_frozen:
            model.backbone.eval()
        v_correct = v_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                if model.backbone_frozen:
                    emb = model.forward_features(images)
                    logits = model.arcface(emb, labels)
                else:
                    logits = model(images, labels)
                pred = logits.argmax(dim=1)
                v_correct += (pred == labels).sum().item()
                v_total += images.size(0)
        val_acc = v_correct / max(v_total, 1)

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch}/{cfg['epochs']} "
            f"loss={train_loss:.4f} train_acc={train_acc:.4f} val_acc={val_acc:.4f}"
        )

        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "num_classes": dataset.num_classes,
            "label_map": dataset.label_map,
            "config": cfg,
        }
        torch.save(ckpt, work_dir / "last.pth")
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save(ckpt, work_dir / "best.pth")
            import shutil

            shutil.copy(work_dir / "best.pth", STAGE2_ROOT / "models" / "resnet50_arcface_best.pth")

    (curves_dir / "history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    cfg_out = work_dir / "config_used.json"
    cfg_out.write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    # Plot curves
    try:
        import matplotlib.pyplot as plt

        fig, ax1 = plt.subplots(figsize=(8, 5))
        ax1.plot(history["epoch"], history["train_loss"], "b-o", label="Train Loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.grid(True, alpha=0.3)
        ax2 = ax1.twinx()
        ax2.plot(history["epoch"], history["train_acc"], "g-s", label="Train Acc")
        ax2.plot(history["epoch"], history["val_acc"], "r-^", label="Val Acc")
        ax2.set_ylabel("Accuracy")
        ax2.set_ylim(0, 1.05)
        lines1, lab1 = ax1.get_legend_handles_labels()
        lines2, lab2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, lab1 + lab2, loc="center right")
        plt.title("ResNet50 + ArcFace on MS1MV3 subset")
        plt.tight_layout()
        plt.savefig(curves_dir / "loss_accuracy_curves.png", dpi=150)
        plt.close()
    except Exception as e:
        print(f"[warn] plot failed: {e}")

    print(f"Best val acc: {best_val_acc:.4f}")
    print(f"Checkpoint: {work_dir / 'best.pth'}")


if __name__ == "__main__":
    main()
