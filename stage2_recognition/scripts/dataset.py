"""Face recognition training dataset from MS1M-style label.txt."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class MS1MLabelTxtDataset(Dataset):
    def __init__(
        self,
        data_root: str | Path,
        label_file: str = "label.txt",
        image_size: int = 112,
        augment: bool = True,
    ) -> None:
        self.root = Path(data_root)
        self.image_size = image_size
        self.augment = augment
        self.samples: list[tuple[str, int]] = []
        label_path = self.root / label_file
        with label_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                path, lid = line.split("\t")
                self.samples.append((path, int(lid)))
        self.labels = sorted({lid for _, lid in self.samples})
        self.label_map = {lid: i for i, lid in enumerate(self.labels)}
        self.num_classes = len(self.labels)

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image(self, rel_path: str) -> np.ndarray:
        p = self.root / rel_path
        img = cv2.imread(str(p))
        if img is None:
            raise FileNotFoundError(p)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def _transform(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        if self.augment:
            if np.random.rand() < 0.5:
                img = np.fliplr(img).copy()
            scale = np.random.uniform(0.9, 1.1)
            nh, nw = int(h * scale), int(w * scale)
            img = cv2.resize(img, (nw, nh))
            h, w = img.shape[:2]
        size = self.image_size
        if h != size or w != size:
            img = cv2.resize(img, (size, size))
        img = img.astype(np.float32) / 255.0
        img = (img - 0.5) / 0.5
        return img.transpose(2, 0, 1)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        rel, lid = self.samples[idx]
        img = self._load_image(rel)
        img = self._transform(img)
        return torch.from_numpy(img), self.label_map[lid]
