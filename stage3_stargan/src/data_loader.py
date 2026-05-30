"""CelebA dataset for StarGAN (aligned faces + binary attributes)."""
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def get_celeba_transform(image_size, mode):
    if mode == "train":
        return transforms.Compose([
            transforms.Resize(int(image_size * 1.12)),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])


class CelebADataset(Dataset):
    """Loads img_align_celeba with selected binary attributes (-1/1 -> 0/1)."""

    def __init__(self, root, attr_path, selected_attrs, transform, partition="train"):
        self.root = Path(root)
        self.img_dir = self.root / "img_align_celeba"
        self.transform = transform
        self.selected_attrs = selected_attrs
        self.partition = partition

        with open(attr_path, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f.readlines()]
        num_images = int(lines[0])
        attr_names = lines[1].split()
        self.attr2idx = {name: i for i, name in enumerate(attr_names)}

        indices = [self.attr2idx[a] for a in selected_attrs]
        partition_path = self.root / "list_eval_partition.txt"
        part_map = {}
        if partition_path.exists():
            with open(partition_path, "r", encoding="utf-8") as f:
                for ln in f:
                    name, part = ln.strip().split()
                    part_map[name] = int(part)

        self.samples = []
        for ln in lines[2:]:
            parts = ln.split()
            if len(parts) < len(attr_names) + 1:
                continue
            fname = parts[0]
            attrs_raw = [int(x) for x in parts[1:]]
            attrs = [(attrs_raw[i] + 1) // 2 for i in indices]

            if partition_path.exists():
                part = part_map.get(fname, 0)
                if partition == "train" and part == 2:
                    continue
                if partition == "val" and part != 2:
                    continue
            self.samples.append((fname, attrs))

        if not partition_path.exists() and partition == "val":
            n = len(self.samples)
            self.samples = self.samples[int(0.95 * n):]
        elif not partition_path.exists() and partition == "train":
            n = len(self.samples)
            self.samples = self.samples[: int(0.95 * n)]

        if len(self.samples) == 0:
            raise RuntimeError(f"No images for partition={partition} under {self.root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fname, label = self.samples[idx]
        path = self.img_dir / fname
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.float32)


def get_loader(root, attr_path, selected_attrs, image_size, batch_size, mode, num_workers=4):
    transform = get_celeba_transform(image_size, mode)
    dataset = CelebADataset(
        root=root,
        attr_path=attr_path,
        selected_attrs=selected_attrs,
        transform=transform,
        partition=mode,
    )
    shuffle = mode == "train"
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(mode == "train"),
    )
