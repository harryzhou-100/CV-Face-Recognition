"""StarGAN v1 Generator and Discriminator (CelebA, 128x128)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(dim, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(dim, affine=True),
        )

    def forward(self, x):
        return x + self.block(x)


class Generator(nn.Module):
    """Encode image + domain vector -> translated image."""

    def __init__(self, conv_dim=64, c_dim=5, repeat_num=6):
        super().__init__()
        layers = [
            nn.Conv2d(3 + c_dim, conv_dim, 7, 1, 3, bias=False),
            nn.InstanceNorm2d(conv_dim, affine=True),
            nn.ReLU(inplace=True),
        ]
        curr = conv_dim
        for _ in range(2):
            layers += [
                nn.Conv2d(curr, curr * 2, 4, 2, 1, bias=False),
                nn.InstanceNorm2d(curr * 2, affine=True),
                nn.ReLU(inplace=True),
            ]
            curr *= 2
        for _ in range(repeat_num):
            layers.append(ResidualBlock(curr))
        for _ in range(2):
            layers += [
                nn.ConvTranspose2d(curr, curr // 2, 4, 2, 1, bias=False),
                nn.InstanceNorm2d(curr // 2, affine=True),
                nn.ReLU(inplace=True),
            ]
            curr //= 2
        layers += [nn.Conv2d(curr, 3, 7, 1, 3), nn.Tanh()]
        self.main = nn.Sequential(*layers)

    def forward(self, x, c):
        c = c.view(c.size(0), c.size(1), 1, 1).expand(-1, -1, x.size(2), x.size(3))
        return self.main(torch.cat([x, c], dim=1))


class Discriminator(nn.Module):
    """PatchGAN-style discriminator with domain classification head."""

    def __init__(self, image_size=128, conv_dim=64, c_dim=5, repeat_num=6):
        super().__init__()
        layers = [
            nn.Conv2d(3, conv_dim, 4, 2, 1),
            nn.LeakyReLU(0.01, inplace=True),
        ]
        curr = conv_dim
        for _ in range(1):
            layers += [
                nn.Conv2d(curr, curr * 2, 4, 2, 1, bias=False),
                nn.InstanceNorm2d(curr * 2, affine=True),
                nn.LeakyReLU(0.01, inplace=True),
            ]
            curr *= 2
        for _ in range(repeat_num):
            layers.append(ResidualBlock(curr))
        self.main = nn.Sequential(*layers)
        self.adv = nn.Conv2d(curr, 1, 4, 1, 0, bias=False)
        self.cls = nn.Conv2d(curr, c_dim, 4, 1, 0, bias=False)

    def forward(self, x):
        h = self.main(x)
        return self.adv(h), self.cls(h)
