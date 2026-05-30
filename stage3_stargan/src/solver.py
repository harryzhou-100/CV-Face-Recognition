"""StarGAN training solver."""
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.utils import save_image

from .networks import Discriminator, Generator


def label2onehot(labels, dim):
    batch = labels.size(0)
    out = torch.zeros(batch, dim, device=labels.device)
    for i in range(batch):
        for j in range(dim):
            if labels[i, j] >= 0.5:
                out[i, j] = 1.0
    return out


def denorm(x):
    return (x + 1) / 2


class Solver:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")
        self.c_dim = cfg["c_dim"]
        self.lambda_cls = cfg["lambda_cls"]
        self.lambda_rec = cfg["lambda_rec"]
        self.n_critic = cfg["n_critic"]

        self.G = Generator(
            conv_dim=cfg["g_conv_dim"],
            c_dim=self.c_dim,
            repeat_num=cfg["g_repeat_num"],
        ).to(self.device)
        self.D = Discriminator(
            image_size=cfg["image_size"],
            conv_dim=cfg["d_conv_dim"],
            c_dim=self.c_dim,
            repeat_num=cfg["d_repeat_num"],
        ).to(self.device)

        self.g_opt = torch.optim.Adam(
            self.G.parameters(), cfg["g_lr"], [cfg["beta1"], cfg["beta2"]]
        )
        self.d_opt = torch.optim.Adam(
            self.D.parameters(), cfg["d_lr"], [cfg["beta1"], cfg["beta2"]]
        )

        self.history = {"d_loss": [], "g_loss": [], "d_cls": [], "g_rec": []}
        self.best_d_loss = float("inf")
        self.grad_clip = cfg.get("grad_clip", 0)
        self.d_loss_abort = cfg.get("d_loss_abort", 0)
        self.d_loss_abort_after = cfg.get("d_loss_abort_after_iter", 0)
        Path(cfg["checkpoint_dir"]).mkdir(parents=True, exist_ok=True)
        Path(cfg["sample_dir"]).mkdir(parents=True, exist_ok=True)

    def restore(self, iters, g_only=False):
        ckpt = Path(self.cfg["checkpoint_dir"]) / f"{iters}-G.pth"
        if ckpt.exists():
            self.G.load_state_dict(torch.load(ckpt, map_location=self.device, weights_only=True))
            if not g_only:
                d_ckpt = Path(self.cfg["checkpoint_dir"]) / f"{iters}-D.pth"
                if d_ckpt.exists():
                    self.D.load_state_dict(torch.load(d_ckpt, map_location=self.device, weights_only=True))
                state_path = Path(self.cfg["checkpoint_dir"]) / f"{iters}-state.pth"
                if state_path.exists():
                    state = torch.load(state_path, map_location=self.device, weights_only=False)
                    self.g_opt.load_state_dict(state["g_opt"])
                    self.d_opt.load_state_dict(state["d_opt"])
                    self.history = state.get("history", self.history)
            print(f"Restored G from {ckpt}" + (" (G only)" if g_only else ""), flush=True)

    def save(self, iters):
        ckpt_dir = Path(self.cfg["checkpoint_dir"])
        torch.save(self.G.state_dict(), ckpt_dir / f"{iters}-G.pth")
        torch.save(self.D.state_dict(), ckpt_dir / f"{iters}-D.pth")
        torch.save(
            {"g_opt": self.g_opt.state_dict(), "d_opt": self.d_opt.state_dict(), "history": self.history},
            ckpt_dir / f"{iters}-state.pth",
        )
        (ckpt_dir / "latest.txt").write_text(str(iters), encoding="utf-8")

    def save_best(self, iters, d_loss):
        if d_loss < self.best_d_loss:
            self.best_d_loss = d_loss
            ckpt_dir = Path(self.cfg["checkpoint_dir"])
            for src_suffix, dst_suffix in (("-G.pth", "-G-best.pth"), ("-D.pth", "-D-best.pth")):
                src = ckpt_dir / f"{iters}{src_suffix}"
                dst = ckpt_dir / dst_suffix
                if src.exists():
                    dst.write_bytes(src.read_bytes())
            print(f"New best checkpoint (D={d_loss:.2f}) saved @ iter {iters}", flush=True)

    def _prune_old_checkpoints(self, keep=5):
        ckpt_dir = Path(self.cfg["checkpoint_dir"])
        tags = sorted({int(p.stem.split("-")[0]) for p in ckpt_dir.glob("*-G.pth")})
        for tag in tags[:-keep]:
            for suffix in ("-G.pth", "-D.pth", "-state.pth"):
                p = ckpt_dir / f"{tag}{suffix}"
                if p.exists():
                    p.unlink()

    def gradient_penalty(self, y_real, y_fake):
        batch = y_real.size(0)
        eps = torch.rand(batch, 1, 1, 1, device=self.device)
        interp = (eps * y_real + (1 - eps) * y_fake).requires_grad_(True)
        out_src, _ = self.D(interp)
        grad = torch.autograd.grad(
            outputs=out_src.sum(),
            inputs=interp,
            create_graph=True,
            retain_graph=True,
        )[0]
        grad = grad.view(batch, -1)
        return ((grad.norm(2, dim=1) - 1) ** 2).mean()

    def classification_loss(self, logit, target):
        if logit.dim() > 2:
            logit = logit.view(logit.size(0), logit.size(1), -1).mean(2)
        return F.binary_cross_entropy_with_logits(logit, target, reduction="mean")

    def train_step(self, x_real, c_org):
        x_real = x_real.to(self.device)
        c_org = c_org.to(self.device)
        batch = x_real.size(0)

        out_src, out_cls = self.D(x_real)
        d_loss_real = -torch.mean(out_src)
        d_loss_cls = self.classification_loss(out_cls, c_org)

        c_trg_list = []
        x_fake_list = []
        for i in range(self.c_dim):
            c_trg = label2onehot(torch.rand(batch, self.c_dim, device=self.device), self.c_dim)
            c_trg[:, i] = 1 - c_org[:, i]
            c_trg_list.append(c_trg)
            x_fake_list.append(self.G(x_real, c_trg))

        x_fake = x_fake_list[-1]
        out_src, _ = self.D(x_fake.detach())
        d_loss_fake = torch.mean(out_src)
        d_loss_gp = self.gradient_penalty(x_real, x_fake.detach())

        d_loss = (
            d_loss_real
            + d_loss_fake
            + self.cfg["lambda_gp"] * d_loss_gp
            + self.lambda_cls * d_loss_cls
        )
        self.d_opt.zero_grad()
        d_loss.backward()
        if self.grad_clip:
            torch.nn.utils.clip_grad_norm_(self.D.parameters(), self.grad_clip)
        self.d_opt.step()

        if self.n_critic > 1 and (self.curr_iters + 1) % self.n_critic != 0:
            return {"d_loss": d_loss.item(), "g_loss": None}

        x_fake = self.G(x_real, c_trg_list[0])
        out_src, out_cls = self.D(x_fake)
        g_loss_fake = -torch.mean(out_src)
        g_loss_cls = self.classification_loss(out_cls, c_trg_list[0])

        x_reconst = self.G(x_fake, c_org)
        g_loss_rec = torch.mean(torch.abs(x_real - x_reconst))

        g_loss = g_loss_fake + self.lambda_cls * g_loss_cls + self.cfg["lambda_rec"] * g_loss_rec
        self.g_opt.zero_grad()
        g_loss.backward()
        if self.grad_clip:
            torch.nn.utils.clip_grad_norm_(self.G.parameters(), self.grad_clip)
        self.g_opt.step()

        return {
            "d_loss": d_loss.item(),
            "g_loss": g_loss.item(),
            "d_cls": d_loss_cls.item(),
            "g_rec": g_loss_rec.item(),
        }

    def update_lr(self, iters):
        decay_start = self.cfg.get("num_iters_decay")
        total = self.cfg.get("num_iters")
        if decay_start is None or total is None or iters < decay_start:
            return
        span = max(total - decay_start, 1)
        for opt in (self.g_opt, self.d_opt):
            for pg in opt.param_groups:
                pg["lr"] -= self.cfg["g_lr"] / span

    @torch.no_grad()
    def sample_grid(self, loader, iters):
        self.G.eval()
        try:
            x, c = next(iter(loader))
        except StopIteration:
            return
        x = x[:8].to(self.device)
        c = c[:8].to(self.device)
        x_fake_list = [x]
        for i in range(self.c_dim):
            c_trg = c.clone()
            c_trg[:, i] = 1 - c[:, i]
            x_fake_list.append(self.G(x, c_trg))
        out = torch.cat(x_fake_list, dim=3)
        save_image(denorm(out), Path(self.cfg["sample_dir"]) / f"{iters}-grid.jpg", nrow=1)
        self.G.train()

    def train(self, loader):
        self.curr_iters = self.cfg.get("resume_iters", 0)
        g_only = self.cfg.get("resume_g_only", False)
        if self.curr_iters:
            self.restore(self.curr_iters, g_only=g_only)

        num_epochs = self.cfg.get("num_epochs", 3)
        steps_per_epoch = len(loader)
        max_steps = self.cfg.get("max_steps_per_epoch") or steps_per_epoch
        total_iters = num_epochs * max_steps
        self.cfg["num_iters"] = total_iters
        if self.cfg.get("num_iters_decay") is None:
            self.cfg["num_iters_decay"] = int(total_iters * 0.66)

        log_step = self.cfg["log_step"]
        sample_step = self.cfg.get("sample_step") or 0
        save_step = self.cfg.get("model_save_step") or 0

        running = {"d": 0.0, "g": 0.0, "n": 0}
        self.G.train()
        self.D.train()

        start_epoch = self.curr_iters // max_steps + 1 if self.curr_iters else 1
        start_batch = self.curr_iters % max_steps

        print(
            f"Training {num_epochs} epoch(s), up to {max_steps} steps/epoch, total {total_iters} iters",
            flush=True,
        )
        if self.curr_iters:
            print(f"Resume @ iter {self.curr_iters} (epoch {start_epoch}, batch {start_batch})", flush=True)

        epoch_d_avg = 0.0
        aborted = False
        keep = self.cfg.get("checkpoint_keep", 5)

        for epoch in range(start_epoch, num_epochs + 1):
            epoch_d_sum, epoch_d_n = 0.0, 0
            for batch_idx, (x, c) in enumerate(loader):
                if epoch == start_epoch and batch_idx < start_batch:
                    continue
                if batch_idx >= max_steps:
                    break
                metrics = self.train_step(x, c)
                self.curr_iters += 1
                self.update_lr(self.curr_iters)

                d_val = metrics["d_loss"]
                if (
                    self.d_loss_abort
                    and self.curr_iters >= self.d_loss_abort_after
                    and d_val > self.d_loss_abort
                ):
                    print(
                        f"ABORT: D loss {d_val:.1f} > {self.d_loss_abort}, stopping to avoid collapse.",
                        flush=True,
                    )
                    self.save(self.curr_iters)
                    aborted = True
                    break

                if metrics["g_loss"] is not None:
                    running["d"] += d_val
                    running["g"] += metrics["g_loss"]
                    running["n"] += 1
                    epoch_d_sum += d_val
                    epoch_d_n += 1

                if self.curr_iters % log_step == 0 and running["n"] > 0:
                    avg_d = running["d"] / running["n"]
                    avg_g = running["g"] / running["n"]
                    self.history["d_loss"].append(avg_d)
                    self.history["g_loss"].append(avg_g)
                    print(
                        f"[epoch {epoch}/{num_epochs} batch {batch_idx+1}/{max_steps}] "
                        f"iter {self.curr_iters}/{total_iters} "
                        f"D: {avg_d:.4f} G: {avg_g:.4f}",
                        flush=True,
                    )
                    running = {"d": 0.0, "g": 0.0, "n": 0}

                if sample_step and self.curr_iters % sample_step == 0:
                    self.sample_grid(loader, self.curr_iters)

                if save_step and self.curr_iters % save_step == 0:
                    self.save(self.curr_iters)
                    self._prune_old_checkpoints(keep=keep)

            if aborted:
                break

            self.sample_grid(loader, self.curr_iters)
            self.save(self.curr_iters)
            self._prune_old_checkpoints(keep=keep)
            epoch_d_avg = epoch_d_sum / max(epoch_d_n, 1)
            self.save_best(self.curr_iters, epoch_d_avg)
            print(
                f"Epoch {epoch}/{num_epochs} finished, checkpoint @ iter {self.curr_iters}, "
                f"epoch D avg: {epoch_d_avg:.2f}",
                flush=True,
            )

        return self.history
