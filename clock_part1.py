#!/usr/bin/env python
"""
Task 2, Part 1 (Person B): four formulations of the "tell-the-time" problem.

    1. classification (12 / 24 / 60 / 720 classes)
    2. regression     (MSE / L1 / SmoothL1 on total minutes)
    3. multi-head     (hour head + minute head, L = L_hour + alpha * L_minute)
    4. trig encoding  (sin/cos of the two hand angles)

All approaches use the same CNN backbone, the same seeded 80/10/10 split and the same
metric: the "common-sense error" (12-hour circular error in minutes). Only TRAIN and VAL
are ever loaded here; the test set is never touched.

Run on a GPU machine
    python clock_part1.py --images images150.npy --labels labels.npy --epochs 40
    python clock_part1.py --mode sanity                      # data checks only (saves PNGs, prints md5)
    python clock_part1.py --mode overfit                     # can the model memorise 512 images?
    python clock_part1.py --approaches trig multihead --epochs 40
    nohup python clock_part1.py --epochs 40 > run.log 2>&1 &   # survives a closed SSH session

Convert to a notebook later (the "# %%" markers are cell boundaries):
    pip install jupytext && jupytext --to ipynb clock_part1.py
In the notebook, skip parse_args() and write:  cfg = Config(images="...", epochs=40)
then call the functions in main() one by one. Also delete the matplotlib.use("Agg") line there.
"""
# %% ----------------------------------------------------------------------- imports
import argparse
import copy
import hashlib
import json
import math
import os
import random
import re
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import matplotlib
matplotlib.use("Agg")  # headless: figures are saved to files, never shown
import matplotlib.pyplot as plt


# %% ----------------------------------------------------------------------- configuration
@dataclass
class Config:
    # data
    images: str = "images.npy"
    labels: str = "labels.npy"
    split_path: str = "clock_split.npz"     # saved split indices (shared with the team)
    out_dir: str = "runs/part1"
    seed: int = 42
    max_train: int = 0                      # 0 = use all training images (set e.g. 2000 for a fast smoke test)
    # training (same budget for every approach, so the comparison is fair)
    epochs: int = 15
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-4
    pool: int = 4                           # spatial size kept before the linear layers (4 = 4x4 map)
    cudnn_benchmark: bool = False           # faster on GPU, slightly less reproducible
    # what to run
    mode: str = "run"                       # run | sanity | overfit
    approaches: List[str] = field(default_factory=lambda: ["classification", "regression", "multihead", "trig"])
    n_classes: List[int] = field(default_factory=lambda: [12, 24, 60, 720])
    reg_losses: List[str] = field(default_factory=lambda: ["mse", "l1", "smoothl1"])
    alphas: List[float] = field(default_factory=lambda: [0.5, 1.0, 2.0])
    minute_modes: List[str] = field(default_factory=lambda: ["cls", "reg"])
    trig_norm: List[int] = field(default_factory=lambda: [0, 1])
    # overfit test
    overfit_n: int = 512
    overfit_epochs: int = 60


def parse_args(argv=None) -> Config:
    d = Config()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--images", default=d.images)
    p.add_argument("--labels", default=d.labels)
    p.add_argument("--split_path", default=d.split_path)
    p.add_argument("--out_dir", default=d.out_dir)
    p.add_argument("--seed", type=int, default=d.seed)
    p.add_argument("--max_train", type=int, default=d.max_train)
    p.add_argument("--epochs", type=int, default=d.epochs)
    p.add_argument("--batch_size", type=int, default=d.batch_size)
    p.add_argument("--lr", type=float, default=d.lr)
    p.add_argument("--weight_decay", type=float, default=d.weight_decay)
    p.add_argument("--pool", type=int, default=d.pool)
    p.add_argument("--cudnn_benchmark", action="store_true")
    p.add_argument("--mode", choices=["run", "sanity", "overfit"], default=d.mode)
    p.add_argument("--approaches", nargs="+", default=d.approaches,
                   choices=["classification", "regression", "multihead", "trig"])
    p.add_argument("--n_classes", nargs="+", type=int, default=d.n_classes)
    p.add_argument("--reg_losses", nargs="+", default=d.reg_losses, choices=["mse", "l1", "smoothl1"])
    p.add_argument("--alphas", nargs="+", type=float, default=d.alphas)
    p.add_argument("--minute_modes", nargs="+", default=d.minute_modes, choices=["cls", "reg"])
    p.add_argument("--trig_norm", nargs="+", type=int, default=d.trig_norm, help="0 = raw linear, 1 = unit-normalised pairs")
    p.add_argument("--overfit_n", type=int, default=d.overfit_n)
    p.add_argument("--overfit_epochs", type=int, default=d.overfit_epochs)
    return Config(**vars(p.parse_args(argv)))


# %% ----------------------------------------------------------------------- basics
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_")


# %% ----------------------------------------------------------------------- common-sense metric
CLOCK_MINUTES = 720  # 12 h * 60


def to_total_minutes(hour, minute):
    """(hour, minute) -> minutes on a 12-hour clock, 0..719. 12 and 0 are the same hour."""
    return (hour % 12) * 60 + minute


def common_sense_error(pred, true):
    """Element-wise circular error in minutes (0..360). Works for numpy and torch."""
    lib = torch if isinstance(pred, torch.Tensor) or isinstance(true, torch.Tensor) else np
    d = lib.abs(pred - true) % CLOCK_MINUTES
    return lib.minimum(d, CLOCK_MINUTES - d)


def summarize_errors(errors) -> Dict[str, float]:
    e = np.asarray(errors, dtype=np.float64)
    return {"mean_err": float(e.mean()), "median_err": float(np.median(e)),
            "within_1": float((e <= 1).mean()), "within_5": float((e <= 5).mean()),
            "within_15": float((e <= 15).mean())}


def selftest_metric() -> None:
    t = to_total_minutes
    assert common_sense_error(t(11, 55), t(0, 5)) == 10
    assert common_sense_error(t(0, 5), t(11, 55)) == 10
    assert common_sense_error(t(3, 30), t(3, 30)) == 0
    assert common_sense_error(t(12, 0), t(0, 0)) == 0
    assert common_sense_error(t(6, 0), t(0, 0)) == 360
    assert common_sense_error(t(2, 59), t(3, 1)) == 2
    out = common_sense_error(torch.tensor([715., 5., 725., -10.]), torch.tensor([5., 715., 5., 710.]))
    assert torch.allclose(out, torch.tensor([10., 10., 0., 0.])), out
    print("metric self-test passed")


# %% ----------------------------------------------------------------------- data
@dataclass
class Data:
    images: np.ndarray
    labels: np.ndarray            # [N, 2] = (hour 0..11, minute 0..59)
    train_idx: np.ndarray
    val_idx: np.ndarray
    mean: float
    std: float
    scale: float
    loaders: Optional[Dict[str, DataLoader]] = None


def md5_of(path: str, chunk: int = 1 << 24) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def describe_files(cfg: Config) -> None:
    """Print size + md5 of the data files, so you can tell whether two machines hold the same copy."""
    for path in (cfg.images, cfg.labels):
        print(f"{path}: {os.path.getsize(path) / 1e6:.1f} MB | md5 {md5_of(path)}")


def load_arrays(cfg: Config):
    images = np.load(cfg.images)
    labels = np.load(cfg.labels).astype(np.int64)
    print(f"images {images.shape} {images.dtype} | labels {labels.shape}")
    assert len(images) == len(labels) and labels.ndim == 2 and labels.shape[1] == 2, "images/labels mismatch"
    print(f"hour range {labels[:, 0].min()}..{labels[:, 0].max()} | minute range {labels[:, 1].min()}..{labels[:, 1].max()}")
    assert labels[:, 1].min() >= 0 and labels[:, 1].max() <= 59
    labels[:, 0] %= 12                      # 12 -> 0, hours are always 0..11
    return images, labels


def make_split(n: int, cfg: Config):
    """Seeded shuffle -> 80/10/10. The indices are saved once and reused, so everyone gets the same split."""
    if os.path.exists(cfg.split_path):
        sp = np.load(cfg.split_path)
        tr, va, te = sp["train_idx"], sp["val_idx"], sp["test_idx"]
        assert len(tr) + len(va) + len(te) == n, "split file does not match this dataset (delete it to recreate)"
        print("loaded existing split from", cfg.split_path)
    else:
        perm = np.random.default_rng(cfg.seed).permutation(n)
        n_tr, n_va = int(0.8 * n), int(0.1 * n)
        tr, va, te = perm[:n_tr], perm[n_tr:n_tr + n_va], perm[n_tr + n_va:]
        np.savez(cfg.split_path, train_idx=tr, val_idx=va, test_idx=te, seed=cfg.seed)
        print("saved new split to", cfg.split_path)
    assert not (set(tr) & set(va)) and not (set(tr) & set(te)) and not (set(va) & set(te)), "splits overlap"
    print(f"train {len(tr)} | val {len(va)} | test {len(te)} (test is never loaded here)")
    print("split fingerprint (compare with teammates):", tr[:5].tolist(), va[:5].tolist(), te[:5].tolist())
    return tr, va, te


def compute_train_stats(images: np.ndarray, train_idx: np.ndarray, scale: float, chunk: int = 1000):
    """Pixel mean/std from the TRAINING images only, computed in chunks (low memory)."""
    idx = np.sort(train_idx)
    s = ss = 0.0
    n = 0
    for i in range(0, len(idx), chunk):
        x = images[idx[i:i + chunk]].astype(np.float64) / scale
        s += x.sum()
        ss += (x ** 2).sum()
        n += x.size
    mean = s / n
    return float(mean), float(math.sqrt(ss / n - mean ** 2))


def make_loader(data: Data, idx: np.ndarray, shuffle: bool, cfg: Config) -> DataLoader:
    x = torch.from_numpy(np.ascontiguousarray(data.images[idx])).unsqueeze(1)   # raw [N,1,H,W]
    y = torch.from_numpy(data.labels[idx])                                      # [N,2]
    g = torch.Generator().manual_seed(cfg.seed)
    return DataLoader(TensorDataset(x, y), batch_size=cfg.batch_size, shuffle=shuffle,
                      generator=g if shuffle else None, pin_memory=torch.cuda.is_available())


def prepare_data(cfg: Config) -> Data:
    images, labels = load_arrays(cfg)
    tr, va, _ = make_split(len(images), cfg)
    if cfg.max_train:
        tr = tr[:cfg.max_train]             # the split is already shuffled, so this is a random subset
    scale = 255.0 if images[:100].max() > 1.5 else 1.0
    mean, std = compute_train_stats(images, tr, scale)
    print(f"train-only normalisation: mean={mean:.4f} std={std:.4f} (pixels divided by {scale})")
    data = Data(images, labels, tr, va, mean, std, scale)
    data.loaders = {"train": make_loader(data, tr, True, cfg), "val": make_loader(data, va, False, cfg)}
    return data


def prep(x: torch.Tensor, data: Data, device: torch.device) -> torch.Tensor:
    """Raw uint8 batch -> normalised float tensor (train-set stats, no augmentation, never flips)."""
    return ((x.to(device, non_blocking=True).float() / data.scale) - data.mean) / data.std


# %% ----------------------------------------------------------------------- data sanity checks (saved as PNGs)
def save_sample_grid(data: Data, cfg: Config) -> None:
    fig, axes = plt.subplots(1, 6, figsize=(15, 3))
    for ax, i in zip(axes, data.train_idx[:6]):
        ax.imshow(data.images[i], cmap="gray"); ax.axis("off")
        ax.set_title(f"{data.labels[i, 0]:02d}:{data.labels[i, 1]:02d}")
    fig.savefig(os.path.join(cfg.out_dir, "samples.png"), dpi=100, bbox_inches="tight")
    plt.close(fig)


def save_alignment_grid(data: Data, cfg: Config) -> None:
    """Images and labels are aligned if, whatever the clock rotation, 00:00 shows overlapping hands,
    06:00 hands pointing opposite ways (a straight line) and 03:00 hands at a right angle."""
    fig, axes = plt.subplots(3, 4, figsize=(12, 9))
    for r, (h, m) in enumerate([(0, 0), (6, 0), (3, 0)]):
        ids = np.where((data.labels[:, 0] == h) & (data.labels[:, 1] == m))[0]
        for c in range(4):
            axes[r, c].axis("off")
            if c < len(ids):
                axes[r, c].imshow(data.images[ids[c]], cmap="gray")
                axes[r, c].set_title(f"idx {ids[c]}: {h:02d}:{m:02d}")
    fig.savefig(os.path.join(cfg.out_dir, "alignment_check.png"), dpi=100, bbox_inches="tight")
    plt.close(fig)


# %% ----------------------------------------------------------------------- models
class ConvBackbone(nn.Module):
    """Shared conv feature extractor. Adaptive pooling makes it work for 75x75 and 150x150 inputs."""

    def __init__(self, channels=(32, 64, 128, 128), feat_dim=256, dropout=0.3, pool=4):
        super().__init__()
        layers, in_c = [], 1
        for c in channels:
            layers += [nn.Conv2d(in_c, c, 3, padding=1), nn.BatchNorm2d(c), nn.ReLU(), nn.MaxPool2d(2)]
            in_c = c
        layers += [nn.AdaptiveAvgPool2d(pool), nn.Flatten(), nn.Linear(in_c * pool * pool, feat_dim),
                   nn.ReLU(), nn.Dropout(dropout)]
        self.net = nn.Sequential(*layers)
        self.out_dim = feat_dim

    def forward(self, x):
        return self.net(x)


class ClockNet(nn.Module):
    """Backbone + one linear output layer (no activation on the output)."""

    def __init__(self, out_dim: int, pool: int = 4):
        super().__init__()
        self.backbone = ConvBackbone(pool=pool)
        self.head = nn.Linear(self.backbone.out_dim, out_dim)

    def forward(self, x):
        return self.head(self.backbone(x))


class MultiHeadNet(nn.Module):
    def __init__(self, minute_mode: str = "cls", pool: int = 4):
        super().__init__()
        self.backbone = ConvBackbone(pool=pool)
        self.hour_head = nn.Linear(self.backbone.out_dim, 12)
        self.minute_head = nn.Linear(self.backbone.out_dim, 60 if minute_mode == "cls" else 1)

    def forward(self, x):
        f = self.backbone(x)
        return self.hour_head(f), self.minute_head(f)


TWO_PI = 2 * math.pi


class TrigNet(nn.Module):
    def __init__(self, normalize: bool = False, pool: int = 4):
        super().__init__()
        self.normalize = normalize
        self.backbone = ConvBackbone(pool=pool)
        self.head = nn.Linear(self.backbone.out_dim, 4)          # sin_h, cos_h, sin_m, cos_m (linear, no ReLU)

    def forward(self, x):
        out = self.head(self.backbone(x))
        if self.normalize:                                       # project each (sin, cos) pair onto the unit circle
            out = F.normalize(out.view(-1, 2, 2), dim=2).view(-1, 4)
        return out


def trig_targets(y: torch.Tensor) -> torch.Tensor:
    hour, minute = y[:, 0].float(), y[:, 1].float()
    th = TWO_PI * (hour + minute / 60.0) / 12.0
    tm = TWO_PI * minute / 60.0
    return torch.stack([torch.sin(th), torch.cos(th), torch.sin(tm), torch.cos(tm)], dim=1)


def decode_trig(out: torch.Tensor) -> torch.Tensor:
    th = torch.atan2(out[:, 0], out[:, 1]) % TWO_PI
    tm = torch.atan2(out[:, 2], out[:, 3]) % TWO_PI
    coarse = th / TWO_PI * 720.0                                 # total minutes from the hour hand
    minute = tm / TWO_PI * 60.0                                  # exact minute from the minute hand
    hour = torch.round((coarse - minute) / 60.0) % 12            # hour consistent with both hands
    return hour * 60.0 + minute


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# %% ----------------------------------------------------------------------- training / evaluation
@torch.no_grad()
def evaluate(model, loader, data, device, loss_fn, decode_fn):
    model.eval()
    errs, total_loss, n = [], 0.0, 0
    for x, y in loader:
        x, y = prep(x, data, device), y.to(device)
        out = model(x)
        total_loss += loss_fn(out, y).item() * len(y)
        n += len(y)
        errs.append(common_sense_error(decode_fn(out).float(), to_total_minutes(y[:, 0], y[:, 1]).float()))
    summary = summarize_errors(torch.cat(errs).cpu().numpy())
    summary["loss"] = total_loss / n
    return summary


def fit(model, loaders, data, device, loss_fn, decode_fn, epochs, lr, weight_decay,
        ckpt_path: Optional[str] = None, verbose: bool = True):
    """AdamW + cosine schedule. Keeps the weights of the best VALIDATION epoch (lowest mean error)."""
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    history, best, best_state = [], None, None
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        run, n = 0.0, 0
        for x, y in loaders["train"]:
            x, y = prep(x, data, device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            run += loss.item() * len(y)
            n += len(y)
        sched.step()
        val = evaluate(model, loaders["val"], data, device, loss_fn, decode_fn)
        val.update(epoch=epoch, train_loss=run / n)
        history.append(val)
        if verbose:
            print(f"  epoch {epoch:3d} | train loss {val['train_loss']:.4f} | val loss {val['loss']:.4f} | "
                  f"val err mean {val['mean_err']:6.1f} median {val['median_err']:6.1f} min | {time.time() - t0:.1f}s", flush=True)
        if best is None or val["mean_err"] < best["mean_err"]:
            best, best_state = dict(val), copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    if ckpt_path:
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        torch.save(best_state, ckpt_path)
    return model, history, best


# %% ----------------------------------------------------------------------- results log
class ResultLog:
    """Collects one row per trained model (the comparison table) and the per-epoch histories."""

    def __init__(self, out_dir: str):
        self.out_dir = out_dir
        self.rows: List[dict] = []
        self.histories: Dict[str, list] = {}

    def add(self, approach, variant, loss_name, epochs, model, best, history):
        self.rows.append({"approach": approach, "variant": variant, "loss": loss_name, "epochs": epochs,
                          "params": count_params(model), "best_epoch": best["epoch"],
                          "val_mean_err": round(best["mean_err"], 3), "val_median_err": round(best["median_err"], 3),
                          "val_within_1": round(best["within_1"], 4), "val_within_5": round(best["within_5"], 4),
                          "val_within_15": round(best["within_15"], 4)})
        self.histories[f"{approach} | {variant}"] = history
        pd.DataFrame(self.rows).to_csv(os.path.join(self.out_dir, "results_val.csv"), index=False)
        with open(os.path.join(self.out_dir, "histories.json"), "w") as f:
            json.dump(self.histories, f)
        print(f"[logged] {approach} | {variant} | val mean err {best['mean_err']:.2f} min", flush=True)
        if best["mean_err"] > 165:
            print("  WARNING: at chance level (random guessing = ~180 min). This run did not learn.", flush=True)


def train_one(cfg, data, device, log, approach, variant, loss_name, build_model, loss_fn, decode_fn):
    """Seed -> build model -> train -> log. Seeding before building gives every run the same init."""
    print(f"\n=== {approach} | {variant} | {loss_name} ===", flush=True)
    set_seed(cfg.seed)
    model = build_model()
    ckpt = os.path.join(cfg.out_dir, "checkpoints", f"{slug(approach)}__{slug(variant)}__{slug(loss_name)}.pt")
    model, hist, best = fit(model, data.loaders, data, device, loss_fn, decode_fn,
                            cfg.epochs, cfg.lr, cfg.weight_decay, ckpt_path=ckpt)
    log.add(approach, variant, loss_name, cfg.epochs, model, best, hist)


# %% ----------------------------------------------------------------------- approach 1: classification
def run_classification(cfg, data, device, log):
    for n_cls in cfg.n_classes:
        assert 720 % n_cls == 0, "n_classes must divide 720"
        width = 720 // n_cls

        def loss_fn(out, y, width=width):
            target = torch.div(to_total_minutes(y[:, 0], y[:, 1]), width, rounding_mode="floor")
            return F.cross_entropy(out, target)      # no softmax before this: the loss applies it

        def decode_fn(out, width=width):
            return out.argmax(dim=1).float() * width + (width - 1) / 2     # centre of the predicted bin

        train_one(cfg, data, device, log, "classification", f"{n_cls} classes", "CrossEntropy",
                  lambda n_cls=n_cls: ClockNet(n_cls, cfg.pool), loss_fn, decode_fn)


# %% ----------------------------------------------------------------------- approach 2: regression
def run_regression(cfg, data, device, log):
    criteria = {"mse": nn.MSELoss(), "l1": nn.L1Loss(), "smoothl1": nn.SmoothL1Loss()}
    for name in cfg.reg_losses:
        crit = criteria[name]

        def loss_fn(out, y, crit=crit):
            return crit(out.squeeze(1), to_total_minutes(y[:, 0], y[:, 1]).float() / 720.0)

        def decode_fn(out):
            return out.squeeze(1) * 720.0            # the metric applies the mod-720 wrap-around itself

        train_one(cfg, data, device, log, "regression", "total-minutes scalar", name,
                  lambda: ClockNet(1, cfg.pool), loss_fn, decode_fn)


# %% ----------------------------------------------------------------------- approach 3: multi-head
def run_multihead(cfg, data, device, log):
    for mode in cfg.minute_modes:
        for alpha in cfg.alphas:
            def loss_fn(out, y, mode=mode, alpha=alpha):
                out_h, out_m = out
                l_hour = F.cross_entropy(out_h, y[:, 0])
                if mode == "cls":
                    l_min = F.cross_entropy(out_m, y[:, 1])
                else:
                    l_min = F.smooth_l1_loss(out_m.squeeze(1), y[:, 1].float() / 60.0)
                return l_hour + alpha * l_min

            def decode_fn(out, mode=mode):
                out_h, out_m = out
                hour = out_h.argmax(dim=1).float()
                minute = out_m.argmax(dim=1).float() if mode == "cls" else (out_m.squeeze(1) * 60.0).clamp(0, 59)
                return hour * 60.0 + minute

            train_one(cfg, data, device, log, "multi-head", f"hour cls + minute {mode}", f"CE + {alpha}*minute",
                      lambda mode=mode: MultiHeadNet(mode, cfg.pool), loss_fn, decode_fn)


# %% ----------------------------------------------------------------------- approach 4: trigonometric encoding
def selftest_trig() -> None:
    """Encode -> decode round trip over all 720 times must be exact."""
    hh, mm = torch.arange(12).repeat_interleave(60), torch.arange(60).repeat(12)
    rt = common_sense_error(decode_trig(trig_targets(torch.stack([hh, mm], 1))), to_total_minutes(hh, mm).float())
    assert rt.max() < 1e-3, rt.max()
    print("trig round-trip self-test passed")


def run_trig(cfg, data, device, log):
    def loss_fn(out, y):
        return F.mse_loss(out, trig_targets(y))

    for norm in cfg.trig_norm:
        train_one(cfg, data, device, log, "trig encoding", "unit-normalised" if norm else "raw linear",
                  "MSE on (sin,cos)x2", lambda norm=norm: TrigNet(bool(norm), cfg.pool), loss_fn, decode_trig)


# %% ----------------------------------------------------------------------- overfit test
def overfit_test(cfg, data, device):
    """Can the model memorise a tiny training subset? It evaluates on the SAME images. If the error does not
    fall to a few minutes, the problem is in the data/labels/model/optimiser, not in 'too few epochs'.
    (Memorising is possible even with wrong labels, so this does not test label alignment: use alignment_check.png.)"""
    idx = data.train_idx[:cfg.overfit_n]
    loaders = {"train": make_loader(data, idx, True, cfg), "val": make_loader(data, idx, False, cfg)}
    set_seed(cfg.seed)
    model = TrigNet(normalize=True, pool=cfg.pool)
    print(f"\n=== overfit test: {len(idx)} images, {cfg.overfit_epochs} epochs ===")
    model, hist, best = fit(model, loaders, data, device, lambda o, y: F.mse_loss(o, trig_targets(y)), decode_trig,
                            cfg.overfit_epochs, cfg.lr, 0.0)
    print(f"\nbest error on the memorised subset: mean {best['mean_err']:.1f} min, median {best['median_err']:.1f} min")
    print("-> if this is still above ~20 min, something is wrong beyond the epoch count." if best["mean_err"] > 20
          else "-> the pipeline can learn; poor full-data results mean undertraining or an architecture limit.")


# %% ----------------------------------------------------------------------- report
def plot_best_curves(log: ResultLog, cfg: Config) -> None:
    df = pd.DataFrame(log.rows)
    best = df.loc[df.groupby("approach")["val_mean_err"].idxmin()]
    plt.figure(figsize=(8, 5))
    for _, r in best.iterrows():
        h = log.histories[f"{r['approach']} | {r['variant']}"]
        plt.plot([e["epoch"] for e in h], [e["mean_err"] for e in h], marker="o", label=f"{r['approach']} ({r['variant']})")
    plt.axhline(180, color="gray", ls="--", label="chance (~180 min)")
    plt.xlabel("epoch"); plt.ylabel("validation mean common-sense error (min)")
    plt.yscale("log"); plt.grid(alpha=0.3); plt.legend(); plt.title("Validation error per approach")
    plt.savefig(os.path.join(cfg.out_dir, "val_curves.png"), dpi=120, bbox_inches="tight")
    plt.close()


def final_report(log: ResultLog, cfg: Config) -> None:
    if not log.rows:
        return
    df = pd.DataFrame(log.rows).sort_values("val_mean_err").reset_index(drop=True)
    print("\n" + "=" * 100 + "\nCOMPARISON (validation set only; all val_* columns are common-sense error in minutes)")
    print("chance level: mean/median ~180 min, within-1 ~0.4%, within-5 ~1.5%, within-15 ~4.3%\n")
    print(df.to_string(index=False))
    print("\nBest variant per approach:")
    print(df.loc[df.groupby("approach")["val_mean_err"].idxmin()].to_string(index=False))
    print("\nLearning check (train loss / val loss, first -> last epoch):")
    for k, h in log.histories.items():
        print(f"  {k:60s} train {h[0]['train_loss']:.3f} -> {h[-1]['train_loss']:.3f} | val {h[0]['loss']:.3f} -> {h[-1]['loss']:.3f}")
    plot_best_curves(log, cfg)
    print(f"\nsaved to {cfg.out_dir}: results_val.csv, histories.json, val_curves.png, checkpoints/")


# %% ----------------------------------------------------------------------- main
def main(argv=None) -> None:
    cfg = parse_args(argv)
    os.makedirs(cfg.out_dir, exist_ok=True)
    with open(os.path.join(cfg.out_dir, "config.json"), "w") as f:
        json.dump(asdict(cfg), f, indent=1)
    torch.backends.cudnn.benchmark = cfg.cudnn_benchmark
    set_seed(cfg.seed)
    device = get_device()
    print("device:", device, "|", torch.cuda.get_device_name(0) if device.type == "cuda" else "no GPU found")
    selftest_metric()
    selftest_trig()

    describe_files(cfg)
    data = prepare_data(cfg)
    save_sample_grid(data, cfg)
    save_alignment_grid(data, cfg)
    print(f"saved {cfg.out_dir}/samples.png and alignment_check.png (look at them before trusting any result)")

    if cfg.mode == "sanity":
        return
    if cfg.mode == "overfit":
        overfit_test(cfg, data, device)
        return

    log = ResultLog(cfg.out_dir)
    runners = {"classification": run_classification, "regression": run_regression,
               "multihead": run_multihead, "trig": run_trig}
    for name in cfg.approaches:
        runners[name](cfg, data, device, log)
    final_report(log, cfg)


if __name__ == "__main__":
    main()
