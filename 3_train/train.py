# 3_train/train.py
import os
import sys
import time
import math
import json
import yaml
import argparse
import random
import shutil
import logging
import pathlib
from typing import Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[1] #根目录/home/chenjingwen/Projects/lixinji/code
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / "_2_models"))

from _2_models.dann_baseline import build_dann  # type: ignore
from _2_models.pdan_selective import build_pdan_selective  # type: ignore
from _2_models.utils_loss import build_optimizer  # type: ignore


class NpyDataset(Dataset):
    def __init__(self, x_path: str, y_path: str, normalize: bool = True):
        self.X = np.load(x_path)
        self.y = np.load(y_path)
        if normalize:
            mu = self.X.mean(axis=1, keepdims=True)
            std = self.X.std(axis=1, keepdims=True) + 1e-6
            self.X = (self.X - mu) / std
        self.X = self.X.astype(np.float32)
        self.y = self.y.astype(np.int64)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = self.X[idx]
        x = np.expand_dims(x, 0)
        y = self.y[idx]
        return torch.from_numpy(x), torch.tensor(y)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dirs(paths):
    for p in paths:
        os.makedirs(p, exist_ok=True)


def load_config(cfg_path: str) -> dict:
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_logger(log_dir: str, name: str = "train") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    for h in list(logger.handlers):
        logger.removeHandler(h)
    fmt = logging.Formatter(fmt="%(asctime)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S")
    fh = logging.FileHandler(os.path.join(log_dir, "train.log"), encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def make_dataloaders(src_name: str, tgt_name: str, processed_dir: str, batch_size: int, num_workers: int = 0) -> Tuple[DataLoader, DataLoader, NpyDataset, NpyDataset]:
    sx = os.path.join(processed_dir, f"{src_name}_X.npy")
    sy = os.path.join(processed_dir, f"{src_name}_y.npy")
    tx = os.path.join(processed_dir, f"{tgt_name}_X.npy")
    ty = os.path.join(processed_dir, f"{tgt_name}_y.npy")
    if not (os.path.isfile(sx) and os.path.isfile(sy) and os.path.isfile(tx) and os.path.isfile(ty)):
        raise FileNotFoundError("预处理文件缺失，请先运行 1_data/data_prep.py")
    ds_src = NpyDataset(sx, sy, normalize=True)
    ds_tgt = NpyDataset(tx, ty, normalize=True)
    dl_src = DataLoader(ds_src, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers)
    dl_tgt = DataLoader(ds_tgt, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers)
    return dl_src, dl_tgt, ds_src, ds_tgt


def cycle(dl):
    while True:
        for b in dl:
            yield b


def train_one_epoch(model: nn.Module, dl_src: DataLoader, dl_tgt: DataLoader, device: torch.device, optimizer: torch.optim.Optimizer, epoch: int, total_epochs: int, logger: logging.Logger, model_name: str, cls_weight: Optional[torch.Tensor] = None) -> Dict[str, float]:
    model.train()
    it_src = cycle(dl_src)
    it_tgt = cycle(dl_tgt)
    steps = min(len(dl_src), len(dl_tgt))
    loss_cls_meter = 0.0
    loss_dom_meter = 0.0
    acc_src_meter = 0.0
    for step in range(steps):
        src_x, src_y = next(it_src)
        tgt_x, _ = next(it_tgt)
        src_x = src_x.to(device)
        src_y = src_y.to(device)
        tgt_x = tgt_x.to(device)
        p = (epoch * steps + step) / max(1, (total_epochs * steps))
        lambd = 2. / (1. + math.exp(-10 * p)) - 1.
        optimizer.zero_grad()
        if model_name == "dann":
            out = model(src_x=src_x, tgt_x=tgt_x, src_y=src_y, grl_lambda=lambd)
        else:
            out = model(src_x=src_x, tgt_x=tgt_x, src_y=src_y, grl_lambda=lambd, adapt=True)
        loss_cls = out.get("cls_loss", torch.tensor(0., device=device))
        loss_dom = out.get("domain_loss", torch.tensor(0., device=device))
        loss = loss_cls + loss_dom
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            logits = out.get("src_logits")
            if logits is not None:
                pred = logits.argmax(-1)
                acc_src = (pred == src_y).float().mean().item()
            else:
                acc_src = 0.0
        loss_cls_meter += float(loss_cls.item())
        loss_dom_meter += float(loss_dom.item())
        acc_src_meter += acc_src
        if (step + 1) % 10 == 0:
            logger.info(f"epoch {epoch} step {step+1}/{steps} cls={loss_cls.item():.4f} dom={loss_dom.item():.4f} src_acc={acc_src:.4f} λ={lambd:.3f}")
    loss_cls_meter /= steps
    loss_dom_meter /= steps
    acc_src_meter /= steps
    return {"cls": loss_cls_meter, "dom": loss_dom_meter, "src_acc": acc_src_meter}


@torch.no_grad()
def evaluate(model: nn.Module, ds: NpyDataset, device: torch.device, batch_size: int = 256) -> Dict[str, object]:
    model.eval()
    xs = torch.from_numpy(ds.X).unsqueeze(1)
    ys = torch.from_numpy(ds.y)
    preds = []
    probs = []
    for i in range(0, xs.size(0), batch_size):
        xb = xs[i:i+batch_size].to(device)
        logits = model.infer(xb)
        pb = F.softmax(logits, -1).cpu().numpy()
        preds.append(pb.argmax(-1))
        probs.append(pb)
    y_pred = np.concatenate(preds, 0)
    y_true = ys.numpy()
    prob = np.concatenate(probs, 0)
    acc = float(accuracy_score(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred)
    rep = classification_report(y_true, y_pred, output_dict=True, digits=4)
    return {"acc": acc, "cm": cm, "report": rep, "probs": prob, "y_true": y_true, "y_pred": y_pred}


def plot_curves(hist: Dict[str, list], out_png: str, title: str):
    plt.figure(figsize=(7, 5))
    x = list(range(1, len(hist["cls"]) + 1))
    plt.plot(x, hist["cls"], label="cls")
    plt.plot(x, hist["dom"], label="domain")
    plt.plot(x, hist["src_acc"], label="src_acc")
    plt.xlabel("epoch")
    plt.ylabel("value")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def plot_confusion(cm: np.ndarray, labels: Tuple[str, ...], out_png: str, title: str):
    plt.figure(figsize=(5, 4))
    im = plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45, ha="right")
    plt.yticks(tick_marks, labels)
    thresh = cm.max() / 2.0 if cm.size else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], "d"), ha="center", va="center", color="white" if cm[i, j] > thresh else "black")
    plt.ylabel("True")
    plt.xlabel("Pred")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def plot_tsne(model: nn.Module, ds: NpyDataset, device: torch.device, out_png: str, title: str):
    model.eval()
    xs = torch.from_numpy(ds.X).unsqueeze(1).to(device)
    feats = []
    with torch.no_grad():
        for i in range(0, xs.size(0), 256):
            xb = xs[i:i+256]
            f = model.feature_extractor(xb)
            feats.append(f.cpu().numpy())
    feats = np.concatenate(feats, 0)
    y = ds.y
    tsne = TSNE(n_components=2, init="random", learning_rate="auto", perplexity=30, n_iter=1000)
    emb = tsne.fit_transform(feats)
    plt.figure(figsize=(6, 5))
    for cls in np.unique(y):
        idx = y == cls
        plt.scatter(emb[idx, 0], emb[idx, 1], s=6, label=str(cls))
    plt.legend()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def save_report(rep: Dict[str, object], out_dir: str, prefix: str):
    js = {
        "acc": rep["acc"],
        "cm": rep["cm"].tolist() if isinstance(rep["cm"], np.ndarray) else rep["cm"],
        "report": rep["report"],
    }
    with open(os.path.join(out_dir, f"{prefix}_report.json"), "w", encoding="utf-8") as f:
        json.dump(js, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(ROOT / "3_train" / "config_train.yaml"))
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--model", type=str, choices=["dann", "pdan"], default="pdan")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    cfg = load_config(args.config) #字典
    task_cfg = None
    for t in cfg["tasks"]:
        if t["name"] == args.task:
            task_cfg = t #task_cfg也是字典 内容举例'name': 'exp_to_2170','source': 'experiment','target': 'centrifuge_2170'
            break
    if task_cfg is None:
        raise ValueError(f"找不到任务: {args.task}")

    exp_name = f'{args.model}_{task_cfg["source"]}_to_{task_cfg["target"]}' #起名dann_experiment_to_centrifuge_2170
    out_root = ROOT / "4_results"
    out_logs = out_root / "logs" / exp_name
    out_weights = out_root / "weights" / exp_name
    out_plots = out_root / "plots" / exp_name
    ensure_dirs([out_logs, out_weights, out_plots])

    logger = build_logger(str(out_logs), name=exp_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    epochs = args.epochs if args.epochs is not None else int(cfg["train"]["epochs"])
    batch_size = args.batch_size if args.batch_size is not None else int(cfg["train"]["batch_size"])
    lr = args.lr if args.lr is not None else float(cfg["train"]["lr"])
    in_channels = int(cfg["data"]["in_channels"]) #1
    num_classes = int(cfg["data"]["num_classes"]) #2
    processed_dir = str(ROOT / "1_data" / "processed")

    dl_src, dl_tgt, ds_src, ds_tgt = make_dataloaders(task_cfg["source"], task_cfg["target"], processed_dir, batch_size, num_workers=int(cfg["train"].get("num_workers", 0)))

    if args.model == "dann":
        model = build_dann(in_channels=in_channels, num_classes=num_classes, feat_dim=int(cfg["model"]["feat_dim"]))
    else:
        model = build_pdan_selective(in_channels=in_channels, num_classes=num_classes, feat_dim=int(cfg["model"]["feat_dim"]))
    model = model.to(device)
    optimizer = build_optimizer(model, lr=lr, wd=float(cfg["train"]["weight_decay"]))
    scaler = torch.cuda.amp.GradScaler(enabled=cfg["train"].get("amp", True) and torch.cuda.is_available()) #！！对梯度进行处理，没用coeff，也不对，要改

    history = {"cls": [], "dom": [], "src_acc": []}
    best_acc = -1.0
    best_path = None

    for epoch in range(1, epochs + 1):
        stats = train_one_epoch(model, dl_src, dl_tgt, device, optimizer, epoch, epochs, logger, args.model)
        history["cls"].append(stats["cls"])
        history["dom"].append(stats["dom"])
        history["src_acc"].append(stats["src_acc"])

        rep_src = evaluate(model, ds_src, device)
        rep_tgt = evaluate(model, ds_tgt, device)
        logger.info(f"epoch {epoch} src_acc={rep_src['acc']:.4f} tgt_acc={rep_tgt['acc']:.4f}")

        plot_curves(history, str(out_plots / "accuracy_curve.png"), title=f"{exp_name} curves")
        plot_confusion(rep_tgt["cm"], tuple(str(i) for i in range(num_classes)), str(out_plots / "confusion_matrix.png"), title=f"{exp_name} tgt")
        save_report(rep_tgt, str(out_logs), prefix="target")
        save_report(rep_src, str(out_logs), prefix="source")

        if epoch == epochs or (epoch % max(1, epochs // 3) == 0):
            try:
                plot_tsne(model, ds_tgt, device, str(out_plots / f"tsne_epoch_{epoch}.png"), title=f"{exp_name} tgt tsne")
            except Exception:
                pass

        ckpt_path = out_weights / f"epoch_{epoch:03d}.pt"
        torch.save({"model": model.state_dict(), "epoch": epoch, "config": cfg, "task": task_cfg}, ckpt_path)
        if rep_tgt["acc"] > best_acc:
            best_acc = rep_tgt["acc"]
            best_path = out_weights / "best.pt"
            shutil.copy(str(ckpt_path), str(best_path))

    logger.info(f"best target acc: {best_acc:.4f}")
    if best_path is not None:
        logger.info(f"best checkpoint: {best_path}")


if __name__ == "__main__":
    main()
