# 3_train/train.py
import os
import sys
import math
import json
import yaml
import argparse
import random
import shutil
import logging
import pathlib
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / "_2_models"))

from _2_models.dann_baseline import build_dann  # type: ignore
from _2_models.pdan_selective import build_pdan_selective  # type: ignore
from _2_models.utils_loss import build_optimizer  # type: ignore
from _2_models.transfer_stack.transfernet import TransferNet  # type: ignore

TRANSFER_MODELS = {"ladv", "sladv", "san", "bsan"}


class NpyDataset(Dataset):
    def __init__(self, x_path: str, y_path: str, s_path: str | None = None, normalize: bool = True):
        self.X = np.load(x_path)
        self.y = np.load(y_path)
        self.s = np.load(s_path).astype(np.float32) if s_path and os.path.isfile(s_path) else np.zeros(self.X.shape[0], dtype=np.float32)
        if normalize:
            mu = self.X.mean(axis=1, keepdims=True)
            std = self.X.std(axis=1, keepdims=True) + 1e-6
            self.X = (self.X - mu) / std
        self.X = self.X.astype(np.float32)
        self.y = self.y.astype(np.int64)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return (
            torch.from_numpy(np.expand_dims(self.X[idx], 0)),
            torch.tensor(self.y[idx]),
            torch.tensor(self.s[idx], dtype=torch.float32),
        )


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


def make_datasets(src_name: str, tgt_name: str, processed_dir: str):
    sx = os.path.join(processed_dir, f"{src_name}_X.npy")
    sy = os.path.join(processed_dir, f"{src_name}_y.npy")
    ss = os.path.join(processed_dir, f"{src_name}_s.npy")
    tx = os.path.join(processed_dir, f"{tgt_name}_X.npy")
    ty = os.path.join(processed_dir, f"{tgt_name}_y.npy")
    ts = os.path.join(processed_dir, f"{tgt_name}_s.npy")
    if not (os.path.isfile(sx) and os.path.isfile(sy) and os.path.isfile(tx) and os.path.isfile(ty)):
        raise FileNotFoundError("预处理文件缺失，请先运行 1_data/data_prep.py")
    return NpyDataset(sx, sy, ss, normalize=True), NpyDataset(tx, ty, ts, normalize=True)


def build_loader(dataset, batch_size, shuffle, num_workers, drop_last):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=drop_last)


def model_infer(model: nn.Module, xb: torch.Tensor, sb: torch.Tensor | None = None):
    if isinstance(model, TransferNet):
        return model.infer(xb, sb)
    return model.infer(xb)


def model_backbone(model: nn.Module, xb: torch.Tensor, sb: torch.Tensor | None = None):
    if isinstance(model, TransferNet):
        return model.encode(xb, sb)
    backbone = getattr(model, "feature_extractor", None)
    if backbone is None:
        backbone = getattr(model, "backbone")
    return backbone(xb)


def evaluate(model: nn.Module, ds: NpyDataset, device: torch.device, batch_size: int = 256, num_classes: int | None = None, shared_classes=None, private_classes=None) -> Dict[str, object]:
    model.eval()
    xs = torch.from_numpy(ds.X).unsqueeze(1)
    ys = torch.from_numpy(ds.y)
    ss = torch.from_numpy(ds.s)
    preds = []
    probs = []
    for i in range(0, xs.size(0), batch_size):
        xb = xs[i:i + batch_size].to(device)
        sb = ss[i:i + batch_size].to(device)
        logits = model_infer(model, xb, sb)
        pb = F.softmax(logits, -1).detach().cpu().numpy()
        preds.append(pb.argmax(-1))
        probs.append(pb)
    y_pred = np.concatenate(preds, 0)
    y_true = ys.numpy()
    prob = np.concatenate(probs, 0)

    if num_classes is None:
        num_classes = int(max(np.max(y_true), np.max(y_pred)) + 1)
    labels = list(range(num_classes))
    acc = float(accuracy_score(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    rep = classification_report(y_true, y_pred, labels=labels, output_dict=True, digits=4, zero_division=0)
    macro_f1 = float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))

    metrics = {}
    shared_classes = list(shared_classes or [])
    private_classes = list(private_classes or [])
    if shared_classes:
        shared_recalls = []
        for cls in shared_classes:
            key = str(cls)
            if key in rep:
                shared_recalls.append(float(rep[key].get("recall", 0.0)))
        if shared_recalls:
            metrics["shared_balanced_acc"] = float(np.mean(shared_recalls))
        if 1 in shared_classes and "1" in rep:
            metrics["fault_recall"] = float(rep["1"].get("recall", 0.0))
            metrics["fault_precision"] = float(rep["1"].get("precision", 0.0))
    if private_classes:
        private_pred_mask = np.isin(y_pred, np.array(private_classes))
        metrics["private_pred_rate"] = float(private_pred_mask.mean())

    return {"acc": acc, "cm": cm, "report": rep, "probs": prob, "y_true": y_true, "y_pred": y_pred, "f1": macro_f1, "metrics": metrics}


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
    ss = torch.from_numpy(ds.s).to(device)
    feats = []
    with torch.no_grad():
        for i in range(0, xs.size(0), 256):
            feats.append(model_backbone(model, xs[i:i + 256], ss[i:i + 256]).cpu().numpy())
    feats = np.concatenate(feats, 0)
    y = ds.y
    emb = TSNE(n_components=2, init="random", learning_rate="auto", perplexity=30, n_iter=1000).fit_transform(feats)
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
        "f1": rep["f1"],
        "cm": rep["cm"].tolist() if isinstance(rep["cm"], np.ndarray) else rep["cm"],
        "report": rep["report"],
        "metrics": rep.get("metrics", {}),
    }
    with open(os.path.join(out_dir, f"{prefix}_report.json"), "w", encoding="utf-8") as f:
        json.dump(js, f, indent=2, ensure_ascii=False)


def train_transfer_epoch(model, source_train, target_train, source_dataset, device, optimizer, epoch, epochs, logger, num_workers, batch_size, mu, lambda_u, rho_0):
    model.train()
    len_source_loader = len(source_train)
    len_target_loader = len(target_train)
    n_batch = min(len_source_loader, len_target_loader)
    model.epoch_based_processing(n_batch)

    iter_source = iter(source_train)
    iter_target = iter(target_train)

    if mu > 0:
        len_share = int(max(0, (batch_size // mu) * (1 - (epoch - 1) / epochs)))
    elif mu == 0:
        len_share = 0
    else:
        len_share = int(batch_size // abs(mu))

    middle_loader = None
    if len_share != 0:
        middle_loader = build_loader(source_dataset, len_share, True, num_workers, True)
        iter_middle = iter(middle_loader)

    loss_cls_meter = 0.0
    loss_dom_meter = 0.0
    acc_src_meter = 0.0

    for step in range(n_batch):
        global_step = (epoch - 1) * n_batch + step
        total_steps = max(1, epochs * n_batch - 1)
        p = global_step / total_steps
        current_rho = max(0.0, rho_0 * (1.0 - p))
        try:
            sample_t, _, speed_t = next(iter_target)
        except StopIteration:
            iter_target = iter(target_train)
            sample_t, _, speed_t = next(iter_target)
        try:
            sample_s, label_s, speed_s = next(iter_source)
        except StopIteration:
            iter_source = iter(source_train)
            sample_s, label_s, speed_s = next(iter_source)

        if middle_loader is not None:
            try:
                sample_m, label_m, speed_m = next(iter_middle)
            except StopIteration:
                iter_middle = iter(middle_loader)
                sample_m, label_m, speed_m = next(iter_middle)
            sample_s = torch.cat((sample_s, sample_m), dim=0)
            label_s = torch.cat((label_s, label_m), dim=0)
            speed_s = torch.cat((speed_s, speed_m), dim=0)

        sample_s, label_s, speed_s = sample_s.to(device), label_s.to(device), speed_s.to(device)
        sample_t, speed_t = sample_t.to(device), speed_t.to(device)

        loss_cls, loss_transfer = model(sample_s, sample_t, label_s, source_speed=speed_s, target_speed=speed_t, epoch=epoch, epochs=epochs)
        loss = loss_cls + lambda_u * loss_transfer
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        logits = model.predict(sample_s[: sample_t.size(0)], speed_s[: sample_t.size(0)])
        pred = logits.argmax(dim=1)
        acc_src = (pred == label_s[: sample_t.size(0)]).float().mean().item()

        loss_cls_meter += float(loss_cls.item())
        loss_dom_meter += float(loss_transfer.item())
        acc_src_meter += acc_src

    logger.info(f"epoch {epoch} source_train cls_loss={loss_cls_meter/n_batch:.4f} transfer_loss={loss_dom_meter/n_batch:.4f} total_loss={(loss_cls_meter + loss_dom_meter)/n_batch:.4f}")
    return {"cls": loss_cls_meter / n_batch, "dom": loss_dom_meter / n_batch, "src_acc": acc_src_meter / n_batch}


def load_pretrained_weights(model: nn.Module, ckpt_path: str, logger: logging.Logger | None = None):
    state = torch.load(ckpt_path, map_location="cpu")
    state_dict = state.get("model", state)
    model_state = model.state_dict()
    matched = {}
    skipped = []
    for key, value in state_dict.items():
        if key in model_state and model_state[key].shape == value.shape:
            matched[key] = value
        else:
            skipped.append(key)
    model_state.update(matched)
    model.load_state_dict(model_state)
    if logger is not None:
        logger.info(f"loaded pretrained weights from {ckpt_path}: matched={len(matched)} skipped={len(skipped)}")


def train_source_only_epoch(model, dl_src, device, optimizer):
    model.train()
    loss_cls_meter = 0.0
    acc_src_meter = 0.0
    steps = max(1, len(dl_src))
    for src_x, src_y, _ in dl_src:
        src_x, src_y = src_x.to(device), src_y.to(device)
        optimizer.zero_grad()
        out = model(src_x=src_x, src_y=src_y)
        loss_cls = out.get("cls_loss", torch.tensor(0.0, device=device))
        logits = out.get("src_logits")
        loss_cls.backward()
        optimizer.step()
        pred = logits.argmax(-1)
        acc_src = (pred == src_y).float().mean().item()
        loss_cls_meter += float(loss_cls.item())
        acc_src_meter += acc_src
    return {"cls": loss_cls_meter / steps, "dom": 0.0, "src_acc": acc_src_meter / steps}


def train_general_epoch(model, dl_src, dl_tgt, device, optimizer, epoch, total_epochs, logger, model_name, rho_0, entropy_weight):
    model.train()
    it_src = iter(dl_src)
    it_tgt = iter(dl_tgt)
    steps = min(len(dl_src), len(dl_tgt))
    loss_cls_meter = 0.0
    loss_dom_meter = 0.0
    acc_src_meter = 0.0
    for step in range(steps):
        src_x, src_y, _ = next(it_src)
        tgt_x, _, _ = next(it_tgt)
        src_x, src_y, tgt_x = src_x.to(device), src_y.to(device), tgt_x.to(device)
        global_step = (epoch - 1) * steps + step
        total_steps = max(1, total_epochs * steps - 1)
        p = global_step / total_steps
        lambd = 2.0 / (1.0 + math.exp(-10 * p)) - 1.0
        current_rho = max(0.0, rho_0 * (1.0 - p))
        optimizer.zero_grad()

        if model_name == "dann":
            out = model(src_x=src_x, tgt_x=tgt_x, src_y=src_y, grl_lambda=lambd)
            loss_cls = out.get("cls_loss", torch.tensor(0.0, device=device))
            loss_dom = out.get("domain_loss", torch.tensor(0.0, device=device))
            loss_ent = torch.tensor(0.0, device=device)
            logits = out.get("src_logits")
        elif model_name == "pdan":
            out = model(src_x=src_x, tgt_x=tgt_x, src_y=src_y, grl_lambda=lambd, adapt=True)
            loss_cls = out.get("cls_loss", torch.tensor(0.0, device=device))
            loss_dom = out.get("domain_loss", torch.tensor(0.0, device=device))
            loss_ent = torch.tensor(0.0, device=device)
            logits = out.get("src_logits")
        else:
            raise ValueError(f"不支持的模型: {model_name}")

        loss = loss_cls + args.domain_weight * loss_dom + entropy_weight * loss_ent
        loss.backward()
        optimizer.step()
        pred = logits.argmax(-1)
        acc_src = (pred == src_y).float().mean().item()
        loss_cls_meter += float(loss_cls.item())
        loss_dom_meter += float(loss_dom.item())
        acc_src_meter += acc_src
    return {"cls": loss_cls_meter / steps, "dom": loss_dom_meter / steps, "src_acc": acc_src_meter / steps}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(ROOT / "3_train" / "config_train.yaml"))
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--model", type=str, choices=["source_only", "dann", "pdan", "ladv", "sladv", "san", "bsan"], default="bsan")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--init_weights", type=str, default=None)
    parser.add_argument("--lambda_u", type=float, default=None)
    parser.add_argument("--domain_weight", type=float, default=1.0)
    parser.add_argument("--entropy_weight", type=float, default=None)
    parser.add_argument("--rho_0", type=float, default=None)
    parser.add_argument("--select_on", type=str, choices=["src", "tgt"], default="tgt")
    args = parser.parse_args()

    set_seed(args.seed)
    cfg = load_config(args.config)
    task_cfg = next((t for t in cfg["tasks"] if t["name"] == args.task), None)
    if task_cfg is None:
        raise ValueError(f"找不到任务: {args.task}")

    exp_name = f'{args.model}_{task_cfg["source"]}_to_{task_cfg["target"]}'
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
    rho_0 = float(args.rho_0) if args.rho_0 is not None else float(cfg["train"].get("rho_0", 1.0))
    entropy_weight = float(args.entropy_weight) if args.entropy_weight is not None else float(cfg["train"].get("entropy_weight", 0.1))
    lambda_u = float(args.lambda_u) if args.lambda_u is not None else float(cfg["train"].get("lambda_u", 1.0))
    aux_source_weight = float(cfg["train"].get("aux_source_weight", 0.0))
    aux_target_weight = float(cfg["train"].get("aux_target_weight", 0.0))
    pseudo_threshold = float(cfg["train"].get("pseudo_threshold", 0.0))
    pseudo_topk = int(cfg["train"].get("pseudo_topk", 0))
    pseudo_warmup_epochs = int(cfg["train"].get("pseudo_warmup_epochs", 0))
    mu = int(cfg["train"].get("mu", 8))
    in_channels = int(cfg["data"]["in_channels"])
    num_classes = int(cfg["data"]["num_classes"])
    class_names = tuple(cfg["data"].get("class_names", [str(i) for i in range(num_classes)]))
    num_workers = int(cfg["train"].get("num_workers", 0))
    processed_dir = str(ROOT / "1_data" / "processed")

    ds_src, ds_tgt = make_datasets(task_cfg["source"], task_cfg["target"], processed_dir)
    source_train = build_loader(ds_src, batch_size, True, num_workers, True)
    target_train = build_loader(ds_tgt, batch_size, True, num_workers, True)

    logger.info(f"task={task_cfg['name']} shared_classes={task_cfg.get('shared_classes')} source_private_classes={task_cfg.get('source_private_classes')}")
    logger.info(f"source_labels={dict(zip(*np.unique(ds_src.y, return_counts=True)))}")
    logger.info(f"target_labels={dict(zip(*np.unique(ds_tgt.y, return_counts=True)))}")
    logger.info(f"source_speed_range=({float(ds_src.s.min()):.4f}, {float(ds_src.s.max()):.4f}) target_speed_range=({float(ds_tgt.s.min()):.4f}, {float(ds_tgt.s.max()):.4f})")

    if args.model == "source_only":
        model = build_dann(in_channels=in_channels, num_classes=num_classes, feat_dim=int(cfg["model"]["feat_dim"]))
    elif args.model == "dann":
        model = build_dann(in_channels=in_channels, num_classes=num_classes, feat_dim=int(cfg["model"]["feat_dim"]))
    elif args.model == "pdan":
        model = build_pdan_selective(in_channels=in_channels, num_classes=num_classes, feat_dim=int(cfg["model"]["feat_dim"]))
    elif args.model in TRANSFER_MODELS:
        model = TransferNet(num_class=num_classes, model_name='cnn_features', in_channel=in_channels, transfer_loss=args.model, max_iter=min(len(source_train), len(target_train)) * epochs, trade_off_adversarial='Step', lam_adversarial=1, batch_size=batch_size, shared_classes=task_cfg.get('shared_classes', []), aux_source_weight=aux_source_weight, aux_target_weight=aux_target_weight, pseudo_threshold=pseudo_threshold, pseudo_topk=pseudo_topk, pseudo_warmup_epochs=pseudo_warmup_epochs)
    else:
        raise ValueError(f"不支持的模型: {args.model}")

    model = model.to(device)
    if args.init_weights:
        load_pretrained_weights(model, args.init_weights, logger)
    if args.model in TRANSFER_MODELS:
        optimizer = torch.optim.Adam(model.get_parameters(initial_lr=lr), lr=lr, weight_decay=float(cfg["train"]["weight_decay"]))
    else:
        optimizer = build_optimizer(model, lr=lr, wd=float(cfg["train"]["weight_decay"]))

    history = {"cls": [], "dom": [], "src_acc": []}
    best_acc = (-1.0,)
    best_path = None
    best_epoch = 0
    shared_classes = task_cfg.get("shared_classes", [])
    private_classes = task_cfg.get("source_private_classes", [])

    for epoch in range(1, epochs + 1):
        if args.model == "source_only":
            stats = train_source_only_epoch(model, source_train, device, optimizer)
        elif args.model in TRANSFER_MODELS:
            stats = train_transfer_epoch(model, source_train, target_train, ds_src, device, optimizer, epoch, epochs, logger, num_workers, batch_size, mu, lambda_u, rho_0)
        else:
            stats = train_general_epoch(model, source_train, target_train, device, optimizer, epoch, epochs, logger, args.model, rho_0, entropy_weight)

        history["cls"].append(stats["cls"])
        history["dom"].append(stats["dom"])
        history["src_acc"].append(stats["src_acc"])
        rep_src = evaluate(model, ds_src, device, num_classes=num_classes)
        rep_tgt = evaluate(model, ds_tgt, device, num_classes=num_classes, shared_classes=shared_classes, private_classes=private_classes)
        tgt_metrics = rep_tgt.get("metrics", {})
        logger.info(
            f"epoch {epoch} src_acc={rep_src['acc']:.4f} tgt_acc={rep_tgt['acc']:.4f} "
            f"tgt_f1={rep_tgt['f1']:.4f} fault_recall={tgt_metrics.get('fault_recall', 0.0):.4f} "
            f"private_pred_rate={tgt_metrics.get('private_pred_rate', 0.0):.4f}"
        )
        plot_curves(history, str(out_plots / "accuracy_curve.png"), title=f"{exp_name} curves")
        plot_confusion(rep_tgt["cm"], class_names, str(out_plots / "confusion_matrix.png"), title=f"{exp_name} tgt")
        if epoch == epochs or (epoch % max(1, epochs // 3) == 0):
            try:
                plot_tsne(model, ds_tgt, device, str(out_plots / f"tsne_epoch_{epoch}.png"), title=f"{exp_name} tgt tsne")
            except Exception:
                pass
        ckpt_path = out_weights / f"epoch_{epoch:03d}.pt"
        torch.save({"model": model.state_dict(), "epoch": epoch, "config": cfg, "task": task_cfg}, ckpt_path)
        if args.select_on == "src" or args.model == "source_only":
            select_key = (float(rep_src["acc"]),)
        else:
            fault_recall = float(tgt_metrics.get("fault_recall", 0.0))
            shared_balanced_acc = float(tgt_metrics.get("shared_balanced_acc", 0.0))
            tgt_acc = float(rep_tgt["acc"])
            select_key = (shared_balanced_acc, fault_recall, tgt_acc)
        if select_key > best_acc:
            best_acc = select_key
            best_epoch = epoch
            best_path = out_weights / "best.pt"
            shutil.copy(str(ckpt_path), str(best_path))

    if args.select_on == "src" or args.model == "source_only":
        logger.info(f"best source acc: {best_acc[0]:.4f}")
    else:
        logger.info(f"best target select_key={best_acc}")
    if best_path is not None:
        logger.info(f"best checkpoint: {best_path}")
        best_state = torch.load(best_path, map_location=device)
        model.load_state_dict(best_state["model"])
        rep_src = evaluate(model, ds_src, device, num_classes=num_classes)
        rep_tgt = evaluate(model, ds_tgt, device, num_classes=num_classes, shared_classes=shared_classes, private_classes=private_classes)
        plot_confusion(rep_tgt["cm"], class_names, str(out_plots / "confusion_matrix.png"), title=f"{exp_name} tgt best")
        save_report(rep_tgt, str(out_logs), prefix="target")
        save_report(rep_src, str(out_logs), prefix="source")
        logger.info(
            f"best epoch={best_epoch} src_acc={rep_src['acc']:.4f} tgt_acc={rep_tgt['acc']:.4f} "
            f"tgt_f1={rep_tgt['f1']:.4f} fault_recall={rep_tgt.get('metrics', {}).get('fault_recall', 0.0):.4f} "
            f"private_pred_rate={rep_tgt.get('metrics', {}).get('private_pred_rate', 0.0):.4f}"
        )


if __name__ == "__main__":
    main()
