"""
Training loop for TFT.

Usage:
  python training/trainer.py                   # full run
  python training/trainer.py --epochs 2 --debug  # quick sanity check
"""
import os
import sys
import argparse
import json
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from training.config import TFTConfig
from training.losses import QuantileLoss, q_risk
from models.tft import TemporalFusionTransformer
from data.dataset import load_datasets, STATIC_CAT_COLS


def get_cat_cardinalities(train_dataset) -> list[int]:
    """Infer max category index per static categorical variable from training samples."""
    max_vals = [0] * len(STATIC_CAT_COLS)
    for sample in train_dataset.samples:
        for i, v in enumerate(sample["static_cat"]):
            max_vals[i] = max(max_vals[i], int(v))
    return [v + 1 for v in max_vals]  # +1 so embedding has enough rows


def train_epoch(model, loader, optimizer, criterion, device, grad_clip):
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(
            batch["static_cat"],
            batch["static_cont"],
            batch["known_past"],
            batch["known_future"],
            batch["observed_past"],
        )
        loss = criterion(out["quantile_preds"], batch["target"])
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, criterion, cfg, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(
            batch["static_cat"],
            batch["static_cont"],
            batch["known_past"],
            batch["known_future"],
            batch["observed_past"],
        )
        loss = criterion(out["quantile_preds"], batch["target"])
        total_loss += loss.item()
        all_preds.append(out["quantile_preds"].cpu())
        all_targets.append(batch["target"].cpu())

    all_preds   = torch.cat(all_preds,   dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    risks = q_risk(all_preds, all_targets, cfg.quantiles)
    return total_loss / len(loader), risks


def main(cfg: TFTConfig, debug: bool = False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Data ─────────────────────────────────────────────────────────────────────
    print("Loading datasets...")
    train_ds, val_ds, test_ds = load_datasets(cfg)
    if debug:
        train_ds.samples = train_ds.samples[:500]
        val_ds.samples   = val_ds.samples[:200]

    print(f"  Train: {len(train_ds):,}  Val: {len(val_ds):,}  Test: {len(test_ds):,}")

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.batch_size, shuffle=False, num_workers=2)

    # ── Model ─────────────────────────────────────────────────────────────────────
    cat_cards = get_cat_cardinalities(train_ds)
    print(f"  Category cardinalities: {cat_cards}")
    model = TemporalFusionTransformer(cfg, cat_cards).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}")

    # ── Optimizer & Loss ──────────────────────────────────────────────────────────
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    criterion = QuantileLoss(cfg.quantiles)

    # ── Training loop ─────────────────────────────────────────────────────────────
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    best_val_loss = float("inf")
    patience_counter = 0
    history = []

    epochs = 2 if debug else cfg.epochs
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, cfg.grad_clip)
        val_loss, val_risks = evaluate(model, val_loader, criterion, cfg, device)
        elapsed = time.time() - t0

        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
               **{f"val_{k}": v for k, v in val_risks.items()}}
        history.append(row)

        print(f"Epoch {epoch:3d}/{epochs}  "
              f"train={train_loss:.4f}  val={val_loss:.4f}  "
              f"p50={val_risks.get('p50', float('nan')):.4f}  "
              f"p90={val_risks.get('p90', float('nan')):.4f}  "
              f"({elapsed:.1f}s)")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            ckpt_path = os.path.join(cfg.checkpoint_dir, "best_model.pt")
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_loss": val_loss,
                "cat_cardinalities": cat_cards,
            }, ckpt_path)
            print(f"  ✓ Saved checkpoint (val_loss={val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= cfg.patience:
                print(f"Early stopping at epoch {epoch} (patience={cfg.patience})")
                break

    # Save training history
    with open(os.path.join(cfg.checkpoint_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    return model, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    cfg = TFTConfig()
    if args.epochs:
        cfg.epochs = args.epochs
    main(cfg, debug=args.debug)
