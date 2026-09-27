"""
End-to-end pipeline runner for Project 4: TFT Demand Forecasting.

Steps:
  1. Extract raw data from PostgreSQL
  2. Preprocess (feature engineering, normalization)
  3. Train TFT model
  4. Evaluate on test set (q-Risk vs. baselines)
  5. Generate interpretability outputs

Usage:
  python run_pipeline.py                  # full run
  python run_pipeline.py --skip-extract   # if parquet already exists
  python run_pipeline.py --debug          # fast 2-epoch smoke test
"""
import os
import sys
import argparse
import torch
import json
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))

from training.config import TFTConfig
from data.extract import extract
from data.preprocess import process
from data.dataset import load_datasets, STATIC_CAT_COLS
from training.trainer import main as train_main, get_cat_cardinalities
from training.losses import QuantileLoss
from models.tft import TemporalFusionTransformer
from evaluation.metrics import evaluate_model
from evaluation.baselines import benchmark_baselines
from interpretability.variable_importance import collect_weights, plot_variable_importance
from interpretability.temporal_patterns import collect_attention, plot_attention_heatmap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-extract",    action="store_true")
    parser.add_argument("--skip-preprocess", action="store_true")
    parser.add_argument("--skip-train",      action="store_true")
    parser.add_argument("--debug",           action="store_true")
    args = parser.parse_args()

    cfg = TFTConfig()
    data_dir   = "data/parquet"
    raw_path   = os.path.join(data_dir, "demand_raw.parquet")
    proc_path  = os.path.join(data_dir, "demand_processed.parquet")
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)

    # ── Step 1: Extract ────────────────────────────────────────────────────────
    if not args.skip_extract:
        if not cfg.db_url:
            raise EnvironmentError("SUPPLY_CHAIN_DB_URL not set. Run: export SUPPLY_CHAIN_DB_URL=...")
        print("=" * 60)
        print("STEP 1: Extracting data from PostgreSQL")
        print("=" * 60)
        extract(cfg, data_dir)
    else:
        print("Skipping extract (--skip-extract)")

    # ── Step 2: Preprocess ────────────────────────────────────────────────────
    if not args.skip_preprocess:
        print("=" * 60)
        print("STEP 2: Preprocessing features")
        print("=" * 60)
        process(raw_path, data_dir)
    else:
        print("Skipping preprocess (--skip-preprocess)")

    # ── Step 3: Train ─────────────────────────────────────────────────────────
    if not args.skip_train:
        print("=" * 60)
        print("STEP 3: Training TFT")
        print("=" * 60)
        model, history = train_main(cfg, debug=args.debug)
    else:
        print("Skipping training (--skip-train) — loading checkpoint")

    # ── Step 4: Evaluate ──────────────────────────────────────────────────────
    print("=" * 60)
    print("STEP 4: Evaluation")
    print("=" * 60)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, _, test_ds = load_datasets(cfg, proc_path)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=2)

    ckpt_path = os.path.join(cfg.checkpoint_dir, "best_model.pt")
    ckpt = torch.load(ckpt_path, map_location=device)
    cat_cards = ckpt["cat_cardinalities"]
    model = TemporalFusionTransformer(cfg, cat_cards).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # TFT metrics
    tft_metrics = evaluate_model(model, test_loader, cfg, device)
    print(f"\nTFT Test Metrics: {json.dumps(tft_metrics, indent=2)}")

    # Baseline comparison
    baseline_metrics = benchmark_baselines(test_loader, cfg)
    print(f"\nBaseline Metrics:")
    for name, m in baseline_metrics.items():
        print(f"  {name}: {m}")

    results = {"tft": tft_metrics, "baselines": baseline_metrics}
    with open(os.path.join(output_dir, "evaluation_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # ── Step 5: Interpretability ───────────────────────────────────────────────
    print("=" * 60)
    print("STEP 5: Interpretability")
    print("=" * 60)
    interp_dir = os.path.join(output_dir, "interpretability")

    weights = collect_weights(model, test_loader, device)
    plot_variable_importance(weights, interp_dir)

    avg_attn = collect_attention(model, test_loader, device, cfg)
    plot_attention_heatmap(avg_attn, cfg, interp_dir)

    print("\nPipeline complete. Outputs saved to:", output_dir)


if __name__ == "__main__":
    main()
