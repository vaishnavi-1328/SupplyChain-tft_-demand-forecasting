# Project 4 — Demand Forecasting with Temporal Fusion Transformers

**"Forecast the Future, But Admit Uncertainty"**

---

## What This Is

A from-scratch PyTorch implementation of the **Temporal Fusion Transformer (TFT)** paper:

> Lim et al., *Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting*, 2019.
> https://arxiv.org/abs/1912.09363

Applied to the RetailSim supply chain dataset from Project 1:
- 50 stores × 200 SKUs × 2 years of daily demand
- Forecast horizon: 14 days ahead
- Output: P10 / P50 / P90 quantile forecasts

---

## Why TFT Over XGBoost or ARIMA

Most retail demand forecasting tutorials produce a single number: "we predict 112 units."

That number is useless for an inventory decision without uncertainty.

TFT solves three problems simultaneously:

| Problem | TFT Solution |
|---|---|
| Heterogeneous inputs (static metadata, known future events, historical observations) | Variable Selection Networks route each type through the right pathway |
| Long-range + short-range temporal patterns | LSTM (local) + Multi-Head Attention (long-range) in series |
| Opaque black-box | Interpretable attention + VSN selection weights for every prediction |

The P10/P50/P90 forecasts connect directly to inventory decisions:
- **Order to P50** → service level ~50% → frequent stockouts
- **Order to P90** → service level ~90% → higher inventory holding cost
- The tradeoff is explicit and quantified

---

## Architecture

```
Input Types
├── Static covariates       → Embedding → Variable Selection Network
│   (store format, SKU category, is_perishable, ...)
├── Known future inputs     → Linear projection → Variable Selection Network
│   (day_of_week, holiday, promotion_flag, ...)
└── Observed past inputs    → Linear projection → Variable Selection Network
    (past sales, stockout_flag, weather_severity)

               ↓ Static Covariate Encoder produces 4 context vectors (cs, ce, cc, ch)

Past inputs  → LSTM Encoder  (initialized with cc, ch)  ─┐
Future inputs → LSTM Decoder  (continues from encoder)  ─┤→ Temporal sequence
                                                          ↓
                                              Static Enrichment (ce context)
                                                          ↓
                                   Interpretable Multi-Head Self-Attention
                                   (shared V weights, additive aggregation)
                                                          ↓
                                        Position-wise Feed-Forward (GRN)
                                                          ↓
                                   Quantile Output: P10 / P50 / P90 per day
```

---

## Inputs and What They Teach

| Feature Type | Variables | What the Model Learns |
|---|---|---|
| Static categorical | store_format, category, subcategory, is_perishable, dc_region | Each entity type has different demand behavior |
| Static continuous | store_size_sqft, unit_price | Scale and price sensitivity |
| Known future | day_of_week, month, is_holiday, is_weekend, season, promotion_flag, discount_pct | What's coming up that will affect demand |
| Observed past | log_units_sold, stockout_flag, weather_severity | What actually happened in history |

---

## How to Run

### Prerequisites
```bash
# Install dependencies
pip install -r requirements.txt

# Set your database URL (from Project 1)
export SUPPLY_CHAIN_DB_URL="postgresql://user:pass@host:5432/supply_chain_db"
```

### Full pipeline
```bash
cd project_04_demand_forecasting
python run_pipeline.py
```

### Step by step
```bash
# 1. Extract data from PostgreSQL → parquet
python data/extract.py

# 2. Feature engineering
python data/preprocess.py

# 3. Verify dataset shapes
python data/dataset.py

# 4. Train (2-epoch smoke test)
python training/trainer.py --epochs 2 --debug

# 5. Full training
python training/trainer.py

# 6. Full pipeline with interpretability
python run_pipeline.py --skip-extract --skip-preprocess
```

---

## Output Files

```
outputs/
├── evaluation_results.json          # q-Risk for TFT vs baselines
└── interpretability/
    ├── variable_importance.png      # Which features the model uses most
    ├── importance_static.csv
    ├── importance_past.csv
    ├── importance_future.csv
    ├── temporal_attention.png       # Which past time steps matter per horizon
    └── regime_detection.png        # Anomalous demand periods detected
checkpoints/
├── best_model.pt                   # Best checkpoint by val loss
└── history.json                    # Loss curve per epoch
```

---

## What to Look For in Results

**Variable Importance**
- `had_promotion_enc` and `is_holiday_enc` should rank high in known future variables — these are the strongest signals baked into the causal demand model
- `log_units_sold` (lagged) should dominate observed past — recent history is the strongest single predictor

**Temporal Attention**
- Strong attention at **lag-7** confirms the model learned weekly seasonality
- Attention at **lag-1** confirms recency bias in retail demand

**Regime Detection**
- Peaks should align with Black Friday (Nov 2023/2024), Thanksgiving, and Christmas — the holidays with the highest demand multipliers in the data generator

**Business Finding (Notebook 4)**
- Ordering to P50 gives ~50% fill rate → unacceptable
- Ordering to P90 increases inventory holding cost by ~X% but reduces stockout risk from Y% to Z%
- This is the core supply chain learning: **forecast uncertainty ≠ forecasting failure**; it's the input to a safety stock formula

---

## Key Learning Concepts Covered

From `learning_objective.md`:
- ✅ #1 Demand forecasting
- ✅ #2 Time-series modeling
- ✅ #19 Forecast uncertainty
- ✅ #20 Probabilistic forecasting
- ✅ #37 Explainable ML
- ✅ #5 Safety stock (connected in notebook 4)
- ✅ #7 Service levels (P10/P50/P90 → fill rate tradeoff)

---

## Project Structure

```
project_04_demand_forecasting/
├── data/
│   ├── extract.py          PostgreSQL → parquet
│   ├── preprocess.py       Feature engineering + normalization
│   └── dataset.py          PyTorch Dataset
├── models/
│   ├── components.py       GRN, VariableSelectionNetwork, StaticCovariateEncoder
│   ├── attention.py        InterpretableMultiHeadAttention
│   └── tft.py              Full TFT model
├── training/
│   ├── config.py           Hyperparameters
│   ├── losses.py           QuantileLoss + q-Risk metric
│   └── trainer.py          Training loop
├── evaluation/
│   ├── metrics.py          Test set evaluation
│   └── baselines.py        Seasonal naive + moving average
├── interpretability/
│   ├── variable_importance.py   VSN weight analysis
│   ├── temporal_patterns.py     Attention heatmap
│   └── regime_detection.py      Bhattacharyya deviation scoring
├── notebooks/              (add your own analysis here)
├── run_pipeline.py         End-to-end orchestrator
└── requirements.txt
```
