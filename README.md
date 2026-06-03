# FX Rate Forecasting — SGD/USD & CNY/USD

**Domain:** Finance & Banking | **Type:** Time Series Regression | **Language:** Python

---

## Overview

This project forecasts daily foreign exchange rates for two currency pairs — **SGD/USD** (Singapore Dollar) and **CNY/USD** (Chinese Yuan) — using 14 years of historical data (Aug 2005–Dec 2019). Three models are trained, evaluated, and compared on a fully held-out 2019 test set.

The project mirrors a real-world bank FX desk workflow: build models on historical data, evaluate on a held-out future period, and interpret results in terms of business impact (bid-ask spread risk, pricing error per transaction).

---

## Results

| Currency | Model | Test RMSE | Test MAE | Train RMSE |
|---|---|---|---|---|
| **SGD/USD** | **Linear Regression ✓** | **0.0025** | **0.0019** | 0.0048 |
| SGD/USD | XGBoost | 0.0027 | 0.0020 | 0.0047 |
| SGD/USD | LightGBM | 0.0027 | 0.0021 | 0.0040 |
| **CNY/USD** | **Linear Regression ✓** | **0.0168** | **0.0111** | 0.0102 |
| CNY/USD | XGBoost | 0.0237 | 0.0168 | 0.0110 |
| CNY/USD | LightGBM | 0.0295 | 0.0196 | 0.0085 |

> Test period: Jan–Dec 2019 (full held-out year, never seen during training)

**Winner: Linear Regression on both currencies.** XGBoost and LightGBM overfit during training — their lower Train RMSE does not translate to better test performance. FX rates behave as a near-random-walk; simple lag-based linear weights generalise better than complex boosting models, especially on CNY's 2019 trade-war regime shifts.

---

## Business Context

Exchange rate forecasting is a core problem in banking and finance:
- **FX desks** need rate predictions to price forward contracts and hedge currency exposure
- **Importers/exporters** use FX forecasts to decide when to convert currency
- A Test RMSE of **0.0025 on SGD/USD** translates to **SGD 2,500 average pricing error per USD 1,000,000** traded — tight enough to support competitive bid-ask spreads

---

## Dataset

| Property | Detail |
|---|---|
| **Source** | `Foreign_Exchange_Rates.csv` (US Federal Reserve) |
| **Raw rows** | 5,217 (Jan 2000 – Dec 2019) |
| **After date filter** | 3,762 (Aug 2005 – Dec 2019) |
| **Targets** | SGD/USD rate · CNY/USD rate |
| **Train** | Aug 2005 – Dec 2018 (3,501 rows, 93.1%) |
| **Test** | Jan 2019 – Dec 2019 (261 rows, 6.9%) |

**Why start from Aug 2005?** China maintained a hard peg of CNY 8.28/USD from 2000 to July 2005. Pre-peg CNY data is government-controlled with zero volatility — not real market data. Both series are trimmed to the post-liberalisation era.

---

## Pipeline

```
Raw CSV (5,217 rows)
  │
  ├─ Step 1:   Setup & Imports
  ├─ Step 2:   Load Data (Google Drive mount)
  ├─ Step 3:   Data Inspection
  │             └─ ND placeholders · type conversion · distribution check
  ├─ Step 4:   Data Cleaning
  │             └─ Replace ND→NaN · forward-fill · parse dates
  ├─ Step 4.5: Date Filter → 3,762 rows (Aug 2005 onwards)
  ├─ Step 5:   EDA
  │             └─ Trend · Moving averages · Seasonality · Correlation · Volatility · Events
  ├─ Step 6:   Feature Engineering (13 features per currency)
  │             └─ lags · rolling stats · daily return · calendar
  ├─ Step 7:   Chronological Train/Test Split (no shuffle)
  ├─ Step 8:   StandardScaler (fit on train only — no leakage)
  ├─ Step 9:   Linear Regression (baseline + winner)
  ├─ Step 10:  XGBoost
  ├─ Step 11:  LightGBM
  └─ Step 12:  Model Comparison & Business Interpretation
```

---

## Features Engineered

| Feature | Description |
|---|---|
| `lag_1`, `lag_2`, `lag_3`, `lag_5`, `lag_10`, `lag_21` | Previous 1–21 days' rates |
| `roll_mean_7`, `roll_mean_30` | 7-day and 30-day rolling average |
| `roll_std_7`, `roll_std_30` | 7-day and 30-day rolling volatility |
| `daily_return` | % change from previous day (momentum) |
| `day_of_week` | 0=Mon to 4=Fri |
| `month` | 1–12 |

All lag and rolling features are **shifted by 1 day** to prevent lookahead leakage.

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| **Chronological train/test split** | No shuffling — future data must never appear in training |
| **Scaler fit on train only** | Prevents leakage of test statistics into the scaler |
| **No validation set** | 3,501 training rows is small; LR has no hyperparameters; XGBoost/LightGBM tuned via manual experiment cell |
| **Dataset starts Aug 2005** | Pre-2005 CNY is a government-fixed peg — not real market data |
| **Outliers kept** | GFC 2008 and CNY 2015 devaluation are genuine market events, not data errors |
| **RMSE as primary metric** | Penalises large errors more than MAE — important for FX risk management |
| **No ARIMA** | ARIMA assumes stationarity; FX rates are non-stationary (trending). Lag features in ML models achieve the same autocorrelation effect without the stationarity constraint |
| **Separate models per currency** | SGD/CNY correlation (0.8979) is partly spurious (shared trend); cross-currency features would add multicollinearity |

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Python | Core language |
| pandas · numpy | Data manipulation |
| scikit-learn | Linear Regression · StandardScaler |
| XGBoost | Gradient boosting |
| LightGBM | Gradient boosting |
| matplotlib · seaborn | Visualisation |
| Google Colab | Execution environment |

---

## How to Run

1. Upload [`Foreign_Exchange_Rates.csv`](https://github.com/reharabi/fx-rate-forecasting-pipeline/commit/0ff535fcae0ecf5972929a360963afea229a54cd) to Google Drive → `MyDrive/`
2. Open [`fx_forecasting.ipynb`](https://github.com/reharabi/fx-rate-forecasting-pipeline/blob/main/fx_forecasting%20(1).ipynb) in Google Colab
3. Run **Step 1** to install LightGBM (`!pip install lightgbm`)
4. Mount Google Drive when prompted in **Step 2**
5. Run all cells in order — no other configuration needed

---

## Limitations

- **Single-step forecasting only** — predictions are one day ahead; errors accumulate for multi-day horizons
- **No macro features** — model cannot anticipate PBOC announcements, Fed rate decisions, or geopolitical shocks
- **Regime shifts** — CNY's 2019 trade-war driven weakening is a structural break that lag-based models cannot predict in advance
- **No walk-forward validation** — a single chronological split is used; rolling-window cross-validation would give more robust error estimates
