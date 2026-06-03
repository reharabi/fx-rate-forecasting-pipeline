# Executive Summary — FX Rate Forecasting

**Project:** Daily Exchange Rate Forecasting for SGD/USD and CNY/USD
**Domain:** Finance & Banking
**Data:** 14 years of daily FX rates (Aug 2005 – Dec 2019, 3,762 observations)
**Evaluation Period:** Jan–Dec 2019 (held-out full year, never seen during training)

---

## What Was Built

A machine learning pipeline that forecasts daily exchange rates for two currency pairs used heavily in Asian banking and trade finance. Three models were trained and compared on identical data and features.

| Model | Role | Test RMSE (SGD) | Test RMSE (CNY) |
|---|---|---|---|
| **Linear Regression** | **Winner** — interpretable, generalises best | **0.0025** | **0.0168** |
| XGBoost | Gradient boosting — non-linear patterns | 0.0027 | 0.0237 |
| LightGBM | Alternative gradient boosting | 0.0027 | 0.0295 |

**Surprise result:** The simplest model wins. Linear Regression outperforms both boosting models on the held-out 2019 test set. XGBoost and LightGBM fit the training data very well but overfit — their patterns don't transfer to CNY's 2019 trade-war regime. Hyperparameter tuning reduced XGBoost's CNY train/test gap from 4.9× to 2.2×, and LightGBM's from 3.8× to 3.5×, but neither can overcome a structural regime shift with price-only features.

---

## What the Numbers Mean for a Bank

The primary metric is **RMSE** — the average gap between the model's predicted rate and the actual rate.

For a bank processing **USD 1,000,000 per transaction**:

| Currency | Model | RMSE | Avg Pricing Error per USD 1M |
|---|---|---|---|
| **SGD/USD** | Linear Regression | 0.0025 | **SGD 2,500** |
| **CNY/USD** | Linear Regression | 0.0168 | **CNY 16,800** |

The smaller the RMSE, the tighter the bank can set bid-ask spreads while remaining profitable. SGD forecasting is accurate enough to consider for live pricing support. CNY forecasting remains challenging in volatile periods — the 2019 trade war caused sudden regime shifts no historical model can anticipate.

---

## Key Findings

### 1. Simpler Is Better for FX
Exchange rates behave like a near-random-walk — today's rate is the best predictor of tomorrow's rate. A simple linear combination of recent lag features captures this cleanly. XGBoost and LightGBM add non-linearity, but there is no non-linear signal to capture — so their extra complexity only introduces overfitting.

### 2. SGD — Stable and Predictable
Singapore's managed float policy produces a stable, gradually appreciating currency. The SGD/USD trend from 2005 to 2018 was a steady 19% appreciation. All models forecast well here, but Linear Regression's test RMSE of **0.0025** edges out both boosting models.

### 3. CNY — Trade War Breaks Everything
The 2018–2019 US-China trade war caused CNY/USD to weaken sharply toward 7.0, driven by surprise tariff escalation announcements. This is a structural break — a regime with no historical precedent in the training data. XGBoost and LightGBM overfit to the smooth appreciation dynamics in training and diverge badly in 2019. Linear Regression, with its fixed lag weights, degrades less severely (RMSE 0.0168 vs 0.0237 for XGBoost).

### 4. No FX Seasonality Exists
Calendar features (day of week, month) are near-zero importance in all three models. The apparent July–August pattern in raw averages is caused by just three years of macro shocks (2015 PBOC devaluation, 2018–2019 trade war) — not a recurring seasonal effect that models can exploit.

---

## What Drives the Forecasts

| Driver | Explanation |
|---|---|
| **Yesterday's rate (lag_1)** | The single strongest predictor across all models — markets are efficient |
| **Rate from 2 days ago (lag_2)** | XGBoost picks up a subtle mean-reversion correction here |
| **7-day rolling average** | Short-term trend direction — models use this for momentum context |
| **30-day rolling average** | Medium-term trend — provides broader regime context |
| **Daily return** | Recent momentum — captures whether rate has been rising or falling |
| **Calendar features** | Near-zero importance — FX markets don't follow a calendar schedule |

---

## Practical Recommendations

| Recommendation | Action |
|---|---|
| **Deploy Linear Regression for both currencies** | It wins on both SGD and CNY; simpler models are more robust for near-random-walk FX rates |
| **Implement volatility alerts** | When 30-day rolling std exceeds 90th percentile, flag as high-risk and automatically widen bid-ask spreads |
| **Treat CNY forecasts as indicative only** | During PBOC announcements or geopolitical escalation, model errors widen sharply — human oversight essential |
| **Retrain monthly** | FX dynamics shift; stale models degrade in performance quickly |
| **Add macro features for CNY** | Interest rate differentials, VIX, and PBOC press release signals would improve CNY forecasting beyond what price data alone can achieve |

---

## What This Project Does Not Cover

| Limitation | Implication |
|---|---|
| No macro/fundamental data | Models miss central bank announcements and policy shifts |
| Single-step forecasting only | Accuracy degrades for multi-day horizons |
| No live trading integration | Results are research-grade, not production-deployed |
| Dataset ends Dec 2019 | COVID-19 volatility (Feb 2020 onward) is not modelled |
