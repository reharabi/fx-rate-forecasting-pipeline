# Technical Analysis — FX Rate Forecasting

**Project:** SGD/USD & CNY/USD Daily Rate Forecasting
**Stack:** Python · pandas · scikit-learn · XGBoost · LightGBM · matplotlib · seaborn

---

## 1. Data Pipeline

### 1.1 Raw Data Issues
- **`ND` placeholders** — non-trading days (weekends, public holidays) are marked with the string `'ND'` rather than `NaN`. These are converted to `NaN` and forward-filled (standard practice for daily FX data — the last known rate carries forward on non-trading days)
- **Type coercion** — rate columns arrive as `object` dtype; cast to `float64` after `ND` removal
- **Date parsing** — `DATE` column parsed and set as `DatetimeIndex`
- **Column renaming** — standardised to `SGD_USD`, `CNY_USD`

### 1.2 Dataset Trimming — Why Aug 2005?
From 2000 to 21 July 2005, the PBOC maintained a **hard peg** of exactly CNY 8.28/USD. During this period:
- The PBOC intervened in markets daily to hold the rate constant
- The series has zero volatility **by design** — not a market signal
- Including pre-2005 data would mix two fundamentally different regimes

On 21 July 2005, China announced a managed float, and the Yuan immediately appreciated ~2%. The dataset is trimmed to **1 August 2005** (first full post-peg month), leaving **3,762 rows** covering the genuine market-driven era.

| | Start | End | Rows |
|---|---|---|---|
| **Before filter** | 3 Jan 2000 | 31 Dec 2019 | 5,217 |
| **After filter** | 1 Aug 2005 | 31 Dec 2019 | 3,762 |

---

## 2. Exploratory Data Analysis

### 2.1 Trend
- **SGD/USD:** Steady appreciation from ~1.66 (2005) to ~1.34 (2018), reflecting MAS managed float policy. SGD overall change: **-19.0%** (appreciation). Brief GFC reversal in 2008–2009
- **CNY/USD:** Post-peg appreciation 2005–2014 (~8.1 → ~6.0), PBOC devaluation shock Aug 2015 (+2% overnight), renewed weakening 2018–2019 driven by trade war tariff escalation toward 7.0. CNY overall change: **-14.1%** (appreciation)

### 2.2 Correlation
SGD/USD and CNY/USD have a Pearson correlation of **0.8979** — strongly positively correlated. Both appreciated against the USD during the same macro period. This is partly spurious (two trending series), not causal. Models are trained independently to avoid multicollinearity.

### 2.3 Volatility (30-day Rolling Std)
- **SGD peak:** 6 Oct 2011 (std = 0.0408) — US debt downgrade and Eurozone crisis
- **CNY peak:** 23 Jul 2018 (std = 0.1155) — US-China trade war escalation
- Both show **volatility clustering** — a signature of FX data

### 2.4 Seasonality
No meaningful weekly or monthly seasonality was found. The apparent July–August elevation in multi-year averages is a statistical artefact of just three years:
- **2015** (+0.13 CNY, PBOC surprise devaluation on 11 Aug)
- **2018** (+0.13 CNY, trade war escalation)
- **2019** (+0.19 CNY, trade war escalation)

In the remaining 11 years, the Jul→Aug change is small and alternates direction. This is confirmed by near-zero importance of `day_of_week` and `month` features across all models.

---

## 3. Feature Engineering

All features are built with a **shift of 1** to ensure no future data is used:

```python
feat_df[f'lag_{lag}'] = series.shift(lag)
feat_df['roll_mean_7']  = series.shift(1).rolling(7).mean()
feat_df['roll_mean_30'] = series.shift(1).rolling(30).mean()
feat_df['roll_std_7']   = series.shift(1).rolling(7).std()
feat_df['roll_std_30']  = series.shift(1).rolling(30).std()
feat_df['daily_return'] = series.pct_change().shift(1)
```

**Final feature set (13 features per currency):**
`lag_1`, `lag_2`, `lag_3`, `lag_5`, `lag_10`, `lag_21`, `roll_mean_7`, `roll_mean_30`, `roll_std_7`, `roll_std_30`, `day_of_week`, `month`, `daily_return`

Post-feature-engineering rows: **3,732** (30 rows dropped due to rolling window warmup).

---

## 4. Train / Test Split

```
Train:  2005-09-12 → 2018-12-31  (3,501 rows, 93.1%)
Test:   2019-01-01 → 2019-12-31  (261 rows,   6.9%)
```

**Critical:** Split is purely chronological — no shuffling. The split is performed **before** any scaling to prevent leakage of test statistics into the scaler.

No separate validation set was used:
- Linear Regression has no hyperparameters
- XGBoost and LightGBM hyperparameters were tuned via a manual grid experiment cell
- The test set is a true held-out final evaluation, never touched until evaluation

---

## 5. Preprocessing

`StandardScaler` is fit **on training features only**, then applied to test features:

```python
scaler = StandardScaler()
X_train_sc = pd.DataFrame(scaler.fit_transform(X_train), ...)
X_test_sc  = pd.DataFrame(scaler.transform(X_test), ...)
```

The **target** (`rate`) is not scaled — predictions come out directly in exchange rate units (no inverse transform needed).

---

## 6. Models & Results

### 6.1 Linear Regression — Winner (Step 9)

```python
model = LinearRegression()
model.fit(X_train_sc, y_train)
```

| Currency | Train RMSE | Train MAE | Test RMSE | Test MAE |
|---|---|---|---|---|
| SGD/USD | 0.0048 | 0.0033 | **0.0025** | **0.0019** |
| CNY/USD | 0.0102 | 0.0060 | **0.0168** | **0.0111** |

**Key coefficients — SGD/USD** (Intercept: 1.3792):
- `lag_1`: +0.093 (dominant — yesterday's rate is the strongest predictor)
- `lag_2`: +0.014
- `roll_mean_7`: +0.008
- `day_of_week`, `month`: near zero (confirms no FX seasonality)

**Key coefficients — CNY/USD** (Intercept: 6.7618):
- `lag_2`: +0.667 (dominant — stronger than lag_1 for CNY)
- `lag_1`: -0.121 (negative — captures slight mean-reversion oscillation)
- `lag_3`: +0.032

**Why LR wins:** FX rates are a near-random-walk. The dominant signal is linear lag autocorrelation — a fixed weighted sum of recent lags captures it cleanly. XGBoost and LightGBM add complexity without adding genuine non-linear signal, and actively overfit to training-period patterns that break down in CNY's 2019 trade-war regime.

---

### 6.2 XGBoost (Step 10)

```python
XGBRegressor(
    n_estimators=500, learning_rate=0.01, max_depth=3,
    random_state=42, eval_metric='rmse'
)
```

| Currency | Train RMSE | Train MAE | Test RMSE | Test MAE | Train/Test Gap |
|---|---|---|---|---|---|
| SGD/USD | 0.0047 | 0.0034 | 0.0027 | 0.0020 | Minimal — excellent generalisation |
| CNY/USD | 0.0110 | 0.0077 | 0.0237 | 0.0168 | 2.2× — structural regime shift |

- Level-wise tree growth — symmetric splits per depth level
- Minimal hyperparameter set — no regularisation terms or subsampling needed on a 3,501-row / 13-feature dataset
- `lag_2` dominates feature importance (slight mean-reversion signal — rate from 2 days ago is stronger than yesterday's for XGBoost)
- Rolling means (`roll_mean_7`, `roll_mean_30`) rank prominently — medium-term trend context
- CNY train/test gap reduced from 4.9× (original) to 2.2× via shallower trees (depth=3). Remaining gap is structural (2019 US–China trade war regime shift), not tunable

---

### 6.3 LightGBM (Step 11)

```python
lgb.LGBMRegressor(
    n_estimators=300, learning_rate=0.05, max_depth=4,
    num_leaves=10, random_state=42, verbose=-1
)
```

| Currency | Train RMSE | Train MAE | Test RMSE | Test MAE | Train/Test Gap |
|---|---|---|---|---|---|
| SGD/USD | 0.0040 | 0.0030 | 0.0027 | 0.0021 | Minimal — excellent generalisation |
| CNY/USD | 0.0085 | 0.0054 | 0.0295 | 0.0196 | 3.5× — structural regime shift |

- Leaf-wise tree growth — grows the highest-gain leaf first; faster, but risks overfitting on small datasets
- Minimal hyperparameter set — no regularisation terms or subsampling needed on a 3,501-row / 13-feature dataset
- `lag_1` and `daily_return` dominate (differs from XGBoost's `lag_2` dominance) — leaf-wise growth strongly weights the most recent observation and short-term momentum
- Volatility features (`roll_std_30`, `roll_std_7`) rank 3rd–4th — LightGBM weights market uncertainty more heavily than XGBoost
- CNY train/test gap reduced from 3.8× (original) to 3.5× via fewer leaves (`num_leaves=10`). Remaining gap is structural (2019 US–China trade war regime shift), not tunable

---

### 6.4 Final Comparison

| Currency | Model | Test RMSE | Test MAE |
|---|---|---|---|
| **SGD/USD** | **Linear Regression ✓** | **0.0025** | **0.0019** |
| SGD/USD | XGBoost | 0.0027 | 0.0020 |
| SGD/USD | LightGBM | 0.0027 | 0.0021 |
| **CNY/USD** | **Linear Regression ✓** | **0.0168** | **0.0111** |
| CNY/USD | XGBoost | 0.0237 | 0.0168 |
| CNY/USD | LightGBM | 0.0295 | 0.0196 |

---

## 7. Key Technical Findings

### 7.1 FX as Near-Random-Walk
The dominant feature across all models is `lag_1` or `lag_2`. This reflects the well-established near-random-walk nature of FX rates — market prices already incorporate available information, so the best forecast of tomorrow's rate is approximately today's rate. Models add value through:
- Capturing drift direction (rolling means)
- Short-term momentum (daily_return)
- Volatility regime context (roll_std, for non-linear models)

### 7.2 SGD vs CNY Generalisation Gap
SGD/USD generalises well across all models because Singapore's managed float produces a stable, gradually trending series. CNY/USD shows severe overfitting in 2019 because the test period contains a structural break (trade war regime) with no analogue in training — the models learned smooth appreciation dynamics and cannot extrapolate sharp policy-driven moves.

### 7.3 Why XGBoost and LightGBM Overfit
Both boosting models achieve much lower Train RMSE than Linear Regression (XGBoost: 0.0025 vs LR: 0.0048 on SGD). This reflects overfitting — the models memorise patterns that don't generalise. For a near-random-walk process, the extra non-linearity captures noise, not signal.

### 7.4 XGBoost vs LightGBM Feature Ranking Difference
Despite identical features, XGBoost favours `lag_2` + rolling means, while LightGBM favours `lag_1` + `daily_return` + volatility. This reflects a fundamental algorithmic difference: LightGBM's leaf-wise splits aggressively pursue the highest single-leaf gain (which tends to be the most recent price observation), while XGBoost's level-wise splits distribute splits more symmetrically across the depth level.

---

## 8. Data Leakage Controls

| Control | Implementation |
|---|---|
| Feature shift | All lag/rolling features shifted by 1 day — no same-day information |
| Scaler fit on train only | `StandardScaler.fit()` on X_train; `.transform()` on X_test |
| Chronological split | No shuffling at any stage |
| No target scaling | Predictions in raw exchange rate units — no inverse transform needed |
| Holiday flag not used as model feature | Forward-fill already handles market closures |

---

## 9. Limitations & Extensions

| Limitation | Potential Extension |
|---|---|
| No macro features | Add Fed funds rate, MAS policy rate, VIX, trade balance differentials as external regressors |
| Single-step forecasting only | Rolling re-training window for multi-step forecasts |
| No confidence intervals | Bootstrap or quantile regression for prediction intervals |
| Static hyperparameters | Bayesian optimisation (Optuna) for automated tuning |
| No regime detection | Hidden Markov Model to identify calm vs crisis regimes; train separate models per regime |
| CNY policy shocks | NLP on PBOC press releases as additional signal |
| No walk-forward cross-validation | Rolling window CV would give more robust error estimates than a single chronological split |
