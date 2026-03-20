# Regime Detection Study

## Project understanding

The trading stack is:

1. `data_engine.py` / local monthly OHLCV archive for hourly market data.
2. `signals.py` for cross-sectional alpha generation.
3. `optimizer.py` for regime-dependent BL + HRP portfolio construction.
4. `risk_manager.py` for drawdown control, stop logic, and correlation-spike defense.
5. `main.py` for live orchestration and the portfolio backtest engine.

The regime model affects more than a label:

- signal weighting is conditioned on the regime,
- the optimizer changes its BL/HRP blend and invested percentage by regime,
- stop logic becomes wider in bear markets,
- the trend filter becomes much more aggressive in bear markets.

Because of that, a bad detector hurts ranking, exposure, and turnover at the same time.

## Diagnosis of the legacy detector

The original `TrendRegimeDetector` is a hand-built BTC-only rule system:

- BTC above or below a 3-day / 7-day SMA,
- 7-day SMA slope,
- 14-day drawdown,
- 24-hour BTC momentum,
- hard hysteresis.

Problems observed:

- It ignores market breadth, cross-sectional stress, and correlation structure even though the bot trades a multi-coin portfolio.
- It collapses persistent downtrend and crash-recovery into the same `BEAR` bucket.
- It uses fixed thresholds with no rolling model fit or feature calibration.
- In the long BTC history study, the legacy `BEAR` state often occurred into rebound windows, so the portfolio was de-risking exactly when short-horizon recovery was strongest.

## Literature takeaways

The full bibliography is in `research/regime_detection_literature.md`.

Repeated themes across the reviewed papers:

- Multi-horizon trend and drawdown features matter more than single-horizon returns.
- Realized volatility, downside volatility, and jump-sensitive measures are essential for separating panic from ordinary trend.
- Volume shocks, liquidity, and order-flow proxies improve short-horizon state recognition.
- Cross-sectional breadth and correlation stress are useful for crypto because market-wide co-movement spikes during crashes.
- Pure return-only HMMs are often unstable or uninformative on hourly crypto data.
- Soft or probabilistic regime filtering usually behaves better than brittle hard thresholds.
- Recovery states should often be separated from persistent bear states.

## Models implemented

Implemented in `regime_models.py`:

- `UnsupervisedFeatureRegimeDetector`
  - `model_kind="hmm"` or `model_kind="gmm"`
  - rolling retraining
  - 4 latent states allowed so crash/recovery can stay separate
  - state collapse back to the 3 macro regimes used by the rest of the stack
- `FeatureScoreRegimeDetector`
  - deterministic feature-score regime model
  - breadth, drawdown, realized vol, correlation, and trend features

Integrated in `signals.py`:

- `AlphaEngine` now supports pluggable regime detectors.
- Default config is now `cfg.SIGNALS.regime_model = "feature_hmm"`.

Research tooling:

- `regime_research.py` runs detector studies and backtests using `data/monthly_ohlcv`.

## Test setup

Universe:

- Local "stable" coin universe discovered from the monthly archive with at least 24 monthly hourly files.
- In the runs below this yielded 39 coins.

Execution assumptions:

- Same backtest engine, optimizer, risk logic, and adaptive signal weighting as the existing system.
- Rebalance frequency: 2 hours.
- Data source: local `data/monthly_ohlcv`, not live Binance fetches.

Important note:

- Standalone regime-label quality metrics were noisy because crypto bear windows contain violent rebounds.
- The primary decision metric was downstream portfolio composite score, not just next-period BTC return by label.

## Results

### Q1 2025 stress window: 2025-01-01 to 2025-03-15

| Detector | Return % | Max DD % | Composite | Trades |
|---|---:|---:|---:|---:|
| legacy_trend | -14.19 | 21.83 | -0.0143 | 1714 |
| hmm_return | -11.59 | 19.79 | -0.0119 | 1860 |
| hmm_trend_vol | -11.59 | 19.79 | -0.0119 | 1860 |
| hmm_market | -12.13 | 22.42 | -0.0086 | 2168 |
| score_market | -17.40 | 23.81 | -0.0170 | 1914 |

Interpretation:

- `hmm_market` had the best composite score in the hardest regime.
- `hmm_return` reduced drawdown more, but its total risk-adjusted score was still worse than `hmm_market`.
- The deterministic score model underperformed badly in this window.

### Q4 2024 bull window: 2024-10-01 to 2024-12-31

| Detector | Return % | Max DD % | Composite | Trades |
|---|---:|---:|---:|---:|
| legacy_trend | 104.82 | 16.56 | 0.0425 | 2726 |
| hmm_market | 121.10 | 19.55 | 0.0423 | 3179 |

Interpretation:

- `hmm_market` made more money but also accepted more drawdown and turnover.
- On the hackathon composite score this window was effectively flat between the two.

### Full mixed window: 2024-10-01 to 2025-03-15

| Detector | Return % | Max DD % | Composite | Trades |
|---|---:|---:|---:|---:|
| legacy_trend | 82.06 | 22.75 | 0.0231 | 4489 |
| hmm_return | 54.80 | 23.98 | 0.0212 | 4333 |
| hmm_market | 102.58 | 23.84 | 0.0252 | 5049 |

Interpretation:

- `hmm_market` was the best full-window detector among the tested variants.
- It improved the composite score versus the legacy detector by 0.0021 in absolute terms.
- It also lifted total return materially, though at the cost of more turnover.

## Conclusion

Selected model: `hmm_market`.

Why:

- Best composite score in the difficult mixed full-window test.
- Best composite score among tested detectors in the Q1 2025 stress period.
- More faithful to the portfolio actually traded, because it uses market-wide breadth and stress features instead of BTC-only thresholds.
- Keeps the rest of the stack intact: same 3 macro regimes, same optimizer contract, same risk-manager behavior.

## Caveats

- The new detector still increases turnover.
- The unsupervised state collapse can still be improved further, especially around fast crash-recovery transitions.
- The current implementation uses only OHLCV-derived features. The literature suggests that funding, open interest, on-chain activity, and order-book features could improve this further.
- The research runs here used the local monthly archive and a stable-coverage universe, so exact numbers will differ from older repo backtests that used a broader dynamic universe.

## Next steps

Highest-value follow-ups:

1. Add perpetual-futures funding and open-interest features.
2. Add a dedicated recovery state internally and map it to neutral or reduced-risk exposure rather than forcing a pure 3-state latent model.
3. Penalize turnover directly inside the regime-selection study, not only downstream in the backtest.
4. Re-run the matrix on a longer 2025-2026 window after expanding the local archive.
