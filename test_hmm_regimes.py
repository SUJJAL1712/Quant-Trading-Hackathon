"""Test HMM with different regime counts on BTC hourly returns."""
import numpy as np
import pandas as pd
import warnings
import logging
from data_engine import BinanceDataClient

logging.basicConfig(level=logging.WARNING)

# Fetch 6 months of BTC hourly data
binance = BinanceDataClient()
start_ms = int(pd.Timestamp('2024-09-01', tz='UTC').timestamp() * 1000)
end_ms = int(pd.Timestamp('2025-03-15', tz='UTC').timestamp() * 1000)

all_data = []
current_start = start_ms
while current_start < end_ms:
    df = binance.fetch_klines('BTCUSDT', '1h', start_time=current_start, end_time=end_ms, limit=1000)
    if df.empty:
        break
    all_data.append(df)
    last_ts = int(df.index[-1].timestamp() * 1000)
    if last_ts <= current_start:
        break
    current_start = last_ts + 1

btc = pd.concat(all_data)
btc = btc[~btc.index.duplicated(keep='last')].sort_index()
returns = np.log(btc['close'] / btc['close'].shift(1)).dropna()
print(f"BTC hourly data: {len(returns)} rows ({returns.index[0]} to {returns.index[-1]})")

# Winsorize
mu, sigma = returns.mean(), returns.std()
returns_clean = returns.clip(mu - 5*sigma, mu + 5*sigma)
data = returns_clean.values.reshape(-1, 1)

from hmmlearn.hmm import GaussianHMM

seeds = [0, 42, 99, 123, 314, 555]
lookbacks = [360, 720, 1080]  # 15d, 30d, 45d

for n_states in [2, 3, 4, 5]:
    print(f"\n{'='*60}")
    print(f"  HMM with {n_states} STATES")
    print(f"{'='*60}")

    for lookback in lookbacks:
        train_data = data[-lookback:]

        best_model = None
        best_score = -np.inf

        hmm_logger = logging.getLogger("hmmlearn")
        hmm_logger.setLevel(logging.ERROR)

        for seed in seeds:
            try:
                model = GaussianHMM(
                    n_components=n_states,
                    covariance_type="full",
                    n_iter=200,
                    tol=1e-4,
                    random_state=seed,
                )
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model.fit(train_data)
                score = model.score(train_data)
                if score > best_score:
                    best_score = score
                    best_model = model
            except Exception:
                continue

        if best_model is None:
            print(f"  Lookback {lookback}h: ALL FITS FAILED")
            continue

        # Analyze states
        means = best_model.means_.flatten()
        stds = np.sqrt(best_model.covars_.flatten())
        sorted_idx = np.argsort(means)

        hidden = best_model.predict(train_data)
        probs = best_model.predict_proba(train_data)

        # Count transitions
        transitions = sum(1 for i in range(1, len(hidden)) if hidden[i] != hidden[i-1])

        # Current state
        current_raw = hidden[-1]
        current_prob = probs[-1][current_raw]

        print(f"\n  Lookback: {lookback}h | Score: {best_score:.1f} | Transitions: {transitions}")

        # Map states by mean return
        state_names = []
        for rank, idx in enumerate(sorted_idx):
            pct_time = (hidden == idx).sum() / len(hidden) * 100
            ann_ret = means[idx] * 24 * 365 * 100  # annualized hourly return
            ann_vol = stds[idx] * np.sqrt(24 * 365) * 100

            if n_states == 2:
                name = ["BEAR", "BULL"][rank]
            elif n_states == 3:
                name = ["BEAR", "NEUTRAL", "BULL"][rank]
            elif n_states == 4:
                name = ["STRONG_BEAR", "MILD_BEAR", "MILD_BULL", "STRONG_BULL"][rank]
            else:
                name = f"STATE_{rank}"

            marker = " <-- CURRENT" if idx == current_raw else ""
            print(f"    {name:12s}: mean={means[idx]*100:+.4f}%/h  vol={stds[idx]*100:.4f}%/h  "
                  f"ann_ret={ann_ret:+.0f}%  ann_vol={ann_vol:.0f}%  time={pct_time:.0f}%{marker}")
            state_names.append(name)

        # Show regime changes in last 2 weeks (Mar 1-15)
        mar1 = pd.Timestamp('2025-03-01', tz='UTC')
        recent_mask = returns_clean.index >= mar1
        recent_hidden = best_model.predict(data[recent_mask])
        recent_probs = best_model.predict_proba(data[recent_mask])
        recent_idx = returns_clean.index[recent_mask]

        if len(recent_hidden) > 0:
            # Show daily regime
            for day_start in pd.date_range('2025-03-01', '2025-03-14', freq='D', tz='UTC'):
                day_mask = (recent_idx >= day_start) & (recent_idx < day_start + pd.Timedelta(days=1))
                day_states = recent_hidden[day_mask[:len(recent_hidden)]]
                if len(day_states) > 0:
                    # Most common state
                    from collections import Counter
                    most_common = Counter(day_states.tolist()).most_common(1)[0]
                    state_idx = most_common[0]
                    # Map to name
                    rank = list(sorted_idx).index(state_idx)
                    if n_states == 2:
                        name = ["BEAR", "BULL"][rank]
                    elif n_states == 3:
                        name = ["BEAR", "NEUTRAL", "BULL"][rank]
                    elif n_states == 4:
                        name = ["STRONG_BEAR", "MILD_BEAR", "MILD_BULL", "STRONG_BULL"][rank]
                    else:
                        name = f"STATE_{rank}"
                    print(f"    {day_start.date()}: {name}")

print("\n\nDONE")
