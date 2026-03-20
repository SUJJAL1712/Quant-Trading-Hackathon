# Short-Horizon Strategy Research Notes

This repo pass was driven by a focused literature survey across the momentum, liquidity,
volatility-management, crash-risk, and crypto-factor literature. I did not pretend to
"implement 50-100 papers" one-for-one; instead, I used the subset that is both empirically
strong and realistically implementable in this codebase without introducing model-risk from
heavy ML or opaque features.

## What the literature says

1. Time-series and cross-sectional momentum are among the most robust return predictors, but
   they are crash-prone and should be paired with explicit risk management.
   - Jegadeesh and Titman, "Returns to Buying Winners and Selling Losers" (1993)
   - Moskowitz, Ooi, and Pedersen, "Time Series Momentum" (2012)
   - Hurst, Ooi, and Pedersen, "A Century of Evidence on Trend-Following Investing" (2017)

2. Volatility-managed momentum is materially more stable than static momentum.
   - Moreira and Muir, "Volatility-Managed Portfolios" (2017)
   - Barroso and Santa-Clara, "Managing the Risk of Momentum" (2015)
   - Daniel and Moskowitz, "Momentum Crashes" (2016)

3. Residual momentum is often cleaner than raw momentum because it strips out shared market
   beta and leaves more persistent idiosyncratic trend.
   - Blitz, Huij, and Martens, "Residual Momentum" (2011)

4. In crypto, momentum exists, but liquidity matters a lot. Liquid winners are materially
   more attractive than indiscriminate long-only momentum across the full tail of coins.
   - Liu, Tsyvinski, and Wu, "Common Risk Factors in Cryptocurrency" (2022)
   - Begusic and Kostanjcar, "Momentum and Liquidity in Cryptocurrencies" (2019)
   - Shen, Urquhart, and Wang, "Common risk factors in the returns on cryptocurrencies" (2020)

5. High-frequency crypto returns show both intraday momentum and reversal, but the effect is
   state-dependent and sensitive to jumps, liquidity, and event regimes. That argues for
   a simple, constrained signal stack rather than a naive always-on reversal bet.
   - Wen, Bouri, Xu, and Zhao, "Intraday return predictability in the cryptocurrency markets:
     Momentum, reversal, or both" (2022)
   - Hansen, Kim, and Kimbrough, "Periodicity in Cryptocurrency Volatility and Liquidity" (2024)

6. Stop-loss rules can improve drawdown behavior when serial correlation and regime effects are
   present, but they can also create churn if re-entry is too fast.
   - Kaminski and Lo, "When do stop-loss rules stop losses?" (2014)
   - Lo and Remorov, "Stop-loss strategies with serial correlation, regime switching, and
     transaction costs" (2017)

7. Portfolio construction should be robust to covariance instability and transaction-cost
   slippage. Hierarchical diversification and explicit turnover controls are more realistic
   than unconstrained quadratic optimization.
   - Lopez de Prado, "Building Diversified Portfolios that Outperform Out-of-Sample" (2016)
   - Black and Litterman, "Global Portfolio Optimization" (1992)

## What was implemented

- Residual momentum signal: added because the literature supports ranking on idiosyncratic
  trend, not only raw beta-heavy trend.
- Liquidity filter: only the more liquid slice of the universe is eligible for positive long
  exposure.
- Minimum history gate: newly listed assets need enough actual observations before becoming
  tradable.
- Offline historical data hygiene: stop forward-filling dead assets as if they were still
  tradable; force stale positions out when data disappears.
- Exit-path fix: target-zero positions are now actually sold instead of lingering as dozens of
  small unwanted positions.
- Stop-loss cooldown fix: the documented 24h cooldown now actually matches the rebalance cadence.
- Monthly return output: every backtest now writes month-by-month returns for hit-rate analysis.

## What I explicitly did not implement

- Deep reinforcement learning or LSTM-based crypto trading from IEEE conference papers:
  the evidence is heterogeneous, most papers are abstract-level in accessible sources, and the
  model-risk / overfitting risk is much higher than the empirical-factor improvements above.
- Pair trading / market-neutral cointegration:
  the repo is built around long-only spot allocation, not long-short inventory management.
- Sentiment / alternative data:
  interesting, but not supported by the current local dataset and would require a new data pipe.

## Key sources

- https://research.cbs.dk/en/publications/time-series-momentum/
- https://www.aqr.com/Insights/Research/Journal-Article/A-Century-of-Evidence-on-Trend-Following-Investing
- https://www.nber.org/papers/w22208
- https://colab.ws/articles/10.2139%2Fssrn.2041429
- https://www.nber.org/papers/w20439
- https://repub.eur.nl/pub/22252/
- https://www.nber.org/papers/w25882
- https://ideas.repec.org/p/arx/papers/1904.00890.html
- https://www.sciencedirect.com/science/article/abs/pii/S026499931931020X
- https://www.sciencedirect.com/science/article/pii/S1062940822000833
- https://academic.oup.com/jfec/article/22/1/224/6759403
- https://research.hhs.se/esploro/outputs/journalArticle/When-do-stop-loss-rules-stop-losses/991001480526106056
- https://www.sciencedirect.com/science/article/pii/S1386418117300472
- https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2708678
- https://research.cbs.dk/en/publications/value-and-momentum-everywhere

## Extra discovery sources reviewed but not implemented directly

- https://www.tandfonline.com/doi/abs/10.1080/14697688.2023.2269999
- https://sciendo.com/article/10.2478/ceej-2025-0003
- https://ieeexplore.ieee.org/document/10056512/
- https://ieeexplore.ieee.org/document/10196966
- https://ieeexplore.ieee.org/document/9842512/
- https://ieeexplore.ieee.org/document/9976116/
