# Problem Statement – SG vs HK University Web3 Quant Hackathon

## **Problem Statement:** AI Web3 Trading Bot Competition

### **Description:**

- Develop an AI-driven trading algorithm, or a traditional quantitative rule-based algorithm, or any creative hybrid strategies to compete on Roostoo’s real-time mock exchange backend.
- Using the APIs provided in the [Roostoo API Documents](https://github.com/roostoo/Roostoo-API-Documents), your task is to design a trading bot that autonomously makes buy, hold, and sell decisions—without any manual intervention—by interacting with the Roostoo backend exchange engine via POST and GET API requests.
- The objective is to maximize portfolio returns while minimizing risk, as measured by portfolio return, Sortino Ratio, Sharpe Ratio, and Calmar Ratio.

This challenge is a unique opportunity to showcase your algorithmic trading and AI skills in an intense, real-time, competitive environment of Web3 crypto markets. 

### Requirements

- Strategies and bot usage are open-ended; there are no limitations or predefined rules. You may use any approach, including LLM models, reinforcement learning algorithms such as PPO agents, traditional trading strategies, or even your own custom solutions built from scratch.
- You are welcome to use any data sources. Roostoo platform data is also freely available via API GET requests, as detailed in the documentation. Please note that Roostoo will only cover cloud server costs; any additional data source costs (such as LLM API calls) are not covered.
- Roostoo will display bot names on the Roostoo app, creating a live competition leaderboard among teams.
    - You can track your bot’s performance on the competition leaderboard through our products:
        - iOS: https://apps.apple.com/us/app/roostoo-mock-crypto-trading/id1483561353
        - Android: https://play.google.com/store/apps/details?id=com.roostoo.roostoo&hl=en
        - Webapp: [app.roostoo.com](http://app.roostoo.com/)
    - Note: You will not have access to your competition account via the frontend, to prevent manual trades that could compromise the integrity of the competition.
    - We strongly recommend that you also record all trades and performance logs of your bot internally, and keep track of the success or failure status of each API request.
- The competition will run for **on AWS cloud infrastructure** (provisioned by Roostoo) during the hackathon.
    - You are required to deploy your bot on an AWS VM and ensure it executes trades automatically on the Roostoo platform.

### **Timeline:**

- **Mar 13, 3pm HKT:** Online Info Session & Workshop.
- **Mar 16 – Mar 20:** Hackathon Begins. Preparation Period — Build your bot and test deployment on Roostoo.
- **Mar 21 – Mar 31:** 1st Round – City Qualifiers Live trading begins.
    - During these 10 days, you are allowed to continue iterate your strategies and redeploy your bots.
    - Each bot must have at least 8 active trading days with enough trades made from strategies each day.
    - Submission of open-source repositories with ReadMe for judging.
- **Before Mar 28:** Submit your repo link
- **Apr 2:** Top 16 Finalists announced; 8 from each city.
    - Teams within the same city are allowed to communicate, collaborate and coordinate strategies
- **Apr 4 – Apr 14**: 2nd Round – Singapore vs Hong Kong Live trading begins.
    - Same rules applied as 1st Round.
- **Apr 17:** Final Submission Deadline (presentation decks).
- **Apr 17 – Apr 21:** Grand Finale at both SG and HK — Top 8 Teams Demo presentations to industry judges and awards ceremony.

### **Preliminary Round – Evaluation Criteria (In Order):**

Evaluation for finalist teams will be conducted in the following stages:

**Screen 1: Rule Compliance (Mandatory Requirement)**

Teams that violate any of the following rules will not be selected as finalists:

1. **Trade Log Integrity**
    
    Bots must demonstrate consistent, autonomous trade execution aligned with their declared strategy.
    
2. **Commit History Transparency**
    
    All strategy updates must have a consistent and traceable commit history. No traces of manually called APIs. 
    

**Screen 2: Portfolio Returns (Leaderboard Qualification)**

- After screen 1 disqualification, the following **Top 20 teams on leaderboard from each region**, ranked by portfolio return, will advance for further evaluation.
- Portfolio Return is calculated as: (Final Portfolio Value – Initial Portfolio Value) / Initial Portfolio Value

This screen determines qualification only and does not carry additional weighting beyond leaderboard ranking.

**Screen 3: Composite Risk‑Adjusted Performance Score (40%)**

Qualified teams will be evaluated using a composite risk-adjusted performance score based on three key metrics.

All underlying data, calculations, and results will be transparently published on the Finale Day.

**Composite Score Formula:**

- **0.4 × Sortino Ratio**
    
    Measures **return per unit of downside risk** (penalizes only negative volatility).
    
    $$
    \text{Sortino Ratio} = \frac{\overline{R_p}}{\sigma_d}
    $$
    
    where 
    
    $$
    \overline{R_p}  = \text{Mean of Portfolio Returns,} \\
    \sigma_d = \text{Standard Deviation of Negative Portfolio Returns}
    $$
    
- **0.3 × Sharpe Ratio**
    
    Measures **excess return per unit of total volatility**.
    
    $$
    \text{Sharpe Ratio} = \frac{\overline{R_p}}{\sigma_p}
    $$
    
    where 
    
    $$
     \sigma_p  = \text{Standard Deviation of Portfolio Returns}\;
    $$
    
- **0.3 × Calmar Ratio**
    
    Measures **return relative to maximum drawdown**, emphasizing capital preservation.
    
    $$
    \text{Calmar Ratio} = \frac{\overline{R_p}}{|\text{Max Drawdown}|}
    $$
    
    where 
    
    $$
    \text{Max Drawdown} = \text{Largest Portfolio peak-to-trough decline}
    $$
    

**Screen 4: Code & Strategy Review (60%)**

Each shortlisted team’s repository and strategy implementation will undergo technical review.

Evaluation criteria include:

1. Clear and coherent implementation of trading strategy logic (30%) 
2. Clean, well-structured, and properly maintained code repository (20%)
3. Fully runnable continuously with compatibility on the Roostoo platform (10%) 

**Top 8 teams from each region are selected as finalists from the above screening criteria.**