# %% [markdown]
# # Downstream Application — Model-Agnostic Trading Strategy
# Implements Sections 3.4–3.8 of the paper:
#   3.4 Daily Sentiment Score Generation
#   3.5 Portfolio Weight Construction
#   3.6 Market Data Loading
#   3.7 Strategy Execution
#   3.8 Performance Evaluation
#
# This file does NOT care about which model produced the sentiment predictions.
# It only requires a DataFrame with a 'sentiment_predictions' column (dict per row)
# and a 'Trading Date' column.
#
# Usage:
#   import downstream
#   df_news = downstream.load_news_data()
#   # ... run your model's predict_all() to add 'sentiment_predictions' column ...
#   downstream.run(df_news)

# %% [markdown]
# ## Imports

# %%
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

# %%
CURRENCY_CODES = ['USD', 'EUR', 'JPY', 'GBP', 'CAD', 'AUD', 'CHF', 'SEK', 'NOK', 'NZD']
ANNUALIZATION_FACTOR = 252    # Number of trading days per year

# %% [markdown]
# ## 3.2 Load News Data

# %%
def load_news_data():
    """Load and filter news data for the downstream application period (2020–2024).
    
    Returns:
        df_news: DataFrame with columns ['Title', 'Full Text', 'mentioned_currencies', 'Trading Date']
    """
    df_news = pd.read_pickle("_2_llm_paper/cache/df_news.pkl")

    df_news = df_news[
        (df_news['Date'] >= pd.to_datetime('2020-01-01')) &
        (df_news['Date'] < pd.to_datetime('2024-07-01'))
    ]

    df_news = df_news.reset_index(drop=True)
    df_news = df_news[['Title', 'Full Text', 'mentioned_currencies', 'Trading Date']]

    print(f"Loaded {len(df_news)} articles for downstream application (2020-01 to 2024-06)")
    return df_news


# %% [markdown]
# ## 3.4 Daily Sentiment Score Generation
# 
# 
# $$S_{i, t} = round(log(1+CountAppreciation_{i, t}) - log(1+CountDepreciation_{i, t}))$$
# 
# where $CountAppreciation_{i, t}$ is the number of articles published on day $t$ for which the model assigns the **future label** of currency $i$ to "appreciation"
# 
# $round$ function is defined below:
# $$
# \widehat{S}_{i,t} = 
# \begin{cases} 
# +1, & \text{if } S_{i,t} > 0, \\
# 0, & \text{if } S_{i,t} = 0, \\
# -1, & \text{if } S_{i,t} < 0.
# \end{cases}
# $$

# %%
def compute_daily_sentiment_scores(df_news):
    """Compute the daily rounded sentiment signal matrix S and aligned date array.

    Args:
        df_news: DataFrame with 'Trading Date' and 'sentiment_predictions' columns.

    Returns:
        S:                  np.ndarray of shape (n_currencies, n_dates) with values in {-1, 0, 1}
        unique_dates_array: DatetimeIndex of the trading dates, sorted
    """
    data_for_S = {}

    for date, group in df_news.groupby('Trading Date'):
        for currency in CURRENCY_CODES:
            count_appreciation = 0
            count_depreciation = 0

            for _, row in group.iterrows():
                future_key = f'{currency}_future'
                prediction = row['sentiment_predictions'].get(future_key, 'unchanged')

                if prediction == 'appreciation':
                    count_appreciation += 1
                elif prediction == 'depreciation':
                    count_depreciation += 1

            # S_{i,t} = log(1 + CountAppreciation) - log(1 + CountDepreciation)
            S_value = np.log(1 + count_appreciation) - np.log(1 + count_depreciation)

            # Round to sign
            if S_value > 0:
                S_value = 1
            elif S_value < 0:
                S_value = -1
            else:
                S_value = 0

            data_for_S[(currency, date)] = S_value

    dates = sorted(set(date for _, date in data_for_S.keys()))

    S = np.array([[data_for_S[(currency, date)] for date in dates] for currency in CURRENCY_CODES])

    unique_dates_array = np.sort(pd.to_datetime(df_news['Trading Date'].unique()))
    unique_dates_array = pd.to_datetime(unique_dates_array)
    unique_dates_array = unique_dates_array.tz_localize(None)
    unique_dates_array = unique_dates_array.normalize()

    return S, unique_dates_array


# %% [markdown]
# ## 3.5 Get Weights
# 
# $$
# w_{i,t:t+1} = 
# \begin{cases} 
# \frac{\widehat{S}_{i,t}}{\sum_{j:\widehat{S}_{j,t}>0} \widehat{S}_{j,t}}, & \text{if } \widehat{S}_{i,t} > 0 \text{ and } \sum_{j:\widehat{S}_{j,t}>0} \widehat{S}_{j,t} > 0 \\
# -\frac{\widehat{S}_{i,t}}{\sum_{j:\widehat{S}_{j,t}<0} |\widehat{S}_{j,t}|}, & \text{if } \widehat{S}_{i,t} < 0 \text{ and } \sum_{j:\widehat{S}_{j,t}<0} |\widehat{S}_{j,t}| > 0 \\
# 0, & \text{otherwise.}
# \end{cases}
# $$
# 
# Long on positive sentiment scores
# 
# Short negative sentiment scores
# 
# Value of each position is proportional to sentiment score 
# 
# Value of all short positions are the same as all the long positions for any given day
# 
# Held for 1 day (close of day t until close of day t+1)
# 
# if no new article was published on the next day, the previous sentiment signal is retained

# %%
def compute_portfolio_weights(S):
    """Compute zero-cost portfolio weights from the sentiment signal matrix.

    Args:
        S: np.ndarray of shape (n_currencies, n_dates) with values in {-1, 0, 1}

    Returns:
        weights: np.ndarray of same shape, with long weights summing to 1 and short to -1
    """
    weights = np.zeros_like(S, dtype=float)

    positive_mask = S > 0
    negative_mask = S < 0

    sum_positive = np.sum(S * positive_mask, axis=0, keepdims=True)
    sum_negative_abs = np.sum(np.abs(S) * negative_mask, axis=0, keepdims=True)

    valid_positive_sums = sum_positive > 0
    valid_negative_sums = sum_negative_abs > 0

    with np.errstate(divide='ignore', invalid='ignore'):
        positive_weights = np.where(valid_positive_sums, S / sum_positive, 0)
        negative_weights = np.where(valid_negative_sums, S / sum_negative_abs, 0)

    weights = np.where(positive_mask & valid_positive_sums, positive_weights, weights)
    weights = np.where(negative_mask & valid_negative_sums, negative_weights, weights)

    return weights


# %% [markdown]
# ## 3.6 Get the Market Data

# %%
def load_market_data(unique_dates_array):
    """Load FX log returns and filter to the dates present in the news data.

    Args:
        unique_dates_array: DatetimeIndex of trading dates

    Returns:
        log_returns: np.ndarray of shape (n_currencies, n_dates)
    """
    df_log_returns = pd.read_pickle("_2_llm_paper/cache/df_log_returns.pkl")
    df_log_returns = df_log_returns[df_log_returns.index.isin(unique_dates_array)]

    print(f"Market data shape: {df_log_returns.shape}, news dates: {len(unique_dates_array)}")

    log_returns = df_log_returns.transpose().values
    return log_returns


# %% [markdown]
# ## 3.7 Execute Strategy

# %%
def execute_strategy(weights, log_returns):
    """Compute daily P&L from weights and next-day returns.

    Args:
        weights:     np.ndarray of shape (n_currencies, n_dates)
        log_returns: np.ndarray of shape (n_currencies, n_dates)

    Returns:
        daily_pnl: np.ndarray of shape (n_dates - 1,)
    """
    # Remove the last day (no future returns available)
    W_active = weights[:, :-1]

    # Align for future returns
    R_future = log_returns[:, 1:]

    # Element-wise multiplication, sum across currencies for each day
    daily_pnl = np.sum(W_active * R_future, axis=0)

    return daily_pnl


# %% [markdown]
# ## 3.8 Evaluate Results
# 
# Assume a zero cost portfolio so risk free rate is 0 
# 
# $$Annualized\ Return = mean\ daily\ return \times 252$$
# 
# $$Annualized\ Volatility = Standard\ Deviation\ of\ Daily\ Returns \times \sqrt{252}$$
# 
# $$Sharpe\ Ratio = \frac{Annualized\ Return}{Annualized\ Volatility}$$
# 
# $$Maximum\ Drawdown =\ The\ largest\ peak\ to\ trough\ decline\ in\ cumulative\ returns$$
# 
# $$Transaction\ Costs = sum\ of\ absolute\ weight\ changes\ per\ day,\ averaged,\ then\ annualized$$

# %%
def evaluate_results(daily_pnl, weights):
    """Compute and print performance metrics for the trading strategy.

    Args:
        daily_pnl: np.ndarray of daily portfolio returns
        weights:   np.ndarray of shape (n_currencies, n_dates) for turnover calculation

    Returns:
        dict with keys: annualized_return_pc, annualized_volatility_pc,
                        sharpe_ratio, max_drawdown_pc, rebalancing_frequency
    """
    # Annualized Return
    ann_return = np.mean(daily_pnl) * ANNUALIZATION_FACTOR
    ann_return_pc = ann_return * 100

    # Annualized Volatility
    ann_vol = np.std(daily_pnl, ddof=1) * np.sqrt(ANNUALIZATION_FACTOR)
    ann_vol_pc = ann_vol * 100

    # Sharpe Ratio
    if ann_vol != 0:
        sharpe_ratio = ann_return / ann_vol
    else:
        sharpe_ratio = 0.0

    # Maximum Drawdown
    cumulative_returns = np.cumsum(daily_pnl)
    running_max = np.maximum.accumulate(cumulative_returns)
    drawdown = running_max - cumulative_returns
    max_drawdown = np.max(drawdown)
    max_drawdown_pc = max_drawdown * 100

    # Transaction Costs (Rebalancing Frequency)
    weight_changes = np.abs(np.diff(weights, axis=1))
    daily_turnover = np.sum(weight_changes, axis=0)
    rebalancing_frequency = np.mean(daily_turnover) * ANNUALIZATION_FACTOR

    metrics = {
        'annualized_return_pc': ann_return_pc,
        'annualized_volatility_pc': ann_vol_pc,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown_pc': max_drawdown_pc,
        'rebalancing_frequency': rebalancing_frequency,
    }

    print(f"Annualized Return:     {ann_return_pc:.2f}%")
    print(f"Annualized Volatility: {ann_vol_pc:.2f}%")
    print(f"Sharpe Ratio:          {sharpe_ratio:.2f}")
    print(f"Max Drawdown:          {max_drawdown_pc:.2f}%")
    print(f"Rebalancing Freq:      {rebalancing_frequency:.2f}")

    return metrics


# %% [markdown]
# ## Run Full Pipeline

# %%
def run(df_news):
    """Run the full downstream trading strategy pipeline.

    Expects df_news to already contain a 'sentiment_predictions' column
    (dict mapping e.g. 'USD_future' -> 'appreciation') and a 'Trading Date' column.

    Args:
        df_news: DataFrame with columns ['sentiment_predictions', 'Trading Date']

    Returns:
        dict of performance metrics
    """
    print("=== Downstream Application: Trading Strategy ===")
    print()

    # 3.4 Daily Sentiment Score Generation
    print("Computing daily sentiment scores...")
    S, unique_dates_array = compute_daily_sentiment_scores(df_news)

    # 3.5 Portfolio Weights
    print("Computing portfolio weights...")
    weights = compute_portfolio_weights(S)

    # 3.6 Market Data
    print("Loading market data...")
    log_returns = load_market_data(unique_dates_array)

    # 3.7 Execute Strategy
    print("Executing strategy...")
    daily_pnl = execute_strategy(weights, log_returns)

    # 3.8 Evaluate Results
    print()
    print("--- Performance Metrics ---")
    metrics = evaluate_results(daily_pnl, weights)

    return metrics


# %% [markdown]
# ## Model Configuration

# %%
get_model_id = {
    'a': "meta-llama/Meta-Llama-3.1-8B-Instruct",
    'b': "meta-llama/Meta-Llama-3-70B-Instruct",

    'c': "meta-llama/Meta-Llama-3.1-8B",
    'd': "meta-llama/Meta-Llama-3-70B",

    'e': "meta-llama/Meta-Llama-4-Maverick-8B-Instruct",
    'f': "meta-llama/Meta-Llama-4-Scout-8B-Instruct",

    'g': "mistralai/Mistral-Nemo-Instruct",

    'h': "Qwen/Qwen3-32B-Chat",
}

# %% [markdown]
# ## Backtest: Finetuned Llama (vLLM)

# %%
def run_finetuned_llama(model_key='a', adapter_dir='finetuned_llama_8b'):
    """End-to-end backtest for the finetuned Llama model using vLLM.

    Args:
        model_key:   letter key into get_model_id (default 'a' = Llama-3.1-8B-Instruct)
        adapter_dir: name of the adapter directory under _2_llm_paper/models/

    Returns:
        dict of performance metrics
    """
    import models_code.finetuned_llama as finetuned_llama

    model_id = get_model_id[model_key]
    print(f"=== Finetuned Llama Backtest ===")
    print(f"Model: {model_id}")
    print(f"Adapter: {adapter_dir}")
    print()

    # Load model via vLLM
    llm, tokenizer, lora_request = finetuned_llama.load_vllm(model_id, adapter_dir)

    # Load news data
    df_news = load_news_data()

    # Predict sentiments
    df_news = finetuned_llama.predict_all(llm, tokenizer, df_news, lora_request=lora_request)

    # Run trading strategy
    metrics = run(df_news)

    return metrics


# %% [markdown]
# ## Backtest: LM Dictionary

# %%
def run_lm_dictionary(dict_path=None):
    """End-to-end backtest for the Loughran-McDonald dictionary method.

    Args:
        dict_path: optional path to LM master dictionary CSV

    Returns:
        dict of performance metrics
    """
    import models_code.lm_dictionary as lm_dictionary

    print(f"=== LM Dictionary Backtest ===")
    print()

    # Load dictionary
    positive_words, negative_words = lm_dictionary.setup(dict_path)

    # Load news data
    df_news = load_news_data()

    # Predict sentiments
    df_news = lm_dictionary.predict_all(positive_words, negative_words, df_news)

    # Run trading strategy
    metrics = run(df_news)

    return metrics
