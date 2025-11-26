import pandas as pd
import plotly.graph_objects as go

# --- FILE PATHS ---
SP500_PATH = "data/benchmark/sp500.csv"
NDCG_PATH = "backtest/backtest_ndcg.csv"

# --- LOAD DATA ---
sp500 = pd.read_csv(SP500_PATH)
ndcg = pd.read_csv(NDCG_PATH)

# --- CLEAN & PREPARE ---
sp500["date"] = pd.to_datetime(sp500["date"], errors="coerce")
ndcg["date"] = pd.to_datetime(ndcg["date"], errors="coerce")

# Select only available columns; include longer horizons if present
_cols = [
    'date', 'sp500_return_date', 'sp500_return',
    'mean_return_5', 'mean_return_10', 'mean_return_20',
    'mean_return_100', 'mean_return_500'
]
ndcg = ndcg[[c for c in _cols if c in ndcg.columns]]


# Merge datasets by nearest date
merged = pd.merge(
    sp500, ndcg,
    on="date",
    how="outer"
)
merged = merged[merged["date"]>=pd.to_datetime("1990-01-01")]
merged.to_csv("merged_sp500_ndcg.csv", index=False)

# --- PLOTLY FIGURE ---
fig = go.Figure()


# Draw line segments from (date, close) to (sp500_return_date, close * mean_return_10)
# using None separators so Plotly renders separate segments in a single trace
# Build segments for k in {5, 10, 20, 100, 500}
x_segments_5, y_segments_5 = [], []
x_segments_10, y_segments_10 = [], []
x_segments_20, y_segments_20 = [], []
x_segments_100, y_segments_100 = [], []
x_segments_500, y_segments_500 = [], []
for row in merged.itertuples(index=False):
    # Ensure numeric types for safe multiplication
    close_val = pd.to_numeric(row.close, errors="coerce")
    mr5 = pd.to_numeric(getattr(row, "mean_return_5", float("nan")), errors="coerce")
    mr10 = pd.to_numeric(getattr(row, "mean_return_10", float("nan")), errors="coerce")
    mr20 = pd.to_numeric(getattr(row, "mean_return_20", float("nan")), errors="coerce")
    mr100 = pd.to_numeric(getattr(row, "mean_return_100", float("nan")), errors="coerce")
    mr500 = pd.to_numeric(getattr(row, "mean_return_500", float("nan")), errors="coerce")
    if pd.notna(row.sp500_return_date) and pd.notna(close_val):
        if pd.notna(mr5):
            x_segments_5.extend([row.date, row.sp500_return_date, None])
            y_segments_5.extend([close_val, close_val * mr5, None])
        if pd.notna(mr10):
            x_segments_10.extend([row.date, row.sp500_return_date, None])
            y_segments_10.extend([close_val, close_val * mr10, None])
        if pd.notna(mr20):
            x_segments_20.extend([row.date, row.sp500_return_date, None])
            y_segments_20.extend([close_val, close_val * mr20, None])
        if pd.notna(mr100):
            x_segments_100.extend([row.date, row.sp500_return_date, None])
            y_segments_100.extend([close_val, close_val * mr100, None])
        if pd.notna(mr500):
            x_segments_500.extend([row.date, row.sp500_return_date, None])
            y_segments_500.extend([close_val, close_val * mr500, None])

# Add traces for each k
fig.add_trace(go.Scatter(
    x=x_segments_5,
    y=y_segments_5,
    mode="lines",
    name="Close → Close * mean_return_5",
    line=dict(color="red", width=0.1),
    yaxis="y1",
    hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Price: %{y:.2f}<extra></extra>"
))

fig.add_trace(go.Scatter(
    x=x_segments_10,
    y=y_segments_10,
    mode="lines",
    name="Close → Close * mean_return_10",
    line=dict(color="green", width=0.1),
    yaxis="y1",
    hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Price: %{y:.2f}<extra></extra>"
))
"""
fig.add_trace(go.Scatter(
    x=x_segments_20,
    y=y_segments_20,
    mode="lines",
    name="Close → Close * mean_return_20",
    line=dict(color="green", width=0.1),
    yaxis="y1",
    hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Price: %{y:.2f}<extra></extra>"
))

fig.add_trace(go.Scatter(
    x=x_segments_100,
    y=y_segments_100,
    mode="lines",
    name="Close → Close * mean_return_100",
    line=dict(color="red", width=0.1),
    yaxis="y1",
    hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Price: %{y:.2f}<extra></extra>"
))

fig.add_trace(go.Scatter(
    x=x_segments_500,
    y=y_segments_500,
    mode="lines",
    name="Close → Close * mean_return_500",
    line=dict(color="cyan", width=0.1),
    yaxis="y1",
    hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Price: %{y:.2f}<extra></extra>"
))
"""

# S&P 500 Close (left y-axis)
fig.add_trace(go.Scatter(
    x=merged["date"],
    y=merged["close"],
    name="S&P 500 Close",
    line=dict(color="blue", width=1),
    yaxis="y1"
))
# --- LAYOUT ---
fig.update_layout(
    title="S&P 500 Closing Price vs. 1Y Forward NDCG@10",
    xaxis=dict(title="Date"),
    yaxis=dict(
        title=dict(text="S&P 500 Close", font=dict(color="blue")),
        tickfont=dict(color="blue")
    ),
    yaxis2=dict(
        title=dict(text="NDCG@10", font=dict(color="orange")),
        tickfont=dict(color="orange"),
        overlaying="y",
        side="right"
    ),
    legend=dict(x=0.01, y=0.99),
    template="plotly_white"
)

fig.show()

