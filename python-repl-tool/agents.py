import os
import json
import asyncio
import pandas as pd
from langchain_experimental.tools import PythonAstREPLTool
from autogen_ext.tools.langchain import LangChainToolAdapter
from autogen_core import CancellationToken
from autogen_agentchat.messages import TextMessage
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import Swarm,RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination,MaxMessageTermination
from autogen_agentchat.ui import Console
from autogen_ext.models.ollama import OllamaChatCompletionClient

from tools import *

# ------------- Utility: Load CSV robustly -------------
def load_financial_csv(path_or_url: str) -> pd.DataFrame:
    df = pd.read_csv(path_or_url)
    # Normalize column names
    df.columns = [c.strip().lower() for c in df.columns]
    # Detect a datetime column
    ts_col = None
    for cand in ["datetime", "date", "timestamp", "time"]:
        if cand in df.columns:
            ts_col = cand
            break
    if ts_col is None:
        raise ValueError("No datetime/date column detected. Expected one of: datetime, date, timestamp, time")

    # Parse datetime and sort
    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce", utc=True)
    df = df.dropna(subset=[ts_col]).sort_values(ts_col)

    # Set index for indicators that expect DatetimeIndex (e.g., VWAP)
    df = df.set_index(ts_col)

    return df

# ------------- Bootstrap a base DataFrame (empty by default) -------------
df = load_financial_csv('./finance_economics_dataset.csv')

# ------------- Bind a safe tool environment -------------
# Only expose df and functions needed; avoid os/system access for safety.
tool_locals = {
    "pd": pd,
    "df": df
    }

py_repl = PythonAstREPLTool(locals=tool_locals)
py_tool = LangChainToolAdapter(py_repl)

model_client = OllamaChatCompletionClient(
    model="qwen3:8b",
    host="http://localhost:11434"
    )

system_msg = """You are a CSV Financial Analyst.
You have one tool: a restricted Python REPL with Pandas bound as `pd`, a global `df` DataFrame with the financial data loaded`.
Rules:
- Prefer writing concise Python that mutates or replaces the global `df`.
- Valid tasks:
* Validate schema and basic stats: df.info(), df.describe()
* Compute columns: returns, rolling metrics, and technical indicators.
* Create clean outputs: df[['close', 'rsi_14', ...]].tail(10).to_csv("out.csv", index=True)
* Generate summary dicts/JSON and print them.
- Do NOT install packages, access filesystem outside local folder, or use network except reading provided CSV URLs.
- If no OHLCV columns exist, explain what is missing and show how to map/rename.
- Keep outputs deterministic and reproducible. Use clear column names with suffixes (e.g., rsi_14).
"""

analyst = AssistantAgent(
    name="csv_fin_analyst",
    model_client=model_client,
    tools=[py_tool],
    system_message=system_msg,
)

# ------------- Optional: small team wrapper (single-turn fine) -------------
team = RoundRobinGroupChat([analyst], max_turns=4)

# ------------- Helper: run a task -------------
async def run_task(instruction: str):
    convo = [TextMessage(content=instruction, source="user")]
    result = Console(team.run_stream(task=convo))  # streams to console
    return await result

if __name__ == "__main__":
    # Example task: replace the path with an actual CSV file or URL
    instruction = """
Task: Load a CSV of OHLCV, compute analytics, and export tidy outputs.

Steps:
1) Load data into df using load_financial_csv("data/ohlcv.csv").
2) Confirm columns, show last 3 rows.
3) Compute:
   - Log and percent returns (cumulative and daily).
   - RSI(14), EMA(20), SMA(50), MACD(12,26,9), Bollinger Bands(20,2).
   - VWAP if high/low/close/volume present.
4) Create a 'signal' column:
   - RSI < 30 and close > ema_20 => 'oversold_bounce'
   - RSI > 70 and close < ema_20 => 'overbought_fade'
   - else 'neutral'
5) Save:
   - metrics.csv with close, returns, key indicators, signal.
   - profile.json with basic summary stats and last signal snapshot.

Return: A short printed JSON summary with last date, last close, last RSI, last MACD histogram, and last signal.
"""
    asyncio.run(run_task(instruction))
