import sqlite3
import pandas as pd
from pathlib import Path

PROJECT_DIR = Path(r"C:\Users\DELL\OneDrive\Documents\AI_Energy_Project")
DB_PATH = PROJECT_DIR / "clean_energy_ai.db"
OUTPUT_DIR = PROJECT_DIR / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)

conn = sqlite3.connect(DB_PATH)

energy = pd.read_sql(
    "SELECT * FROM clean_energy",
    conn
)

ai = pd.read_sql(
    "SELECT * FROM clean_ai_demand",
    conn
)

conn.close()

world = energy[
    energy["country"]=="World"
].copy()

world = world.sort_values("year")

latest = world.iloc[-1]

growth_rate = (
    world["electricity_demand"]
    .pct_change()
    .tail(10)
    .mean()
)

forecast_rows = []

for year in [2030,2035]:

    years_forward = year - int(latest["year"])

    forecast = (
        latest["electricity_demand"]
        * ((1 + growth_rate) ** years_forward)
    )

    forecast_rows.append({
        "year":year,
        "forecast_demand":forecast
    })

forecast_df = pd.DataFrame(forecast_rows)

ai = ai[
    (ai["metric_category"].str.lower()=="electricity consumption (twh)")
    &
    (ai["segment"]=="Total")
]

comparison = ai.merge(
    forecast_df,
    on="year",
    how="left"
)

comparison = comparison[
    comparison["scenario"] != "Historical"
]

comparison["ai_share_pct"] = (
    comparison["value"]
    /
    comparison["forecast_demand"]
    * 100
)

comparison["stress_level"] = comparison["ai_share_pct"].apply(
    lambda x:
    "LOW" if x < 3
    else "MODERATE" if x < 5
    else "HIGH"
)

comparison = comparison[[
    "scenario",
    "year",
    "value",
    "forecast_demand",
    "ai_share_pct",
    "stress_level"
]]

comparison.columns = [
    "scenario",
    "year",
    "ai_demand_twh",
    "forecast_global_demand_twh",
    "ai_share_pct",
    "stress_level"
]

print(comparison)

output_file = OUTPUT_DIR / "scenario_comparison.csv"

comparison.to_csv(output_file,index=False)

print(f"\nSaved: {output_file}")