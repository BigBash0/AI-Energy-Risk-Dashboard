import sqlite3
import pandas as pd
from pathlib import Path

PROJECT_DIR = Path(
    r"C:\Users\DELL\OneDrive\Documents\AI_Energy_Project"
)

DB_PATH = PROJECT_DIR / "clean_energy_ai.db"
OUTPUT_DIR = PROJECT_DIR / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)

SOLAR_CAPACITY_FACTOR = 0.20
WIND_CAPACITY_FACTOR = 0.35
HOURS_PER_YEAR = 8760

conn = sqlite3.connect(DB_PATH)

ai = pd.read_sql(
    "SELECT * FROM clean_ai_demand",
    conn
)

conn.close()

ai = ai[
    (ai["metric_category"].str.lower() == "electricity consumption (twh)")
    &
    (ai["segment"] == "Total")
]

results = []

for _, row in ai.iterrows():

    demand_twh = row["value"]

    solar_gw = (
        demand_twh * 1000
    ) / (
        HOURS_PER_YEAR * SOLAR_CAPACITY_FACTOR
    )

    wind_gw = (
        demand_twh * 1000
    ) / (
        HOURS_PER_YEAR * WIND_CAPACITY_FACTOR
    )

    results.append({
        "scenario": row["scenario"],
        "year": row["year"],
        "ai_demand_twh": round(demand_twh, 2),
        "solar_capacity_needed_gw": round(solar_gw, 2),
        "wind_capacity_needed_gw": round(wind_gw, 2)
    })

transition = pd.DataFrame(results)

print(transition)

output_file = OUTPUT_DIR / "energy_transition_requirements.csv"

transition.to_csv(
    output_file,
    index=False
)

print(f"\nSaved: {output_file}")