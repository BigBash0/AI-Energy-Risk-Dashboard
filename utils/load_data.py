import sqlite3
import pandas as pd
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]

DB_PATH = PROJECT_DIR / "clean_energy_ai.db"
OUTPUT_DIR = PROJECT_DIR / "outputs"


def load_database():

    conn = sqlite3.connect(DB_PATH)

    energy = pd.read_sql(
        "SELECT * FROM clean_energy",
        conn
    )

    ai = pd.read_sql(
        "SELECT * FROM clean_ai_demand",
        conn
    )

    price = pd.read_sql(
        "SELECT * FROM clean_price",
        conn
    )

    conn.close()

    return energy, ai, price


def load_outputs():

    outputs = {}

    files = [
        "ai_energy_readiness_rankings.csv",
        "ai_energy_stress_test.csv",
        "electricity_forecasts.csv",
        "grid_stress_index.csv",
        "regional_ai_energy_risk_model.csv",
        "scenario_comparison.csv",
        "energy_transition_requirements.csv"
    ]

    for file in files:

        path = OUTPUT_DIR / file

        if path.exists():

            outputs[file] = pd.read_csv(path)

    return outputs