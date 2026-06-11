import sqlite3
import pandas as pd
from pathlib import Path

PROJECT_DIR = Path(r"C:\Users\DELL\OneDrive\Documents\AI_Energy_Project")
DB_PATH = PROJECT_DIR / "clean_energy_ai.db"
OUTPUT_DIR = PROJECT_DIR / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)


def load_data():

    conn = sqlite3.connect(DB_PATH)

    energy = pd.read_sql(
        "SELECT * FROM clean_energy",
        conn
    )

    conn.close()

    return energy


def calculate_grid_stress(df):

    latest_year = df["year"].max()

    latest = df[
        df["year"] == latest_year
    ].copy()

    latest = latest.dropna(
        subset=[
            "fossil_share_elec",
            "carbon_intensity_elec",
            "electricity_demand"
        ]
    )

    latest["demand_score"] = (
        latest["electricity_demand"]
        /
        latest["electricity_demand"].max()
        * 100
    )

    latest["carbon_score"] = (
        latest["carbon_intensity_elec"]
        /
        latest["carbon_intensity_elec"].max()
        * 100
    )

    latest["grid_stress_index"] = (
        latest["demand_score"] * 0.40 +
        latest["carbon_score"] * 0.30 +
        latest["fossil_share_elec"] * 0.30
    )

    latest["grid_stress_index"] = (
        latest["grid_stress_index"]
        .round(2)
    )

    latest["risk_level"] = pd.cut(
        latest["grid_stress_index"],
        bins=[0,40,60,80,100],
        labels=[
            "Low",
            "Moderate",
            "High",
            "Critical"
        ]
    )

    return latest.sort_values(
        "grid_stress_index",
        ascending=False
    )


def main():

    energy = load_data()

    stress = calculate_grid_stress(
        energy
    )

    print(stress.head(20))

    stress.to_csv(
        OUTPUT_DIR /
        "grid_stress_index.csv",
        index=False
    )

    print(
        "\nSaved: grid_stress_index.csv"
    )


if __name__ == "__main__":
    main()