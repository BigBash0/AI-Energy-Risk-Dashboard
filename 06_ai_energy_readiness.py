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


def calculate_readiness_score(df):

    latest_year = df["year"].max()

    latest = df[
        df["year"] == latest_year
    ].copy()

    required = [
        "country",
        "renewables_share_elec",
        "low_carbon_share_elec",
        "fossil_share_elec",
        "carbon_intensity_elec",
        "electricity_generation"
    ]

    latest = latest[required].dropna()

    latest["carbon_score"] = (
        100 -
        (
            latest["carbon_intensity_elec"]
            /
            latest["carbon_intensity_elec"].max()
            * 100
        )
    )

    latest["generation_score"] = (
        latest["electricity_generation"]
        /
        latest["electricity_generation"].max()
        * 100
    )

    latest["ai_readiness_score"] = (
        latest["renewables_share_elec"] * 0.25 +
        latest["low_carbon_share_elec"] * 0.30 +
        latest["carbon_score"] * 0.25 +
        latest["generation_score"] * 0.20
    )

    latest["ai_readiness_score"] = (
        latest["ai_readiness_score"]
        .round(2)
    )

    return latest.sort_values(
        "ai_readiness_score",
        ascending=False
    )


def main():

    energy = load_data()

    rankings = calculate_readiness_score(
        energy
    )

    print(rankings.head(20))

    rankings.to_csv(
        OUTPUT_DIR /
        "ai_energy_readiness_rankings.csv",
        index=False
    )

    print(
        "\nSaved: ai_energy_readiness_rankings.csv"
    )


if __name__ == "__main__":
    main()