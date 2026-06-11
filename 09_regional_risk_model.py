import sqlite3
import pandas as pd
import numpy as np
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


def prepare_latest_data(df):

    latest_year = df["year"].max()

    latest = df[
        df["year"] == latest_year
    ].copy()

    exclude_patterns = (
        "World|Ember|OECD|G20|G7|EU"
    )

    latest = latest[
        ~latest["country"].str.contains(
            exclude_patterns,
            case=False,
            na=False
        )
    ]

    return latest


def build_risk_scores(df):

    required = [
        "country",
        "electricity_generation",
        "electricity_demand",
        "renewables_share_elec",
        "fossil_share_elec",
        "carbon_intensity_elec"
    ]

    df = df[required].copy()

    df = df.dropna()

    max_generation = df[
        "electricity_generation"
    ].max()

    max_carbon = df[
        "carbon_intensity_elec"
    ].max()

    max_demand = df[
        "electricity_demand"
    ].max()

    df["generation_score"] = (
        df["electricity_generation"]
        /
        max_generation
        * 100
    )

    df["demand_score"] = (
        df["electricity_demand"]
        /
        max_demand
        * 100
    )

    df["carbon_score"] = (
        df["carbon_intensity_elec"]
        /
        max_carbon
        * 100
    )

    df["sustainability_score"] = (
        df["renewables_share_elec"] * 0.50
        +
        (100 - df["fossil_share_elec"]) * 0.50
    )

    df["risk_score"] = (
        df["carbon_score"] * 0.35
        +
        df["fossil_share_elec"] * 0.25
        +
        df["demand_score"] * 0.20
        -
        df["generation_score"] * 0.10
        -
        df["sustainability_score"] * 0.10
    )

    df["risk_score"] = (
        df["risk_score"]
        .round(2)
    )

    return df


def classify_risk(df):

    conditions = [
        df["risk_score"] < 20,
        df["risk_score"].between(20, 40),
        df["risk_score"].between(40, 60),
        df["risk_score"] > 60
    ]

    labels = [
        "Low Risk",
        "Moderate Risk",
        "High Risk",
        "Critical Risk"
    ]

    df["risk_level"] = np.select(
        conditions,
        labels,
        default="Unknown"
    )

    return df


def save_results(df):

    output_file = (
        OUTPUT_DIR /
        "regional_ai_energy_risk_model.csv"
    )

    df.to_csv(
        output_file,
        index=False
    )

    print(
        f"\nSaved: {output_file}"
    )


def print_summary(df):

    print("\n" + "=" * 70)
    print("TOP 20 LOWEST RISK COUNTRIES")
    print("=" * 70)

    print(
        df.sort_values(
            "risk_score"
        )
        [
            [
                "country",
                "risk_score",
                "risk_level",
                "renewables_share_elec",
                "fossil_share_elec",
                "carbon_intensity_elec"
            ]
        ]
        .head(20)
    )

    print("\n" + "=" * 70)
    print("TOP 20 HIGHEST RISK COUNTRIES")
    print("=" * 70)

    print(
        df.sort_values(
            "risk_score",
            ascending=False
        )
        [
            [
                "country",
                "risk_score",
                "risk_level",
                "renewables_share_elec",
                "fossil_share_elec",
                "carbon_intensity_elec"
            ]
        ]
        .head(20)
    )

    print("\nRisk Distribution")

    print(
        df["risk_level"]
        .value_counts()
    )


def main():

    energy = load_data()

    latest = prepare_latest_data(
        energy
    )

    risk = build_risk_scores(
        latest
    )

    risk = classify_risk(
        risk
    )

    print_summary(
        risk
    )

    save_results(
        risk
    )


if __name__ == "__main__":
    main()