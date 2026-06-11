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

    ai = pd.read_sql(
        "SELECT * FROM clean_ai_demand",
        conn
    )

    conn.close()

    return energy, ai


def calculate_cagr(start_value, end_value, periods):
    if start_value <= 0 or end_value <= 0:
        return np.nan

    return (end_value / start_value) ** (1 / periods) - 1


def forecast_world_electricity(energy):

    world = (
        energy[
            energy["country"] == "World"
        ]
        .copy()
        .sort_values("year")
    )

    history = world[
        ["year", "electricity_demand"]
    ].dropna()

    recent = history.tail(10)

    start = recent.iloc[0]
    end = recent.iloc[-1]

    cagr = calculate_cagr(
        start["electricity_demand"],
        end["electricity_demand"],
        int(end["year"] - start["year"])
    )

    forecasts = []

    for year in range(
        int(end["year"]) + 1,
        2036
    ):
        demand = end["electricity_demand"] * (
            (1 + cagr) **
            (year - int(end["year"]))
        )

        forecasts.append({
            "year": year,
            "forecast_world_demand_twh": demand
        })

    return pd.DataFrame(forecasts)


def get_ai_scenarios(ai):

    ai_power = ai[
        (
            ai["metric_category"]
            .str.lower()
            ==
            "electricity consumption (twh)"
        )
        &
        (
            ai["segment"] == "Total"
        )
    ].copy()

    return ai_power


def build_stress_results(world_forecast, ai_power):

    merged = pd.merge(
        ai_power,
        world_forecast,
        on="year",
        how="inner"
    )

    merged["ai_share_pct"] = (
        merged["value"]
        /
        merged["forecast_world_demand_twh"]
        * 100
    )

    conditions = [
        merged["ai_share_pct"] < 3,
        merged["ai_share_pct"].between(3, 5),
        merged["ai_share_pct"] > 5
    ]

    labels = [
        "LOW STRESS",
        "MODERATE STRESS",
        "HIGH STRESS"
    ]

    merged["stress_level"] = np.select(
        conditions,
        labels,
        default="UNKNOWN"
    )

    return merged


def print_summary(results):

    print("\n" + "=" * 70)
    print("AI ENERGY SYSTEM STRESS TEST")
    print("=" * 70)

    summary = results[
        [
            "scenario",
            "year",
            "value",
            "forecast_world_demand_twh",
            "ai_share_pct",
            "stress_level"
        ]
    ]

    print(summary)

    print("\nHighest Stress Scenario")

    highest = (
        summary
        .sort_values(
            "ai_share_pct",
            ascending=False
        )
        .head(1)
    )

    print(highest)


def save_results(results):

    output_file = (
        OUTPUT_DIR /
        "ai_energy_stress_test.csv"
    )

    results.to_csv(
        output_file,
        index=False
    )

    print(
        f"\nSaved to: {output_file}"
    )


def main():

    energy, ai = load_data()

    world_forecast = forecast_world_electricity(
        energy
    )

    ai_power = get_ai_scenarios(
        ai
    )

    results = build_stress_results(
        world_forecast,
        ai_power
    )

    print_summary(
        results
    )

    save_results(
        results
    )


if __name__ == "__main__":
    main()