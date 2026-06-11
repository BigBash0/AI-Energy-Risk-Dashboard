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


def calculate_cagr(start,end,years):

    return (
        (end/start)**(1/years)-1
    )


def build_forecasts(world):

    world = (
        world
        .sort_values("year")
        .dropna(
            subset=[
                "electricity_demand"
            ]
        )
    )

    recent = world.tail(10)

    start = recent.iloc[0]
    end = recent.iloc[-1]

    cagr = calculate_cagr(
        start["electricity_demand"],
        end["electricity_demand"],
        int(end["year"]-start["year"])
    )

    forecasts = []

    for year in range(
        int(end["year"])+1,
        2036
    ):

        periods = (
            year -
            int(end["year"])
        )

        base = (
            end["electricity_demand"]
            *
            ((1+cagr)**periods)
        )

        optimistic = (
            end["electricity_demand"]
            *
            ((1+(cagr*0.75))**periods)
        )

        accelerated = (
            end["electricity_demand"]
            *
            ((1+(cagr*1.25))**periods)
        )

        forecasts.append({
            "year":year,
            "base_case":base,
            "optimistic_case":optimistic,
            "accelerated_case":accelerated
        })

    return pd.DataFrame(
        forecasts
    )


def main():

    energy = load_data()

    world = energy[
        energy["country"] == "World"
    ].copy()

    forecasts = build_forecasts(
        world
    )

    print(
        forecasts.head()
    )

    forecasts.to_csv(
        OUTPUT_DIR /
        "electricity_forecasts.csv",
        index=False
    )

    print(
        "\nSaved: electricity_forecasts.csv"
    )


if __name__ == "__main__":
    main()