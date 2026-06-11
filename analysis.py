import sqlite3
import pandas as pd
from pathlib import Path


PROJECT_DIR = Path(r"C:\Users\DELL\OneDrive\Documents\AI_Energy_Project")
DB_PATH = PROJECT_DIR / "clean_energy_ai.db"
OUTPUT_DIR = PROJECT_DIR / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)


def connect_database():
    return sqlite3.connect(DB_PATH)


def load_tables(conn):
    clean_energy = pd.read_sql("SELECT * FROM clean_energy", conn)
    clean_ai = pd.read_sql("SELECT * FROM clean_ai_demand", conn)
    clean_price = pd.read_sql("SELECT * FROM clean_price", conn)

    return clean_energy, clean_ai, clean_price


def analyze_global_electricity_growth(clean_energy):
    print("\n1. Global Electricity Demand Growth")

    df = clean_energy.copy()

    global_df = df[df["country"].str.lower() == "world"].copy()

    if global_df.empty:
        print("No 'World' records found in clean_energy.")
        return pd.DataFrame()

    global_df = global_df.sort_values("year")

    global_df["demand_growth_pct"] = (
        global_df["electricity_demand"].pct_change() * 100
    )

    result = global_df[
        [
            "country",
            "year",
            "electricity_demand",
            "electricity_generation",
            "per_capita_electricity",
            "demand_growth_pct",
        ]
    ]

    print(result.tail(10))
    result.to_csv(OUTPUT_DIR / "global_electricity_growth.csv", index=False)

    return result


def analyze_top_demand_regions(clean_energy):
    print("\n2. Top Countries/Regions by Electricity Demand")

    df = clean_energy.copy()

    latest_year = df["year"].max()

    latest_df = df[
        (df["year"] == latest_year) &
        (df["electricity_demand"].notna())
    ].copy()

    result = latest_df.sort_values(
        "electricity_demand",
        ascending=False
    ).head(15)

    result = result[
        [
            "country",
            "year",
            "electricity_demand",
            "electricity_generation",
            "fossil_share_elec",
            "renewables_share_elec",
            "low_carbon_share_elec",
            "carbon_intensity_elec",
        ]
    ]

    print(result)
    result.to_csv(OUTPUT_DIR / "top_demand_regions.csv", index=False)

    return result


def analyze_ai_demand_scenarios(clean_ai):
    print("\n3. AI/Data Center Electricity Demand Scenarios")

    df = clean_ai.copy()

    ai_electricity = df[
        df["metric_category"].str.lower().eq("electricity consumption (twh)")
    ].copy()

    result = ai_electricity.sort_values(
        ["segment", "scenario", "year"]
    )

    print(result.head(20))
    result.to_csv(OUTPUT_DIR / "ai_electricity_demand_scenarios.csv", index=False)

    return result


def analyze_energy_mix(clean_energy):
    print("\n4. Energy Mix Comparison")

    df = clean_energy.copy()

    latest_year = df["year"].max()

    latest_df = df[
        (df["year"] == latest_year) &
        (df["fossil_share_elec"].notna()) &
        (df["renewables_share_elec"].notna())
    ].copy()

    result = latest_df[
        [
            "country",
            "year",
            "coal_share_elec",
            "gas_share_elec",
            "oil_share_elec",
            "fossil_share_elec",
            "renewables_share_elec",
            "low_carbon_share_elec",
            "nuclear_share_elec",
            "carbon_intensity_elec",
        ]
    ].sort_values("fossil_share_elec", ascending=False).head(20)

    print(result)
    result.to_csv(OUTPUT_DIR / "energy_mix_comparison.csv", index=False)

    return result


def analyze_electricity_prices(clean_price):
    print("\n5. Electricity Price Trend")

    df = clean_price.copy()

    df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")

    price_df = df[
        df["wtd_avg_price_usd_mwh"].notna()
    ].copy()

    monthly_prices = (
        price_df
        .groupby(["price_hub", "year", "month"], as_index=False)
        .agg(
            avg_price_usd_mwh=("wtd_avg_price_usd_mwh", "mean"),
            max_price_usd_mwh=("high_price_usd_mwh", "max"),
            min_price_usd_mwh=("low_price_usd_mwh", "min"),
            total_volume_mwh=("daily_volume_mwh", "sum"),
            number_of_trades=("number_of_trades", "sum")
        )
    )

    result = monthly_prices.sort_values(
        ["year", "month", "avg_price_usd_mwh"],
        ascending=[True, True, False]
    )

    print(result.tail(20))
    result.to_csv(OUTPUT_DIR / "electricity_price_trends.csv", index=False)

    return result


def main():
    print("=" * 70)
    print("AI ENERGY ANALYSIS")
    print("=" * 70)

    conn = connect_database()

    clean_energy, clean_ai, clean_price = load_tables(conn)

    conn.close()

    analyze_global_electricity_growth(clean_energy)
    analyze_top_demand_regions(clean_energy)
    analyze_ai_demand_scenarios(clean_ai)
    analyze_energy_mix(clean_energy)
    analyze_electricity_prices(clean_price)

    print("\nAnalysis completed successfully.")
    print(f"Results saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()