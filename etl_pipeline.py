import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path


PROJECT_DIR = Path(r"C:\Users\DELL\OneDrive\Documents\AI_Energy_Project")
RAW_DB_PATH = PROJECT_DIR / "raw_data.db"
CLEAN_DB_PATH = PROJECT_DIR / "clean_energy_ai.db"


def clean_column_names(df):
    df = df.copy()
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace("/", "_", regex=False)
        .str.replace("$", "usd", regex=False)
        .str.replace("%", "pct", regex=False)
        .str.replace("(", "", regex=False)
        .str.replace(")", "", regex=False)
    )
    return df


def transform_energy_data(df):
    print("\nTransforming energy data...")

    df = clean_column_names(df)

    important_columns = [
        "country",
        "year",
        "iso_code",
        "population",
        "gdp",
        "electricity_demand",
        "electricity_generation",
        "per_capita_electricity",
        "primary_energy_consumption",
        "coal_share_elec",
        "gas_share_elec",
        "oil_share_elec",
        "fossil_share_elec",
        "renewables_share_elec",
        "low_carbon_share_elec",
        "nuclear_share_elec",
        "solar_share_elec",
        "wind_share_elec",
        "hydro_share_elec",
        "carbon_intensity_elec"
    ]

    existing_columns = [col for col in important_columns if col in df.columns]
    df = df[existing_columns]

    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.dropna(subset=["country", "year"])
    df["year"] = df["year"].astype(int)

    numeric_columns = [col for col in df.columns if col not in ["country", "iso_code"]]
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.drop_duplicates()

    print(f"Energy data cleaned: {len(df)} rows, {len(df.columns)} columns")
    return df


def transform_ai_data(df):
    print("\nTransforming AI/data center demand data...")

    df = df.copy()
    df = df.dropna(how="all")

    # The AI annex has scenario names and years spread across columns.
    # We reshape it into a clean long format for analysis.
    column_map = {
        3: ("Historical", 2020),
        4: ("Historical", 2023),
        5: ("Historical", 2024),
        7: ("Base", 2030),
        8: ("Base", 2035),
        10: ("Lift-Off", 2030),
        11: ("Lift-Off", 2035),
        13: ("High Efficiency", 2030),
        14: ("High Efficiency", 2035),
        16: ("Headwinds", 2030),
        17: ("Headwinds", 2035),
    }

    records = []
    current_metric_category = None

    for _, row in df.iterrows():
        label = row.iloc[2]

        if pd.isna(label):
            continue

        label = str(label).strip()

        # Skip rows that only contain scenario names or years
        if label.lower() in ["nan", ""]:
            continue

        values = []
        for col_index in column_map:
            if col_index < len(row):
                values.append(row.iloc[col_index])

        numeric_values = pd.to_numeric(pd.Series(values), errors="coerce")

        # If a row has no numeric values, it is a section/category header
        if numeric_values.notna().sum() == 0:
            current_metric_category = label
            continue

        # Otherwise, it is a data row under the current metric category
        for col_index, (scenario, year) in column_map.items():
            if col_index >= len(row):
                continue

            value = pd.to_numeric(row.iloc[col_index], errors="coerce")

            if pd.notna(value):
                records.append({
                    "metric_category": current_metric_category,
                    "segment": label,
                    "scenario": scenario,
                    "year": year,
                    "value": value
                })

    clean_ai = pd.DataFrame(records)

    clean_ai = clean_ai.drop_duplicates()
    clean_ai = clean_ai.sort_values(
        ["metric_category", "segment", "scenario", "year"]
    ).reset_index(drop=True)

    print(f"AI/data center data cleaned: {len(clean_ai)} rows, {len(clean_ai.columns)} columns")
    return clean_ai


def transform_price_data(df):
    print("\nTransforming electricity price data...")

    df = clean_column_names(df)

    date_columns = [
        "trade_date",
        "delivery_start_date",
        "delivery__end_date"
    ]

    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    numeric_columns = [
        "high_price_usd_mwh",
        "low_price_usd_mwh",
        "wtd_avg_price_usd_mwh",
        "change",
        "daily_volume_mwh",
        "number_of_trades",
        "number_of_counterparties"
    ]

    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "trade_date" in df.columns:
        df["year"] = df["trade_date"].dt.year
        df["month"] = df["trade_date"].dt.month

    df = df.drop_duplicates()

    print(f"Electricity price data cleaned: {len(df)} rows, {len(df.columns)} columns")
    return df


def load_clean_data(clean_energy, clean_ai, clean_price):
    print("\nLoading cleaned data into database...")

    conn = sqlite3.connect(CLEAN_DB_PATH)

    clean_energy.to_sql("clean_energy", conn, if_exists="replace", index=False)
    clean_ai.to_sql("clean_ai_demand", conn, if_exists="replace", index=False)
    clean_price.to_sql("clean_price", conn, if_exists="replace", index=False)

    conn.close()

    print(f"Clean database saved at: {CLEAN_DB_PATH}")


def show_clean_tables():
    conn = sqlite3.connect(CLEAN_DB_PATH)

    tables = pd.read_sql(
        """
        SELECT name
        FROM sqlite_master
        WHERE type='table';
        """,
        conn
    )

    print("\nTables in clean database:")
    print(tables)

    for table in tables["name"]:
        print(f"\nPreview of {table}:")
        preview = pd.read_sql(f'SELECT * FROM "{table}" LIMIT 5', conn)
        print(preview)

    conn.close()


def main():
    print("=" * 70)
    print("AI ENERGY ETL PIPELINE")
    print("=" * 70)

    conn = sqlite3.connect(RAW_DB_PATH)

    raw_energy = pd.read_sql("SELECT * FROM energy", conn)
    raw_ai = pd.read_sql("SELECT * FROM Energy_and_AI", conn)
    raw_price = pd.read_sql("SELECT * FROM Price", conn)

    conn.close()

    clean_energy = transform_energy_data(raw_energy)
    clean_ai = transform_ai_data(raw_ai)
    clean_price = transform_price_data(raw_price)

    load_clean_data(clean_energy, clean_ai, clean_price)
    show_clean_tables()

    print("\nETL pipeline completed successfully.")


if __name__ == "__main__":
    main()