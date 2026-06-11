import sqlite3
import pandas as pd
from pathlib import Path


PROJECT_DIR = Path(r"C:\Users\DELL\OneDrive\Documents\AI_Energy_Project")
DATA_DIR = PROJECT_DIR / "data"
DB_PATH = PROJECT_DIR / "raw_data.db"

FILES = {
    "energy": DATA_DIR / "energy-data.xlsx.csv",
    "Energy_and_AI": DATA_DIR / "Data_annex_Energy_and_AI.xlsx.xlsx",
    "Price": DATA_DIR / "ICE_Electricity_Price.xlsx.csv",
}


def create_connection(db_path):
    return sqlite3.connect(db_path)


def read_data_file(file_path):
    file_name = file_path.name.lower()

    if file_name.endswith(".csv"):
        return pd.read_csv(file_path)

    if file_name.endswith(".xlsx") or file_name.endswith(".xls"):
        return pd.read_excel(file_path)

    raise ValueError(f"Unsupported file type: {file_path}")


def ingest_files(conn, files):
    for table_name, file_path in files.items():
        if not file_path.exists():
            print(f"File not found: {file_path}")
            continue

        df = read_data_file(file_path)

        df.to_sql(
            name=table_name,
            con=conn,
            if_exists="replace",
            index=False
        )

        print(f"Loaded {len(df)} rows into table: {table_name}")


def show_tables(conn):
    query = """
    SELECT name
    FROM sqlite_master
    WHERE type='table';
    """

    tables = pd.read_sql(query, conn)

    print("\nTables in database:")
    print(tables)

    for table in tables["name"]:
        print(f"\nPreview of {table}:")
        preview = pd.read_sql(f'SELECT * FROM "{table}" LIMIT 5', conn)
        print(preview)


def main():
    conn = create_connection(DB_PATH)

    try:
        ingest_files(conn, FILES)
        show_tables(conn)
    finally:
        conn.close()

    print("\nData ingestion completed successfully.")
    print(f"Database saved at: {DB_PATH}")


if __name__ == "__main__":
    main()