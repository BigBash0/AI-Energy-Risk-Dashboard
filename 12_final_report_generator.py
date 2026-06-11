from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(r"C:\Users\DELL\OneDrive\Documents\AI_Energy_Project")

OUTPUT_DIR = PROJECT_DIR / "outputs"

stress = pd.read_csv(
    OUTPUT_DIR / "ai_energy_stress_test.csv"
)

readiness = pd.read_csv(
    OUTPUT_DIR / "ai_energy_readiness_rankings.csv"
)

risk = pd.read_csv(
    OUTPUT_DIR / "regional_ai_energy_risk_model.csv"
)

forecast = pd.read_csv(
    OUTPUT_DIR / "electricity_forecasts.csv"
)

comparison = pd.read_csv(
    OUTPUT_DIR / "scenario_comparison.csv"
)

transition = pd.read_csv(
    OUTPUT_DIR / "energy_transition_requirements.csv"
)

top_ready = readiness.head(10)

high_risk = risk[
    risk["risk_level"]=="High Risk"
].head(10)

highest_stress = stress.sort_values(
    "ai_share_pct",
    ascending=False
).iloc[0]

report = []

report.append(
    "CAN GLOBAL ENERGY SYSTEMS SUSTAIN THE RAPID GROWTH OF AI?"
)

report.append("\n")
report.append("="*70)

report.append("\nEXECUTIVE SUMMARY\n")

report.append(
    "This study evaluates whether global energy systems can support rapidly growing AI and data-center electricity demand."
)

report.append(
    f"\nHighest stress scenario: {highest_stress['scenario']} ({highest_stress['year']})"
)

report.append(
    f"\nAI Share of Demand: {highest_stress['ai_share_pct']:.2f}%"
)

report.append("\n")

report.append("TOP AI READY COUNTRIES")

for country in top_ready["country"]:
    report.append(f"\n- {country}")

report.append("\n")

report.append("HIGH RISK COUNTRIES")

for country in high_risk["country"]:
    report.append(f"\n- {country}")

report.append("\n")

report.append("KEY FINDINGS")

report.append(
    "\n1. AI demand remains a relatively small share of total global electricity demand."
)

report.append(
    "\n2. Regional grid readiness is more important than global electricity supply."
)

report.append(
    "\n3. Renewable-rich countries are best positioned for AI expansion."
)

report.append(
    "\n4. Carbon-intensive systems face higher risk."
)

report.append(
    "\n5. Grid infrastructure investment will become increasingly important."
)

report.append("\n")

report.append("ENERGY TRANSITION REQUIREMENTS")

for _, row in transition.iterrows():

    report.append(
        f"\n{row['scenario']} {row['year']}"
    )

    report.append(
        f"\nSolar Needed: {row['solar_capacity_needed_gw']:.0f} GW"
    )

    report.append(
        f"\nWind Needed: {row['wind_capacity_needed_gw']:.0f} GW"
    )

report.append("\n")

report.append("FINAL CONCLUSION")

report.append(
    "\nGlobal energy systems can support AI growth, but regional constraints, carbon intensity, and infrastructure readiness will determine where AI expansion occurs successfully."
)

report_file = OUTPUT_DIR / "final_project_report.txt"

with open(report_file,"w",encoding="utf-8") as f:
    f.write("\n".join(report))

print(f"\nSaved: {report_file}")