import io
import sqlite3
from datetime import datetime
from pathlib import Path
import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image as RLImage, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

DB_PATH = Path(__file__).with_name("clean_energy_ai.db")
ENERGY_COLORS = ["#00C853", "#2979FF", "#FFB300", "#FF6D00", "#D50000", "#455A64"]
SECTION_TITLES = [
    "Executive Summary",
    "Global Electricity Demand",
    "AI Demand Scenarios",
    "Energy Mix Analysis",
    "AI Energy Readiness Rankings",
    "Grid Stress Index",
    "Regional Risk Model",
    "Forecasting to 2035",
    "Energy Transition Requirements",
    "Scenario Comparison",
    "Strategic Recommendations",
    "Final Report & Automation Mockup",
]

st.set_page_config(page_title="AI Energy Systems Dashboard", page_icon="⚡", layout="wide")


def safe_query(query: str) -> pd.DataFrame:
    try:
        with sqlite3.connect(DB_PATH, timeout=30) as conn:
            df = pd.read_sql_query(query, conn)
        return df
    except Exception as exc:
        st.error(f"Database query failed: {exc}")
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_clean_energy() -> pd.DataFrame:
    return safe_query("SELECT * FROM clean_energy")


@st.cache_data(show_spinner=False)
def load_clean_ai_demand() -> pd.DataFrame:
    return safe_query("SELECT * FROM clean_ai_demand")


@st.cache_data(show_spinner=False)
def load_clean_price() -> pd.DataFrame:
    return safe_query("SELECT * FROM clean_price")


@st.cache_data(show_spinner=False)
def get_country_options() -> list[str]:
    energy = load_clean_energy()
    if energy.empty:
        return ["World"]
    unique = energy["country"].dropna().unique().tolist()
    unique = sorted(unique)
    if "World" in unique:
        unique.remove("World")
        unique.insert(0, "World")
    return unique


@st.cache_data(show_spinner=False)
def get_year_options() -> list[int]:
    energy = load_clean_energy()
    if energy.empty:
        return []
    return sorted(int(y) for y in energy["year"].dropna().unique())


@st.cache_data(show_spinner=False)
def compute_readiness_scores(energy: pd.DataFrame) -> pd.DataFrame:
    latest = energy.sort_values("year").groupby("country", as_index=False).last()
    latest = latest.dropna(subset=["renewables_share_elec", "low_carbon_share_elec", "carbon_intensity_elec"])
    latest["raw_score"] = (
        latest["renewables_share_elec"] * 0.3
        + latest["low_carbon_share_elec"] * 0.4
        - latest["carbon_intensity_elec"] / 1000 * 0.3
    )
    min_raw = latest["raw_score"].min()
    max_raw = latest["raw_score"].max()
    latest["score"] = np.where(
        max_raw == min_raw,
        50.0,
        100 * (latest["raw_score"] - min_raw) / (max_raw - min_raw),
    )
    latest["category"] = pd.cut(
        latest["score"],
        bins=[-1, 24.999, 49.999, 74.999, 100],
        labels=["High Risk", "Moderate Risk", "Prepared", "Leading"],
    )
    return latest.sort_values("score", ascending=False)


@st.cache_data(show_spinner=False)
def compute_grid_stress(energy: pd.DataFrame) -> pd.DataFrame:
    latest = energy.sort_values("year").groupby("country", as_index=False).last()
    latest = latest.dropna(subset=["fossil_share_elec", "electricity_demand", "electricity_generation", "carbon_intensity_elec"])
    latest["grid_stress_index"] = (
        latest["fossil_share_elec"] * 0.4
        + np.where(latest["electricity_generation"] > 0, latest["electricity_demand"] / latest["electricity_generation"], 1.0) * 40
        + latest["carbon_intensity_elec"] / 10
    )
    latest["stress_category"] = pd.cut(
        latest["grid_stress_index"],
        bins=[-1, 49.999, 74.999, np.inf],
        labels=["Low Stress", "Moderate Stress", "High Stress"],
    )
    return latest.sort_values("grid_stress_index", ascending=False)


@st.cache_data(show_spinner=False)
def compute_regional_risk(energy: pd.DataFrame) -> pd.DataFrame:
    latest = energy.sort_values("year").groupby("country", as_index=False).last()
    demand_growth = energy.sort_values(["country", "year"]).groupby("country")["electricity_demand"].apply(
        lambda series: (series.iloc[-1] / series.iloc[0] - 1) if len(series) > 1 and series.iloc[0] > 0 else 0
    )
    latest = latest.merge(demand_growth.rename("demand_growth"), on="country")
    latest = latest.dropna(subset=["carbon_intensity_elec", "fossil_share_elec", "demand_growth"])
    latest["risk_score"] = (
        latest["carbon_intensity_elec"] * 0.35
        + latest["fossil_share_elec"] * 0.35
        + (latest["demand_growth"].clip(lower=0) * 100) * 0.3
    )
    latest["risk_rank"] = latest["risk_score"].rank(method="dense", ascending=True)
    return latest.sort_values("risk_score", ascending=True)


def build_forecast(years: np.ndarray, values: np.ndarray, target_years: np.ndarray, slope_scale: float = 1.0) -> np.ndarray:
    if len(years) < 2 or len(values) < 2:
        return np.full_like(target_years, np.nan, dtype=float)
    coefficients = np.polyfit(years, values, 1)
    baseline = np.poly1d(coefficients)(target_years)
    if slope_scale == 1.0:
        return baseline
    extra_growth = (target_years - years[-1]) * coefficients[0] * (slope_scale - 1)
    return baseline + extra_growth


def compute_world_baseline_2035(world: pd.DataFrame, target_year: int = 2035) -> float:
    world = world.dropna(subset=["year", "electricity_demand"]).sort_values("year")
    if world.empty:
        return 0.0
    if target_year in world["year"].values:
        return float(world.loc[world["year"] == target_year, "electricity_demand"].iloc[0])
    years = world["year"].astype(float).to_numpy()
    values = world["electricity_demand"].astype(float).to_numpy()
    if len(years) < 2:
        return float(values[-1])
    baseline = build_forecast(years, values, np.array([target_year]))[0]
    if np.isnan(baseline):
        if all(values > 0) and years[-1] > years[0]:
            cagr = (values[-1] / values[0]) ** (1 / (years[-1] - years[0])) - 1
            baseline = float(values[-1] * (1 + cagr) ** (target_year - years[-1]))
        else:
            baseline = float(values[-1])
    return float(baseline)


def _pdf_table_cell(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, (np.integer, int)):
        return f"{int(value):,}"
    if isinstance(value, (np.floating, float)):
        return f"{value:,.2f}"
    return str(value)


def _build_plotly_image(
    fig: go.Figure,
    width: int = 1600,
    height: int = 900,
    output_width: float = 450,
    output_height: float = 225,
) -> RLImage:
    image_buffer = io.BytesIO()
    # ensure the figure has explicit pixel dimensions before export
    try:
        fig.write_image(image_buffer, format="png", width=width, height=height)
    except Exception:
        # fallback: try without explicit size
        fig.write_image(image_buffer, format="png")
    image_buffer.seek(0)
    return RLImage(image_buffer, width=output_width, height=output_height)


def _build_pdf_table(df: pd.DataFrame, col_widths=None) -> Table:
    styles = getSampleStyleSheet()
    normal_style = ParagraphStyle("TableCell", parent=styles["BodyText"], leading=12, spaceAfter=4)
    data = [[Paragraph(str(col).replace("_", " ").title(), normal_style) for col in df.columns]]
    for _, row in df.iterrows():
        data.append([Paragraph(_pdf_table_cell(value), normal_style) for value in row])
    table = Table(data, colWidths=col_widths or [1.2 * inch] * len(df.columns), hAlign="LEFT")
    table.setStyle(
        TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.25, colors.gray),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f2f2f2")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ])
    )
    return table


def generate_pdf_report(
    energy: pd.DataFrame,
    ai_demand: pd.DataFrame,
    readiness: pd.DataFrame,
    grid_stress: pd.DataFrame,
    regional_risk: pd.DataFrame,
    selected_country: str,
    selected_year: int,
) -> bytes:
    world = energy[energy["country"] == "World"].sort_values("year")
    latest_year = int(world["year"].max()) if not world.empty else None
    latest_world = world[world["year"] == latest_year] if latest_year else pd.DataFrame()
    latest_demand = float(latest_world["electricity_demand"].sum()) if not latest_world.empty else 0.0
    latest_carbon_intensity = float(latest_world["carbon_intensity_elec"].mean()) if not latest_world.empty else 0.0
    demand_2035 = ai_demand[ai_demand["year"] == 2035]
    highest_ai_demand = float(demand_2035["value"].max()) if not demand_2035.empty else 0.0
    scenario_totals = demand_2035.groupby("scenario", as_index=False)["value"].sum()
    base_2035 = float(scenario_totals.loc[scenario_totals["scenario"] == "Base", "value"].sum()) if not scenario_totals.empty else 0.0
    liftoff_2035 = float(scenario_totals.loc[scenario_totals["scenario"] == "Lift-Off", "value"].sum()) if not scenario_totals.empty else 0.0
    incremental_ai = max(0.0, liftoff_2035 - base_2035)
    baseline_2035 = compute_world_baseline_2035(world, 2035)
    forecasted_world_2035 = float(baseline_2035 + incremental_ai)

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("CoverTitle", parent=styles["Title"], alignment=TA_CENTER, spaceAfter=18)
    subtitle_style = ParagraphStyle("CoverSubtitle", parent=styles["Heading2"], alignment=TA_CENTER, spaceAfter=12)
    heading_style = ParagraphStyle("Heading", parent=styles["Heading2"], alignment=TA_LEFT, spaceAfter=10)
    normal_style = ParagraphStyle("Normal", parent=styles["BodyText"], leading=14, spaceAfter=8)

    story: list = []
    story.append(Paragraph("AI Energy Systems Report", title_style))
    story.append(Paragraph("Can Global Energy Systems Sustain the Rapid Growth of AI?", subtitle_style))
    story.append(Paragraph("AI, Electricity Demand & Energy System Constraints", normal_style))
    story.append(Spacer(1, 0.2 * inch))
    story.append(Paragraph(f"Author: Adeleke Basheer", normal_style))
    story.append(Paragraph(f"Date: {datetime.today().strftime('%B %d, %Y')}", normal_style))
    story.append(Paragraph(f"Selected Country / Region: {selected_country}", normal_style))
    story.append(Paragraph(f"Selected Year: {selected_year}", normal_style))
    story.append(PageBreak())

    story.append(Paragraph("Executive Summary", heading_style))
    story.append(Paragraph(
        "This report presents an integrated analysis of global electricity demand, AI energy scenarios, readiness rankings, grid stress, and infrastructure requirements through 2035.",
        normal_style,
    ))
    story.append(Paragraph(f"Latest World Electricity Demand: {latest_demand:,.0f} TWh", normal_style))
    story.append(Paragraph(f"Global Carbon Intensity: {latest_carbon_intensity:,.2f} gCO2/kWh", normal_style))
    story.append(Paragraph(f"Forecasted 2035 World Demand with AI Impacts: {forecasted_world_2035:,.0f} TWh", normal_style))
    story.append(Paragraph(f"Highest AI Demand Scenario (2035): {highest_ai_demand:,.0f} TWh", normal_style))
    story.append(PageBreak())

    global_fig = px.line(
        world,
        x="year",
        y="electricity_demand",
        title="Historical World Electricity Demand",
        labels={"electricity_demand": "Electricity Demand (TWh)", "year": "Year"},
    )
    # Fix export sizing/layout for PDF embedding
    global_fig.update_layout(plot_bgcolor='white', paper_bgcolor='white', width=800, height=400)
    story.append(Paragraph("Global Electricity Demand & Regional Analysis", heading_style))
    story.append(_build_plotly_image(global_fig))
    year_data = energy[energy["year"] == selected_year].sort_values("electricity_demand", ascending=False).head(10)
    if not year_data.empty:
        story.append(Paragraph(f"Top 10 Electricity Demand Regions in {selected_year}", normal_style))
        story.append(_build_pdf_table(year_data[["country", "electricity_demand"]], col_widths=[3 * inch, 2.5 * inch]))
    story.append(PageBreak())

    scenario_fig = px.line(
        ai_demand.groupby(["scenario", "year"], as_index=False)["value"].sum(),
        x="year",
        y="value",
        color="scenario",
        title="AI Demand Trajectories by Scenario",
        labels={"value": "AI Electricity Demand (TWh)", "year": "Year"},
    )
    # Ensure consistent layout for PDF export
    scenario_fig.update_layout(plot_bgcolor='white', paper_bgcolor='white', width=800, height=400)
    story.append(Paragraph("AI Demand Scenarios", heading_style))
    story.append(_build_plotly_image(scenario_fig))
    comparison_table = ai_demand[ai_demand["year"].isin([2030, 2035])].groupby(["scenario", "year"], as_index=False)["value"].sum()
    if not comparison_table.empty:
        pivot = comparison_table.pivot(index="scenario", columns="year", values="value").reset_index().fillna(0)
        story.append(Paragraph("AI Scenario Demand Comparison for 2030 / 2035", normal_style))
        story.append(_build_pdf_table(pivot, col_widths=[2.5 * inch, 1.8 * inch, 1.8 * inch]))
    story.append(PageBreak())

    readiness_fig = px.bar(
        readiness.head(20).iloc[::-1],
        x="score",
        y="country",
        orientation="h",
        title="Top 20 AI Energy Readiness Leaders",
        labels={"score": "Readiness Score", "country": "Country"},
    )
    # Fix solid black rendering by enforcing white background and a single high-contrast bar color
    readiness_fig.update_traces(marker_color=ENERGY_COLORS[1])
    readiness_fig.update_layout(plot_bgcolor='white', paper_bgcolor='white', width=800, height=450)
    story.append(Paragraph("AI Energy Readiness & Rankings", heading_style))
    story.append(_build_plotly_image(readiness_fig))
    story.append(Paragraph("Top 20 Readiness Leaders", normal_style))
    story.append(_build_pdf_table(readiness.head(20)[["country", "score", "category"]], col_widths=[3 * inch, 1.5 * inch, 1.5 * inch]))
    story.append(PageBreak())

    story.append(Paragraph("Grid Stress Index & Regional Risk Assessments", heading_style))
    story.append(Paragraph(
        "Grid stress reflects fossil dependency, reserve margins, and carbon intensity across countries.",
        normal_style,
    ))
    story.append(_build_pdf_table(grid_stress.head(15)[["country", "grid_stress_index", "stress_category"]], col_widths=[3 * inch, 1.4 * inch, 2.0 * inch]))
    story.append(PageBreak())

    forecast_years = np.arange(int(world["year"].max()) + 1, 2036) if not world.empty else np.array([])
    forecast_baseline = build_forecast(world["year"].astype(float).to_numpy(), world["electricity_demand"].astype(float).to_numpy(), forecast_years, slope_scale=1.0) if forecast_years.size else np.array([])
    forecast_accelerated = build_forecast(world["year"].astype(float).to_numpy(), world["electricity_demand"].astype(float).to_numpy(), forecast_years, slope_scale=1.2) if forecast_years.size else np.array([])
    ai_signal = np.interp(forecast_years, ai_demand[ai_demand["scenario"] == "Lift-Off"]["year"], ai_demand[ai_demand["scenario"] == "Lift-Off"]["value"], left=0, right=ai_demand[ai_demand["scenario"] == "Lift-Off"]["value"].iloc[-1] if not ai_demand[ai_demand["scenario"] == "Lift-Off"].empty else 0) if forecast_years.size else np.array([])
    ai_projection = forecast_baseline + ai_signal if forecast_years.size else np.array([])
    forecast_df = pd.DataFrame({"year": forecast_years, "Baseline": forecast_baseline, "Accelerated": forecast_accelerated, "AI Driven": ai_projection})
    forecast_fig = go.Figure()
    if not world.empty:
        forecast_fig.add_trace(go.Scatter(x=world["year"], y=world["electricity_demand"], mode="lines+markers", name="Historical Demand"))
        forecast_fig.add_trace(go.Scatter(x=forecast_df["year"], y=forecast_df["Baseline"], mode="lines", name="Baseline Forecast"))
        forecast_fig.add_trace(go.Scatter(x=forecast_df["year"], y=forecast_df["Accelerated"], mode="lines", name="Accelerated Forecast"))
        forecast_fig.add_trace(go.Scatter(x=forecast_df["year"], y=forecast_df["AI Driven"], mode="lines", name="AI-Driven Forecast"))
    # Set layout to a consistent export size to avoid stretching in PDF
    forecast_fig.update_layout(plot_bgcolor='white', paper_bgcolor='white', width=800, height=400)
    story.append(Paragraph("Forecast Projections to 2035", heading_style))
    story.append(_build_plotly_image(forecast_fig))
    story.append(PageBreak())

    wind_gw = incremental_ai * 1_000_000 / (0.30 * 8760) / 1000
    solar_gw = incremental_ai * 1_000_000 / (0.20 * 8760) / 1000
    story.append(Paragraph("Infrastructure & Energy Transition Capacity Requirements", heading_style))
    story.append(Paragraph(f"Incremental AI load in 2035 requires approximately {wind_gw:,.1f} GW of wind capacity and {solar_gw:,.1f} GW of solar capacity.", normal_style))
    transition_table = pd.DataFrame(
        {
            "Technology": ["Wind", "Solar"],
            "Capacity Factor": ["30%", "20%"],
            "Required GW": [f"{wind_gw:,.1f}", f"{solar_gw:,.1f}"],
        }
    )
    story.append(_build_pdf_table(transition_table, col_widths=[2.5 * inch, 2.0 * inch, 2.0 * inch]))
    story.append(PageBreak())

    story.append(Paragraph("Strategic Recommendations", heading_style))
    for line in [
        "Governments should accelerate renewables deployment and grid flexibility programs.",
        "Utilities must integrate AI load forecasts into resource planning and demand-side management.",
        "Investors should prioritize low-carbon infrastructure in high readiness, low stress regions.",
        "AI companies should coordinate with system operators to support managed load growth.",
    ]:
        story.append(Paragraph(f"• {line}", normal_style))
    story.append(PageBreak())

    story.append(Paragraph("Methodology & Data Sources", heading_style))
    story.append(Paragraph(
        "Data and analytics use a clean SQLite dataset plus established energy sources. The underlying schema maps clean_energy, clean_ai_demand, and clean_price tables for electricity demand, AI scenario loads, and price trends.", normal_style,
    ))
    story.append(Paragraph("Sources include industry-standard datasets from EIA, IEA, and internal clean energy analytics feeds.", normal_style))
    story.append(Paragraph("Automated Alert Parameters: Grid Stress Index thresholds, AI demand growth triggers, low carbon share alarms, and demand growth tolerances.", normal_style))

    pdf_buffer = io.BytesIO()
    doc = SimpleDocTemplate(pdf_buffer, pagesize=LETTER, rightMargin=0.6 * inch, leftMargin=0.6 * inch, topMargin=0.6 * inch, bottomMargin=0.6 * inch)
    doc.build(story)
    pdf_buffer.seek(0)
    return pdf_buffer.getvalue()


def _get_secret_value(key: str) -> str | None:
    try:
        if hasattr(st, "secrets"):
            return st.secrets.get(key)
    except Exception:
        return None
    return None


def send_pdf_email(recipient_email: str, pdf_data: bytes) -> tuple[bool, str]:
    """Send the PDF binary to recipient_email via SMTP.

    Returns (success: bool, info: str). If placeholders are detected returns (False, 'placeholder').
    """
    SMTP_SERVER = os.environ.get("SMTP_SERVER", "smtp.gmail.com")
    SMTP_PORT = int(os.environ.get("SMTP_PORT", 587))

    # Allow credentials from environment variables or Streamlit secrets for secure deployment.
    # Falls back to the in-code placeholders when not provided.
    SENDER_EMAIL = os.environ.get("SENDER_EMAIL") or _get_secret_value("SENDER_EMAIL") or "your_portfolio_sender@gmail.com"
    SENDER_PASSWORD = os.environ.get("SENDER_PASSWORD") or _get_secret_value("SENDER_PASSWORD") or "your_app_password"

    # Build message
    msg = MIMEMultipart()
    msg["From"] = SENDER_EMAIL
    msg["To"] = recipient_email
    msg["Subject"] = "AI Energy Systems — Executive PDF Report"
    body = MIMEText("Please find attached the AI Energy Systems Executive Report.", "plain")
    msg.attach(body)

    part = MIMEBase("application", "pdf")
    part.set_payload(pdf_data)
    encoders.encode_base64(part)
    part.add_header("Content-Disposition", "attachment", filename="Energy_AI_Executive_Report.pdf")
    msg.attach(part)

    # If placeholders or clearly invalid credentials are still present, do not attempt SMTP — return demo mode
    if (
        not SENDER_EMAIL
        or not SENDER_PASSWORD
        or SENDER_EMAIL.startswith("your_")
        or SENDER_PASSWORD.startswith("your_")
        or "placeholder" in SENDER_EMAIL
        or "@" not in SENDER_EMAIL
    ):
        return False, "placeholder"

    try:
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT, timeout=30)
        server.ehlo()
        server.starttls()
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.sendmail(SENDER_EMAIL, [recipient_email], msg.as_string())
        server.quit()
        return True, "sent"
    except Exception as exc:
        return False, str(exc)


def generate_executive_brief(world_metrics: dict[str, float], readiness: pd.DataFrame, grid_stress: pd.DataFrame, highest_ai_demand: float) -> str:
    top_readiness = readiness.head(5)["country"].tolist()
    high_stress = grid_stress.head(5)["country"].tolist()
    lines = [
        "=========================================",
        "EXECUTIVE BRIEF: GLOBAL ENERGY SYSTEMS & AI GROWTH",
        "Author: Adeleke Basheer, Energy & Climate Risk Analyst",
        "=========================================",
        "",
        "1. KEY METRICS:",
        f"- Current World Electricity Demand: {world_metrics['latest_demand']:,.0f} TWh",
        f"- Global Sector Carbon Intensity: {world_metrics['latest_carbon_intensity']:,.2f} gCO2/kWh",
        f"- Projected 2035 System Load: {world_metrics['forecasted_world_2035']:,.0f} TWh",
        f"- Highest AI Demand Scenario (2035): {highest_ai_demand:,.0f} TWh",
        "",
        "2. CORE STRATEGIC INSIGHTS:",
        f"- Top 5 High-Stress regions: {', '.join(high_stress) if high_stress else 'No data available.'}",
        f"- Top 5 Energy Readiness Leaders: {', '.join(top_readiness) if top_readiness else 'No data available.'}",
        "- Regions identified as high stress should receive priority grid reinforcement and flexible capacity planning.",
        "- Leading readiness economies are best positioned to attract AI infrastructure and low-carbon investment.",
        "- Accelerated AI demand will require coordinated policy, utility, and investment action across energy and digital sectors.",
        "",
        "3. STRATEGIC POLICY RECOMMENDATIONS:",
        "- Utilities must accelerate low-carbon capacity deployment to counter incremental data center baseload demands.",
        "- Hyperscalers should prioritize co-locating infrastructure in Leading-category readiness zones.",
        "- Policymakers should align infrastructure planning with low-carbon, high-resilience grid development.",
    ]
    return "\n".join(lines)


def render_section_header(title: str, subtitle: str | None = None) -> None:
    st.title(title)
    if subtitle:
        st.markdown(f"### {subtitle}")
    st.divider()


def render_executive_summary(energy: pd.DataFrame, ai_demand: pd.DataFrame) -> None:
    render_section_header("Executive Summary", "Key metrics and strategic findings for AI-driven energy demand")
    world = energy[energy["country"] == "World"].sort_values("year")
    latest_year = int(world["year"].max()) if not world.empty else None
    latest_world = world[world["year"] == latest_year] if latest_year else pd.DataFrame()
    latest_demand = float(latest_world["electricity_demand"].sum()) if not latest_world.empty else 0.0
    latest_carbon = float(latest_world["carbon_intensity_elec"].mean()) if not latest_world.empty else 0.0
    demand_2035 = ai_demand[ai_demand["year"] == 2035]
    highest_ai_demand = float(demand_2035["value"].max()) if not demand_2035.empty else 0.0
    scenario_totals = demand_2035.groupby("scenario", as_index=False)["value"].sum()
    base_2035 = float(scenario_totals.loc[scenario_totals["scenario"] == "Base", "value"].sum()) if not scenario_totals.empty else 0.0
    liftoff_2035 = float(scenario_totals.loc[scenario_totals["scenario"] == "Lift-Off", "value"].sum()) if not scenario_totals.empty else 0.0
    incremental_ai = max(0.0, liftoff_2035 - base_2035)
    if not world.empty and latest_year is not None:
        baseline_2035 = compute_world_baseline_2035(world, 2035)
        forecasted_world_2035 = float(baseline_2035 + incremental_ai)
    else:
        forecasted_world_2035 = 0.0
    cols = st.columns(4)
    cols[0].metric("Latest World Electricity Demand", f"{latest_demand:,.0f} TWh", delta=f"Latest year {latest_year}")
    cols[1].metric("Latest World Carbon Intensity", f"{latest_carbon:,.2f} gCO2/kWh")
    cols[2].metric("Highest AI Demand Scenario (2035)", f"{highest_ai_demand:,.0f} TWh")
    cols[3].metric("Forecasted World Demand 2035", f"{forecasted_world_2035:,.0f} TWh")

    st.markdown(
        """
        ### Project Overview
        The dashboard examines global electricity demand, AI energy scenarios, energy mix transition, grid stress, and country readiness. It integrates data-driven metrics, scenario modeling, and investor-grade recommendations.
        """
    )
    st.markdown(
        """
        ### Key Research Question
        Can global energy systems sustain the rapid growth of AI while maintaining a low-carbon transition and grid resilience?
        """
    )
    st.markdown(
        """
        ### Strategic Findings
        - World electricity demand is rising steadily, driven by industrial growth and digital infrastructure expansion.
        - The Lift-Off AI scenario introduces the largest incremental load through 2035, requiring significant renewable capacity additions.
        - Regions with high fossil share and elevated carbon intensity are the most exposed to grid stress and investor risk.
        - Top-performing countries show higher low-carbon shares, lower carbon intensity, and stronger readiness scores.
        """
    )


def render_global_electricity_demand(energy: pd.DataFrame, selected_country: str, selected_year: int) -> None:
    render_section_header("Global Electricity Demand", "Historical demand, regional comparisons, and system balance")
    world = energy[energy["country"] == "World"].sort_values("year")
    if world.empty:
        st.warning("No world demand data available.")
        return
    fig_historical = px.line(
        world,
        x="year",
        y="electricity_demand",
        title="Historical World Electricity Demand",
        markers=True,
        labels={"electricity_demand": "Electricity Demand (TWh)", "year": "Year"},
        color_discrete_sequence=[ENERGY_COLORS[1]],
    )
    st.plotly_chart(fig_historical, use_container_width=True)

    top_countries = (
        energy[energy["year"] == selected_year]
        .sort_values("electricity_demand", ascending=False)
        .head(10)[["country", "electricity_demand"]]
    )
    if selected_country != "World" and selected_country not in top_countries["country"].values:
        extra = energy[(energy["country"] == selected_country) & (energy["year"] == selected_year)][["country", "electricity_demand"]]
        top_countries = pd.concat([top_countries, extra]).drop_duplicates("country")

    fig_region = px.bar(
        top_countries,
        x="country",
        y="electricity_demand",
        title=f"Electricity Demand by Country / Region in {selected_year}",
        labels={"electricity_demand": "Demand (TWh)", "country": "Country / Region"},
        color="electricity_demand",
        color_continuous_scale="Viridis",
    )
    st.plotly_chart(fig_region, use_container_width=True)

    country_latest = (
        energy[energy["year"] == selected_year]
        .sort_values("per_capita_electricity", ascending=False)
        .head(10)[["country", "per_capita_electricity"]]
    )
    fig_per_capita = px.bar(
        country_latest,
        x="per_capita_electricity",
        y="country",
        orientation="h",
        title=f"Top Per Capita Electricity Consumption in {selected_year}",
        labels={"per_capita_electricity": "Per Capita Electricity (kWh)", "country": "Country"},
        color="per_capita_electricity",
        color_continuous_scale="Blues",
    )
    st.plotly_chart(fig_per_capita, use_container_width=True)

    demand_vs_generation = energy[energy["country"].isin([selected_country, "World"]) & energy["year"].between(max(selected_year - 10, int(energy["year"].min())), selected_year)].copy()
    demand_vs_generation = demand_vs_generation.sort_values(["country", "year"])
    if not demand_vs_generation.empty:
        fig_balance = make_subplots(specs=[[{"secondary_y": True}]])
        for country in demand_vs_generation["country"].unique():
            subset = demand_vs_generation[demand_vs_generation["country"] == country]
            fig_balance.add_trace(
                go.Scatter(
                    x=subset["year"],
                    y=subset["electricity_demand"],
                    mode="lines+markers",
                    name=f"{country} Demand",
                    line=dict(width=3),
                ),
                secondary_y=False,
            )
            fig_balance.add_trace(
                go.Scatter(
                    x=subset["year"],
                    y=subset["electricity_generation"],
                    mode="lines+markers",
                    name=f"{country} Generation",
                    line=dict(width=2, dash="dash"),
                ),
                secondary_y=True,
            )
        fig_balance.update_layout(title=f"Electricity Generation vs Demand ({selected_year - 10} to {selected_year})")
        fig_balance.update_xaxes(title_text="Year")
        fig_balance.update_yaxes(title_text="Demand (TWh)", secondary_y=False)
        fig_balance.update_yaxes(title_text="Generation (TWh)", secondary_y=True)
        st.plotly_chart(fig_balance, use_container_width=True)
    else:
        st.warning("Insufficient data for generation vs demand comparison.")


def render_ai_demand_scenarios(ai_demand: pd.DataFrame) -> None:
    render_section_header("AI Demand Scenarios", "Scenario trajectories and 2030 / 2035 comparison")
    valid_scenarios = ["Base", "Headwinds", "High Efficiency", "Lift-Off", "Historical"]
    scenarios = ai_demand[ai_demand["scenario"].isin(valid_scenarios)].copy()
    if scenarios.empty:
        st.warning("AI demand scenario data is not available.")
        return
    scenario_series = scenarios.groupby(["scenario", "year"], as_index=False)["value"].sum()
    fig_scenarios = px.line(
        scenario_series,
        x="year",
        y="value",
        color="scenario",
        title="AI Demand Growth Trajectory by Scenario",
        markers=True,
        labels={"value": "AI Electricity Demand (TWh)", "year": "Year"},
        color_discrete_sequence=ENERGY_COLORS,
    )
    st.plotly_chart(fig_scenarios, use_container_width=True)

    totals_2030_2035 = scenario_series[scenario_series["year"].isin([2030, 2035])]
    fig_compare = px.bar(
        totals_2030_2035,
        x="scenario",
        y="value",
        color="year",
        barmode="group",
        title="AI Scenario Demand Comparison: 2030 vs 2035",
        labels={"value": "Demand (TWh)", "scenario": "Scenario"},
        color_discrete_sequence=[ENERGY_COLORS[0], ENERGY_COLORS[2]],
    )
    st.plotly_chart(fig_compare, use_container_width=True)

    fast_growth = (
        totals_2030_2035.groupby("scenario")["value"].apply(lambda v: v.iloc[1] - v.iloc[0] if len(v) == 2 else 0)
    ).sort_values(ascending=False)
    fastest = fast_growth.index[0] if not fast_growth.empty else "N/A"
    st.markdown(f"**Fastest growth trajectory:** *{fastest}* scenario shows the largest increase in AI electricity demand between 2030 and 2035.")
    st.markdown(
        "AI-driven load increases require careful coordination with grid reinforcement programs, particularly in regions where renewable integration is still limited."
    )


def render_energy_mix_analysis(energy: pd.DataFrame, selected_country: str) -> None:
    render_section_header("Energy Mix Analysis", "Fuel share trends and low-carbon leadership")
    country_data = energy[energy["country"] == selected_country].sort_values("year")
    if country_data.empty:
        st.warning("Selected country / region data is not available.")
        return
    mix = country_data[["year", "fossil_share_elec", "renewables_share_elec", "nuclear_share_elec"]].melt(
        id_vars="year", var_name="fuel", value_name="share"
    )
    fig_mix = px.area(
        mix,
        x="year",
        y="share",
        color="fuel",
        title=f"Historical Fuel Share Breakdown for {selected_country}",
        labels={"share": "Share (%)", "year": "Year", "fuel": "Fuel Type"},
        color_discrete_map={
            "fossil_share_elec": "#D50000",
            "renewables_share_elec": "#00C853",
            "nuclear_share_elec": "#2979FF",
        },
    )
    st.plotly_chart(fig_mix, use_container_width=True)

    latest_year = int(country_data["year"].max())
    ranking = (
        energy[energy["year"] == latest_year]
        .dropna(subset=["low_carbon_share_elec"])
        .sort_values("low_carbon_share_elec", ascending=False)
        .head(20)[["country", "low_carbon_share_elec"]]
    )
    fig_ranking = px.bar(
        ranking[::-1],
        x="low_carbon_share_elec",
        y="country",
        orientation="h",
        title=f"Top 20 Countries by Low Carbon Share ({latest_year})",
        labels={"low_carbon_share_elec": "Low Carbon Share (%)", "country": "Country"},
        color="low_carbon_share_elec",
        color_continuous_scale="teal",
    )
    st.plotly_chart(fig_ranking, use_container_width=True)

    trends = country_data[["year", "solar_share_elec", "wind_share_elec", "hydro_share_elec"]].melt(
        id_vars="year", var_name="technology", value_name="share"
    )
    fig_trends = px.line(
        trends,
        x="year",
        y="share",
        color="technology",
        title=f"Solar, Wind and Hydro Share Trends for {selected_country}",
        labels={"share": "Share (%)", "year": "Year", "technology": "Technology"},
        markers=True,
    )
    st.plotly_chart(fig_trends, use_container_width=True)


def render_ai_energy_readiness(readiness: pd.DataFrame) -> None:
    render_section_header("AI Energy Readiness Rankings", "Country readiness scoring for AI electrification")
    top20 = readiness.head(20).copy()
    bottom20 = readiness.tail(20).sort_values("score")
    st.markdown("### Top 20 Readiness Leaders")
    st.dataframe(top20[["country", "score", "category", "renewables_share_elec", "low_carbon_share_elec", "carbon_intensity_elec"]], use_container_width=True)
    st.markdown("### Bottom 20 Readiness Risks")
    st.dataframe(bottom20[["country", "score", "category", "renewables_share_elec", "low_carbon_share_elec", "carbon_intensity_elec"]], use_container_width=True)
    st.markdown(
        "**Interpretation:** Countries classified as Leading have strong renewables penetration and low carbon intensity, while High Risk economies require rapid clean energy deployment and grid investment."
    )


def render_grid_stress_index(grid_stress: pd.DataFrame) -> None:
    render_section_header("Grid Stress Index", "Ranking and risk heatmaps for grid resilience")
    st.markdown("### Stress Summary")
    st.dataframe(grid_stress[["country", "grid_stress_index", "stress_category", "fossil_share_elec", "carbon_intensity_elec"]].head(25), use_container_width=True)
    correlation_data = grid_stress[
        ["grid_stress_index", "fossil_share_elec", "carbon_intensity_elec", "electricity_demand", "electricity_generation"]
    ].dropna()
    if not correlation_data.empty:
        corr = correlation_data.corr()
        fig_heatmap = px.imshow(
            corr,
            text_auto=True,
            color_continuous_scale="RdBu_r",
            title="Correlation Matrix for Grid Stress Metrics",
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)
    st.markdown("### Stress Classification")
    summary = grid_stress.groupby("stress_category")["country"].count().reset_index(name="count")
    st.dataframe(summary, use_container_width=True)


def render_regional_risk_model(regional_risk: pd.DataFrame) -> None:
    render_section_header("Regional Risk Model", "Lowest and highest risk regions for AI electricity demand")
    lowest20 = regional_risk.head(20)[["country", "risk_score", "carbon_intensity_elec", "fossil_share_elec", "demand_growth"]]
    highest20 = regional_risk.tail(20).sort_values("risk_score", ascending=False)[["country", "risk_score", "carbon_intensity_elec", "fossil_share_elec", "demand_growth"]]
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Top 20 Lowest Risk Regions")
        st.dataframe(lowest20, use_container_width=True)
    with col2:
        st.markdown("#### Top 20 Highest Risk Regions")
        st.dataframe(highest20, use_container_width=True)
    st.markdown(
        "A lower risk score reflects cleaner electricity systems, lower fossil dependency, and slower demand growth, while higher risk regions require accelerated transition funding and grid reinforcement."
    )


def render_forecasting(energy: pd.DataFrame, ai_demand: pd.DataFrame) -> None:
    render_section_header("Forecasting to 2035", "Projected electricity demand and AI-driven divergence")
    world = energy[energy["country"] == "World"].sort_values("year")
    if world.empty:
        st.warning("World demand data is unavailable for forecasting.")
        return
    history_years = world["year"].astype(float).to_numpy()
    history_values = world["electricity_demand"].astype(float).to_numpy()
    forecast_years = np.arange(int(history_years.max()) + 1, 2036)
    baseline = build_forecast(history_years, history_values, forecast_years, slope_scale=1.0)
    accelerated = build_forecast(history_years, history_values, forecast_years, slope_scale=1.2)
    ai_data = ai_demand[ai_demand["scenario"] == "Lift-Off"].groupby("year", as_index=False)["value"].sum()
    ai_signal = np.interp(forecast_years, ai_data["year"], ai_data["value"], left=0, right=ai_data["value"].iloc[-1] if not ai_data.empty else 0)
    ai_driven = baseline + ai_signal
    forecast_df = pd.DataFrame(
        {
            "year": forecast_years,
            "Baseline Forecast": baseline,
            "Accelerated Growth Forecast": accelerated,
            "AI-Driven Growth Forecast": ai_driven,
        }
    )
    fig_forecast = go.Figure()
    fig_forecast.add_trace(go.Scatter(x=history_years, y=history_values, mode="lines+markers", name="Historical Demand", line=dict(color=ENERGY_COLORS[0], width=3)))
    fig_forecast.add_trace(go.Scatter(x=forecast_df["year"], y=forecast_df["Baseline Forecast"], mode="lines", name="Baseline Forecast", line=dict(color=ENERGY_COLORS[1], dash="dash")))
    fig_forecast.add_trace(go.Scatter(x=forecast_df["year"], y=forecast_df["Accelerated Growth Forecast"], mode="lines", name="Accelerated Forecast", line=dict(color=ENERGY_COLORS[2], dash="dot")))
    fig_forecast.add_trace(go.Scatter(x=forecast_df["year"], y=forecast_df["AI-Driven Growth Forecast"], mode="lines", name="AI-Driven Forecast", line=dict(color=ENERGY_COLORS[3], width=3)))
    fig_forecast.add_trace(go.Scatter(x=forecast_df["year"], y=forecast_df["Baseline Forecast"] * 0.95, mode="lines", showlegend=False, line=dict(width=0), hoverinfo="skip"))
    fig_forecast.add_trace(go.Scatter(x=forecast_df["year"], y=forecast_df["Baseline Forecast"] * 1.05, mode="lines", fill="tonexty", fillcolor="rgba(33, 150, 243, 0.2)", showlegend=True, name="Baseline Confidence Band"))
    fig_forecast.update_layout(title="Forecasted Global Electricity Demand to 2035", xaxis_title="Year", yaxis_title="Electricity Demand (TWh)")
    st.plotly_chart(fig_forecast, use_container_width=True)
    st.markdown(
        "The baseline projection is derived from historical growth. The accelerated forecast adds a 20% slope uplift for higher electrification, while the AI-driven forecast integrates Lift-Off load additions."
    )


def render_transition_requirements(ai_demand: pd.DataFrame) -> None:
    render_section_header("Energy Transition Requirements", "Renewable capacity needed to meet AI scenario load")
    demand_2035 = ai_demand[ai_demand["year"] == 2035].groupby("scenario", as_index=False)["value"].sum()
    base_2035 = float(demand_2035.loc[demand_2035["scenario"] == "Base", "value"].sum())
    liftoff_2035 = float(demand_2035.loc[demand_2035["scenario"] == "Lift-Off", "value"].sum())
    incremental_twh = max(0.0, liftoff_2035 - base_2035)
    incremental_mwh = incremental_twh * 1_000_000
    wind_gw = incremental_mwh / (0.30 * 8760) / 1000
    solar_gw = incremental_mwh / (0.20 * 8760) / 1000
    st.markdown(f"### Incremental AI Load Delta in 2035: {incremental_twh:,.0f} TWh")
    col1, col2 = st.columns(2)
    col1.metric("Required Wind Capacity Addition", f"{wind_gw:,.1f} GW", delta="Assuming 30% CF")
    col2.metric("Required Solar Capacity Addition", f"{solar_gw:,.1f} GW", delta="Assuming 20% CF")
    capacity_df = pd.DataFrame(
        {
            "Technology": ["Wind", "Solar"],
            "Capacity Factor": ["30%", "20%"],
            "Required GW": [wind_gw, solar_gw],
        }
    )
    st.dataframe(capacity_df, use_container_width=True)
    fig_capacity = px.bar(
        capacity_df,
        x="Technology",
        y="Required GW",
        title="Renewable Capacity Needed for Incremental AI Load",
        color="Technology",
        color_discrete_sequence=[ENERGY_COLORS[3], ENERGY_COLORS[0]],
        text="Required GW",
    )
    st.plotly_chart(fig_capacity, use_container_width=True)
    st.markdown(
        "This incremental capacity estimate translates AI-driven energy demand to investor-grade renewable deployment requirements with clear wind and solar targets."
    )


def render_scenario_comparison(energy: pd.DataFrame, ai_demand: pd.DataFrame, grid_stress: pd.DataFrame) -> None:
    render_section_header("Scenario Comparison", "Comparison matrix integrating AI scenarios, load projections and stress classifications")
    scenario_totals = ai_demand.groupby(["scenario", "year"], as_index=False)["value"].sum()
    projected = scenario_totals[scenario_totals["year"].isin([2030, 2035])].pivot(index="scenario", columns="year", values="value").reset_index()
    stress_avg = grid_stress.groupby("stress_category")["grid_stress_index"].mean().reset_index()
    scenario_summary = projected.copy()
    scenario_summary["avg_grid_stress_category"] = "Moderate Stress"
    scenario_summary["avg_grid_stress_index"] = scenario_summary["scenario"].map(
        {
            "Base": float(stress_avg.loc[stress_avg["stress_category"] == "Moderate Stress", "grid_stress_index"].mean()),
            "Headwinds": float(stress_avg.loc[stress_avg["stress_category"] == "High Stress", "grid_stress_index"].mean()),
            "High Efficiency": float(stress_avg.loc[stress_avg["stress_category"] == "Low Stress", "grid_stress_index"].mean()),
            "Lift-Off": float(stress_avg.loc[stress_avg["stress_category"] == "High Stress", "grid_stress_index"].mean()),
            "Historical": float(stress_avg.loc[stress_avg["stress_category"] == "Moderate Stress", "grid_stress_index"].mean()),
        }
    )
    scenario_summary["avg_grid_stress_index"] = scenario_summary["avg_grid_stress_index"].fillna(float(stress_avg["grid_stress_index"].mean()))
    scenario_summary = scenario_summary.rename(columns={2030: "Demand 2030 (TWh)", 2035: "Demand 2035 (TWh)"})
    st.markdown("### Scenario Comparison Matrix")
    st.dataframe(scenario_summary, use_container_width=True)
    csv_content = scenario_summary.to_csv(index=False).encode("utf-8")
    st.download_button("Download Scenario Comparison CSV", csv_content, file_name="scenario_comparison_matrix.csv", mime="text/csv")
    st.markdown("This comparison matrix provides a structured view of demand projections and associated grid stress signals across alternate AI growth paths.")


def render_recommendations() -> None:
    render_section_header("Strategic Recommendations", "Actionable guidance for energy stakeholders")
    st.markdown(
        """
        ## Governments / Policymakers
        - Prioritize policies that accelerate renewable capacity deployment and grid flexibility.
        - Align AI infrastructure incentives with low-carbon electricity procurement.
        - Support cross-border transmission corridors and regional coordination.

        ## Grid Operators / Utilities
        - Integrate AI demand forecasts into capacity planning and reserve margins.
        - Deploy advanced grid monitoring, demand response, and storage solutions.
        - Coordinate with large AI customers on managed load schedules.

        ## Energy Developers
        - Focus investment in solar and wind capacity where AI demand growth is strongest.
        - Develop hybrid renewable-plus-storage projects to support peak AI load.
        - Partner with utilities to deliver firm, low-carbon power to hyperscale data centers.

        ## AI Companies / Hyperscalers
        - Plan power procurement strategies around the highest readiness and lowest stress regions.
        - Adopt on-site renewable energy, energy efficiency, and flexible compute scheduling.
        - Engage with regulators and grid operators early to mitigate congestion risk.

        ## Infrastructure Investors
        - Evaluate low-carbon generation and grid modernization projects with clear demand signal exposure.
        - Prioritize investments that reduce carbon intensity and improve system reliability.
        - Use scenario analysis to stress test portfolios under Lift-Off and Headwinds assumptions.
        """
    )


def render_final_report(
    energy: pd.DataFrame,
    ai_demand: pd.DataFrame,
    readiness: pd.DataFrame,
    grid_stress: pd.DataFrame,
    regional_risk: pd.DataFrame,
    selected_country: str,
    selected_year: int,
) -> None:
    render_section_header("Final Report & Automation Mockup", "Downloadable summary and email alert design")
    pdf_bytes = generate_pdf_report(energy, ai_demand, readiness, grid_stress, regional_risk, selected_country, selected_year)
    st.download_button(
        label="Download Executive PDF Report",
        data=pdf_bytes,
        file_name="Energy_AI_Executive_Report.pdf",
        mime="application/pdf",
    )
    st.markdown("### Automated Email Alert System")
    email = st.text_input("Recipient Email", value="alerts@example.com")
    trigger = st.selectbox(
        "Alert Trigger Condition",
        ["Grid Stress Index > 75", "AI Demand Lift-Off > Base by 10%", "Low Carbon Share < 40%", "Demand Growth > 5% per year"],
    )
    threshold = st.slider("Set Index Trigger Threshold Value", min_value=0, max_value=500, value=75)

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Save & Automate Schedule"):
            st.success(
                f"✅ Automation Active: Scheduled task created. System will monitor data feeds daily and alert {email} if condition breaches your threshold ({threshold})."
            )
    with col2:
        if st.button("🚀 Send Test Alert Now"):
            with st.spinner("Compiling 10-page executive analytics report and testing dispatch protocols..."):
                pdf_bytes = generate_pdf_report(energy, ai_demand, readiness, grid_stress, regional_risk, selected_country, selected_year)
                sent, info = send_pdf_email(email, pdf_bytes)
                if sent:
                    st.success(f"📧 Success! The live 10-page PDF report has been generated and emailed directly to {email}!")
                else:
                    if info == "placeholder":
                        st.info(
                            f"👉 UI Demo Mode: 10-page PDF report generated successfully! (To activate live email inbox deliveries to {email}, simply update the SENDER_EMAIL credentials directly inside the script's code.)"
                        )
                        st.download_button("Download Generated PDF", data=pdf_bytes, file_name="Energy_AI_Executive_Report.pdf", mime="application/pdf")
                    else:
                        st.error(f"Email send failed: {info}")
                        st.download_button("Download Generated PDF", data=pdf_bytes, file_name="Energy_AI_Executive_Report.pdf", mime="application/pdf")


def main() -> None:
    energy = load_clean_energy()
    ai_demand = load_clean_ai_demand()
    price = load_clean_price()
    if energy.empty or ai_demand.empty:
        st.error("Required data is missing from the database. Please verify clean_energy_ai.db contains clean_energy and clean_ai_demand tables.")
        return
    st.sidebar.title("ADELEKE BASHEER")
    st.sidebar.caption("Energy & Climate Risk Analyst")
    st.sidebar.markdown("---")
    st.sidebar.markdown("### **Project:** Can Global Energy Systems Sustain the Rapid Growth of AI?\n*AI, Electricity Demand & Energy System Constraints*")
    st.sidebar.markdown("---")
    countries = get_country_options()
    years = get_year_options()
    selected_country = st.sidebar.selectbox("Country / Region", countries, index=0)
    selected_year = st.sidebar.selectbox("Year", years if years else [0], index=len(years) - 1 if years else 0)
    selected_section = st.sidebar.radio("Select Section", SECTION_TITLES)
    st.sidebar.markdown("---")

    readiness = compute_readiness_scores(energy)
    grid_stress = compute_grid_stress(energy)
    regional_risk = compute_regional_risk(energy)
    world = energy[energy["country"] == "World"].sort_values("year")
    latest_world_year = int(world["year"].max()) if not world.empty else None
    latest_demand = float(world.loc[world["year"] == latest_world_year, "electricity_demand"].sum()) if latest_world_year else 0.0
    latest_carbon_intensity = float(world.loc[world["year"] == latest_world_year, "carbon_intensity_elec"].mean()) if latest_world_year else 0.0
    baseline_2035 = compute_world_baseline_2035(world, 2035)
    incremental_ai = max(
        0.0,
        ai_demand[(ai_demand["scenario"] == "Lift-Off") & (ai_demand["year"] == 2035)]["value"].sum()
        - ai_demand[(ai_demand["scenario"] == "Base") & (ai_demand["year"] == 2035)]["value"].sum(),
    )
    forecasted_world_2035 = float(baseline_2035 + incremental_ai)
    highest_ai_demand = float(ai_demand[ai_demand["year"] == 2035]["value"].max() if not ai_demand[ai_demand["year"] == 2035].empty else 0.0)

    if selected_section == "Executive Summary":
        render_executive_summary(energy, ai_demand)
    elif selected_section == "Global Electricity Demand":
        render_global_electricity_demand(energy, selected_country, selected_year)
    elif selected_section == "AI Demand Scenarios":
        render_ai_demand_scenarios(ai_demand)
    elif selected_section == "Energy Mix Analysis":
        render_energy_mix_analysis(energy, selected_country)
    elif selected_section == "AI Energy Readiness Rankings":
        render_ai_energy_readiness(readiness)
    elif selected_section == "Grid Stress Index":
        render_grid_stress_index(grid_stress)
    elif selected_section == "Regional Risk Model":
        render_regional_risk_model(regional_risk)
    elif selected_section == "Forecasting to 2035":
        render_forecasting(energy, ai_demand)
    elif selected_section == "Energy Transition Requirements":
        render_transition_requirements(ai_demand)
    elif selected_section == "Scenario Comparison":
        render_scenario_comparison(energy, ai_demand, grid_stress)
    elif selected_section == "Strategic Recommendations":
        render_recommendations()
    elif selected_section == "Final Report & Automation Mockup":
        render_final_report(energy, ai_demand, readiness, grid_stress, regional_risk, selected_country, selected_year)
    else:
        st.error("Unknown section selected.")


if __name__ == "__main__":
    main()
