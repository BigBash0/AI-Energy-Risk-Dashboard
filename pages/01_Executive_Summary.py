import streamlit as st

from utils.load_data import (
    load_database,
    load_outputs
)

energy, ai, price = load_database()

outputs = load_outputs()

st.title("Executive Summary")

world = energy[
    energy["country"] == "World"
]

latest = world.sort_values(
    "year"
).iloc[-1]

col1,col2,col3 = st.columns(3)

col1.metric(
    "World Demand",
    f"{latest['electricity_demand']:,.0f} TWh"
)

col2.metric(
    "Carbon Intensity",
    f"{latest['carbon_intensity_elec']:,.0f}"
)

col3.metric(
    "Latest Year",
    int(latest["year"])
)

st.dataframe(
    world.tail()
)