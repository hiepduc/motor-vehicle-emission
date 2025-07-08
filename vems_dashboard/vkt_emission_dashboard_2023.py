import streamlit as st
import xarray as xr
import numpy as np
import pandas as pd
import plotly.express as px
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium

# App setup
st.set_page_config(layout="wide", page_title="2023 VKT & Emissions Viewer")
st.title("Motor Vehicle Model Dashboard — 2023")
st.sidebar.title("2023 Viewer")

# --- File paths ---
VKT_FILE = "/home/duch/mvems/vkt/vkt_nsw2023_hourly_gridded_latlon.nc"
EMISSION_FILES = {
    "HOT": "/home/duch/mvems/emission/2023/nsw/hot_nsw_x0x1x2.nc",
    "COLD": "/home/duch/mvems/emission/2023/nsw/cold_x2y0_x2y1.nc",
    "NEPM": "/home/duch/mvems/emission/2023/nsw/nepm_nsw_x0x1x2.nc",
    "EVAPS": "/home/duch/mvems/emission/2023/nsw/evaps_nsw_x0x1x2.nc"
}

# --- Selection Mode ---
mode = st.sidebar.radio("Select Data Type", ["VKT", "Emissions"])

# --- Load Dataset ---
@st.cache_resource
def load_dataset(path):
    return xr.open_dataset(path)

if mode == "VKT":
    ds = load_dataset(VKT_FILE)
    var_map = {
        "Passenger Car (PC)": "VKT_cat1",
        "Light Commercial Vehicle (LCV)": "VKT_cat3",
        "Rigid Trucks (RIG)": "VKT_cat11",
        "Articulated Trucks (ART)": "VKT_cat12",
        "Bus": "VKT_cat13",
        "Motorcycle (MC)": "VKT_cat20"
    }
    options = [k for k in var_map if var_map[k] in ds.data_vars]
    var_name_readable = st.sidebar.selectbox("Select Vehicle Category", options)
    variable = var_map[var_name_readable]
    legend_label = var_name_readable
    time_dim = "time"
    lat = ds["lat"].values
    lon = ds["lon"].values
    time = ds[time_dim].values

else:
    emission_type = st.sidebar.selectbox("Emission Source", list(EMISSION_FILES.keys()))
    ds = load_dataset(EMISSION_FILES[emission_type])
    variable = st.sidebar.selectbox("Select Emission Variable", list(ds.data_vars))
    legend_label = variable
    time_dim = "Time"
    lat = ds["lat"].values
    lon = ds["lon"].values
    time = ds[time_dim].values
    month_sel = st.sidebar.selectbox("Month", list(range(1, 13)))
    day_sel = st.sidebar.selectbox("Day Type", ["Weekday", "Weekend"])
    wdwe = 1 if day_sel == "Weekday" else 2
    #st.sidebar.markdown("Month = January, Weekday = 1 assumed")
    #ds = ds.sel(Month=1, WD_WE_ID=1)
    ds = ds.sel(Month=month_sel, WD_WE_ID=wdwe)

# --- Hour Selection ---
hour = st.sidebar.slider("Select Hour", 0, len(time)-1, 0)

# --- Spatial Map ---
lat_grid, lon_grid = np.meshgrid(lat, lon, indexing="ij")
data = ds[variable].isel({time_dim: hour}).values

# Summary stats
total_val = np.nansum(data)
mean_val = np.nanmean(data)

st.sidebar.markdown("---")
st.sidebar.subheader("Regional Summary")
st.sidebar.metric("Total (This Hour)", f"{total_val:,.4f}")
st.sidebar.metric("Mean Value", f"{mean_val:,.4f}")

# Line plot (region-wide over 24h)
if mode == "VKT":
    total_region = ds[variable].sum(dim=("lat", "lon"), skipna=True).values
else:
    total_region = ds[variable].sum(dim=("lat", "lon"), skipna=True).values

fig_line = px.line(x=np.arange(0, 24), y=total_region, markers=True,
                   labels={"x": "Hour", "y": f"Total {legend_label}"})
fig_line.update_layout(height=250, margin=dict(l=0, r=0, t=30, b=0))
st.sidebar.plotly_chart(fig_line, use_container_width=True)

# --- Map Display ---
st.subheader(f"{legend_label} | Hour: {hour}")
map_df = pd.DataFrame({
    "lat": lat_grid.flatten(),
    "lon": lon_grid.flatten(),
    "value": data.flatten()
}).dropna()

m = folium.Map(location=[-33.8, 151.2], zoom_start=7)
HeatMap(map_df[["lat", "lon", "value"]].values.tolist(), radius=8).add_to(m)

col1, col2 = st.columns([2, 1])
with col1:
    click_data = st_folium(m, height=600)

with col2:
    if click_data and click_data.get("last_clicked"):
        lat_c = click_data["last_clicked"]["lat"]
        lon_c = click_data["last_clicked"]["lng"]
        st.markdown(f"**Lat:** `{lat_c:.4f}`, **Lon:** `{lon_c:.4f}`")

        dist = np.sqrt((lat_grid - lat_c) ** 2 + (lon_grid - lon_c) ** 2)
        n_idx, e_idx = np.unravel_index(np.argmin(dist), dist.shape)

        time_series = ds[variable].isel(lat=n_idx, lon=e_idx).values

        st.subheader("Time Series")
        fig_ts = px.line(x=np.arange(0, 24), y=time_series, markers=True,
                         labels={"x": "Hour", "y": legend_label})
        st.plotly_chart(fig_ts, use_container_width=True)

        st.markdown("### Summary")
        st.markdown(f"- Total 24h: `{np.nansum(time_series):.4f}`")
        st.markdown(f"- Mean hourly: `{np.nanmean(time_series):.4f}`")

        ts_df = pd.DataFrame({"Hour": np.arange(0, 24), variable: time_series})
        st.download_button(
            "Download Time Series CSV",
            ts_df.to_csv(index=False),
            file_name=f"{variable}_ts_lat{lat_c:.2f}_lon{lon_c:.2f}_2023.csv",
            mime="text/csv"
        )
    else:
        st.info("Click on the map to view time series for a specific location.")

