import streamlit as st
import xarray as xr
import pandas as pd
import numpy as np
import plotly.express as px
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium

st.set_page_config(layout="wide", page_title="NSW VKT Viewer")
st.sidebar.title("VKT  2023")

# File path for 2023 data
file_path = "/home/duch/mvems/vkt/vkt_nsw2023_hourly_gridded_latlon.nc"

@st.cache_resource
def load_dataset(path):
    return xr.open_dataset(path)

ds = load_dataset(file_path)
var_names = list(ds.data_vars)
variable = st.sidebar.selectbox("Select variable", var_names)
time_index = st.sidebar.slider("Select hour", 0, len(ds.time) - 1, 0)

# --- Hourly Total and Mean VKT over NSW ---
hourly_total_vkt = ds[variable].sum(dim=("lat", "lon"), skipna=True).values
hourly_mean_vkt = ds[variable].mean(dim=("lat", "lon"), skipna=True).values
time_hours = ds["time"].values

# Daily summary
total_vkt_day = np.nansum(hourly_total_vkt)
mean_vkt_day = np.nanmean(hourly_mean_vkt)

# Display in sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("Daily Summary (NSW Total)")
st.sidebar.metric("Total VKT (24h)", f"{total_vkt_day:,.2f}")
st.sidebar.metric("Mean Hourly VKT", f"{mean_vkt_day:,.2f}")

# Plot time series (sidebar)
st.sidebar.markdown("**Total VKT per Hour (NSW)**")
fig_region = px.line(
    x=time_hours,
    y=hourly_total_vkt,
    labels={"x": "Hour", "y": "Total VKT"},
    markers=True
)
fig_region.update_layout(height=250, margin=dict(l=0, r=0, t=30, b=0))
st.sidebar.plotly_chart(fig_region, use_container_width=True)

# Optional download
summary_df = pd.DataFrame({
    "Hour": time_hours,
    "Total_VKT": hourly_total_vkt,
    "Mean_VKT": hourly_mean_vkt
})
st.sidebar.download_button(
    "Download hourly summary CSV",
    summary_df.to_csv(index=False),
    file_name=f"{variable}_NSW_summary.csv",
    mime="text/csv"
)

# Lat/Lon and slice for selected hour
lat_vals = ds["lat"].values
lon_vals = ds["lon"].values
data_slice = ds[variable].isel(time=time_index).values

# Prepare data for map
lat_grid, lon_grid = np.meshgrid(lat_vals, lon_vals, indexing="ij")
heatmap_df = pd.DataFrame({
    "lat": lat_grid.flatten(),
    "lon": lon_grid.flatten(),
    "value": data_slice.flatten()
}).dropna()

# Folium map centered on NSW
m = folium.Map(location=[-33.5, 147.0], zoom_start=6)
HeatMap(heatmap_df[["lat", "lon", "value"]].values.tolist(), radius=8).add_to(m)

# Layout
st.title("Motor Vehicle VKT Viewer (NSW 2023)")
st.subheader(f"Variable: {variable} | Hour: {time_index}")
col1, col2 = st.columns([2, 1])

with col1:
    click_data = st_folium(m, height=600)

with col2:
    if click_data and click_data.get("last_clicked"):
        clicked_lat = click_data["last_clicked"]["lat"]
        clicked_lon = click_data["last_clicked"]["lng"]
        st.markdown(f"**Lat:** `{clicked_lat:.4f}`, **Lon:** `{clicked_lon:.4f}`")

        # Find nearest index
        dist = np.sqrt((lat_grid - clicked_lat)**2 + (lon_grid - clicked_lon)**2)
        n_idx, e_idx = np.unravel_index(np.argmin(dist), dist.shape)

        # Time series at that cell
        time_series = ds[variable][:, n_idx, e_idx].values

        st.subheader("Time Series")
        fig = px.line(x=time_hours, y=time_series, markers=True, labels={"x": "Hour", "y": variable})
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Summary Stats")
        st.markdown(f"- **Total VKT (24h):** `{np.nansum(time_series):.2f}`")
        st.markdown(f"- **Avg Hourly VKT:** `{np.nanmean(time_series):.2f}`")

        ts_df = pd.DataFrame({"Hour": time_hours, variable: time_series})
        st.download_button(
            "Download time series CSV",
            ts_df.to_csv(index=False),
            file_name=f"{variable}_lat{clicked_lat:.2f}_lon{clicked_lon:.2f}.csv",
            mime="text/csv"
        )
    else:
        st.info("Click on the map to see time series at a location.")

