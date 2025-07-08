import streamlit as st
import xarray as xr
import pandas as pd
import numpy as np
import plotly.express as px
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
from pyproj import Transformer

st.set_page_config(layout="wide", page_title="VKT Viewer")
st.sidebar.title("VKT mode 2013")

# File selection
file_option = st.sidebar.selectbox("Select emission type", ["Cold Start (CS)", "Hot Start (Hot)"])
file_path = {
    "Cold Start (CS)": "/home/duch/mvems/vkt/2013/vkt_CS_hourly_utm_1km.nc",
    "Hot Start (Hot)": "/home/duch/mvems/vkt/2013/vkt_hourly_utm_1km.nc"
}[file_option]

@st.cache_resource
def load_dataset(path):
    return xr.open_dataset(path)

ds = load_dataset(file_path)
var_names = list(ds.data_vars)
variable = st.sidebar.selectbox("Select variable", var_names)
time_index = st.sidebar.slider("Select hour", 0, len(ds.time) - 1, 0)

# --- Hourly Regional Summary (Total and Mean VKT over grid) ---

# Total VKT per hour across all grid cells
hourly_total_vkt = ds[variable].sum(dim=("northing", "easting"), skipna=True).values

# Mean VKT per hour across all grid cells
hourly_mean_vkt = ds[variable].mean(dim=("northing", "easting"), skipna=True).values

time_hours = ds["time"].values

# Display as line plot
st.subheader("Hourly Total and Mean VKT Over Entire Domain")

fig = px.line(
    x=time_hours,
    y=[hourly_total_vkt, hourly_mean_vkt],
    labels={"x": "Hour", "value": "VKT"},
    markers=True
)
fig.update_layout(
    title="Total and Mean Hourly VKT (Entire Region)",
    legend=dict(title="Statistic"),
)
fig.data[0].name = "Total VKT"
fig.data[1].name = "Mean VKT"
st.plotly_chart(fig, use_container_width=True)

# Build DataFrame for CSV
summary_df = pd.DataFrame({
    "Hour": time_hours,
    "Total_VKT": hourly_total_vkt,
    "Mean_VKT": hourly_mean_vkt
})

# Download button
st.download_button(
    "Download hourly VKT summary (entire region)",
    summary_df.to_csv(index=False),
    file_name=f"{variable}_region_summary.csv",
    mime="text/csv"
)

# --- Summary Statistics for the whole domain (entire NSW or UTM grid) ---

# Total daily VKT (sum across time and grid)
#total_vkt_region = float(ds[variable].sum(dim=("time", "northing", "easting"), skipna=True).values)

# Average hourly VKT across all cells (mean over time and grid)
#average_vkt_region = float(ds[variable].mean(dim=("time", "northing", "easting"), skipna=True).values)

# Display summary
#st.sidebar.markdown("### Regional Summary Statistics")
#st.sidebar.markdown(f"- **Total Daily VKT (All Cells):** `{total_vkt_region:,.2f}` vehicle-km")
#st.sidebar.markdown(f"- **Average Hourly VKT (All Cells):** `{average_vkt_region:,.2f}` vehicle-km")

easting_vals = ds["easting"].values
northing_vals = ds["northing"].values

# UTM → Lat/Lon
transformer = Transformer.from_crs("EPSG:32756", "EPSG:4326", always_xy=True)

@st.cache_data
def build_latlon_grid(eastings, northings):
    xx, yy = np.meshgrid(eastings, northings)
    lon, lat = transformer.transform(xx.flatten(), yy.flatten())
    return lat.reshape(xx.shape), lon.reshape(xx.shape)

lat_grid, lon_grid = build_latlon_grid(easting_vals, northing_vals)
data_slice = ds[variable].isel(time=time_index).values

# Prepare DataFrame
heatmap_df = pd.DataFrame({
    "lat": lat_grid.flatten(),
    "lon": lon_grid.flatten(),
    "value": data_slice.flatten()
}).dropna()

# Folium map
m = folium.Map(location=[-33.8, 151.2], zoom_start=8)

# Heatmap data as list of [lat, lon, weight]
heat_data = heatmap_df[["lat", "lon", "value"]].values.tolist()
HeatMap(heat_data, radius=10).add_to(m)

# Display title
st.title("Motor Vehicle VKT Viewer")
st.subheader(f"Variable: {variable} | Hour: {time_index}")

# Two-column layout: Map on left, Time series on right
col1, col2 = st.columns([2, 1])  # wider map

with col1:
    # Map display
    click_data = st_folium(m, height=600)

with col2:
    if click_data and click_data.get("last_clicked"):
        clicked_lat = click_data["last_clicked"]["lat"]
        clicked_lon = click_data["last_clicked"]["lng"]

        st.markdown(f"**Lat:** `{clicked_lat:.4f}`, **Lon:** `{clicked_lon:.4f}`")

        # Nearest grid index
        dist = np.sqrt((lat_grid - clicked_lat)**2 + (lon_grid - clicked_lon)**2)
        n_idx, e_idx = np.unravel_index(np.argmin(dist), dist.shape)

        time_series = ds[variable][:, n_idx, e_idx].values
        time_hours = ds["time"].values

        st.subheader("Time Series")
        fig = px.line(x=time_hours, y=time_series, markers=True, labels={"x": "Hour", "y": variable})
        st.plotly_chart(fig, use_container_width=True)
        # ✅ Summary statistics
        total_vkt = np.nansum(time_series)
        average_vkt = np.nanmean(time_series)

        st.markdown("### Summary Statistics")
        st.markdown(f"- **Total VKT (24h):** `{total_vkt:.3f}` vehicle-km")
        st.markdown(f"- **Average Hourly VKT:** `{average_vkt:.3f}` vehicle-km")

        ts_df = pd.DataFrame({"Hour": time_hours, variable: time_series})
        st.download_button(
            "Download CSV",
            ts_df.to_csv(index=False),
            file_name=f"{variable}_timeseries_lat{clicked_lat:.2f}_lon{clicked_lon:.2f}.csv",
            mime="text/csv"
        )
    else:
        st.info("Click on the map to see time series at that location.")

