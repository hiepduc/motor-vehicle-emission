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
st.sidebar.title("Vehicle Kilometers Traveled (VKT) Viewer")

# --- User selections ---
year_options = ["2013", "2021", "2023", "Compare 2013 vs 2021"]
selected_mode = st.sidebar.selectbox("Select Mode", year_options)

# --- File paths ---
file_paths = {
    "2013": {
        "Cold Start (CS)": "/home/duch/mvems/vkt/2013/vkt_CS_hourly_utm_1km.nc",
        "Hot Start (Hot)": "/home/duch/mvems/vkt/2013/vkt_hourly_utm_1km.nc"
    },
    "2021": {
        "Cold Start (CS)": "/home/duch/mvems/vkt/2021/vkt_2021_CS_hourly_utm_1km.nc",
        "Hot Start (Hot)": "/home/duch/mvems/vkt/2021/vkt_2021_hourly_utm_1km.nc"
    },
    "2023": "/home/duch/mvems/vkt/vkt_nsw2023_hourly_gridded_latlon.nc"
}

@st.cache_resource
def load_dataset(path):
    return xr.open_dataset(path)

transformer = Transformer.from_crs("EPSG:32756", "EPSG:4326", always_xy=True)

@st.cache_data
def build_latlon_grid(eastings, northings):
    xx, yy = np.meshgrid(eastings, northings)
    lon, lat = transformer.transform(xx.flatten(), yy.flatten())
    return lat.reshape(xx.shape), lon.reshape(xx.shape)

# --- Mode: Comparison ---
if selected_mode == "Compare 2013 vs 2021":
    file_option = st.sidebar.selectbox("Select emission type", ["Cold Start (CS)", "Hot Start (Hot)"])

    ds_2013 = load_dataset(file_paths["2013"][file_option])
    ds_2021 = load_dataset(file_paths["2021"][file_option])

    common_vars = list(set(ds_2013.data_vars).intersection(set(ds_2021.data_vars)))
    if not common_vars:
        st.error("No common variables between 2013 and 2021 datasets.")
        st.stop()

    variable = st.sidebar.selectbox("Select Variable", common_vars)
    time_index = st.sidebar.slider("Select hour", 0, len(ds_2013.time) - 1, 0)

    # Hourly regional total
    hourly_2013 = ds_2013[variable].sum(dim=("northing", "easting"), skipna=True).values
    hourly_2021 = ds_2021[variable].sum(dim=("northing", "easting"), skipna=True).values
    hourly_diff = hourly_2021 - hourly_2013

    total_diff_day = np.nansum(hourly_diff)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Total VKT Difference (2021 - 2013)")
    st.sidebar.metric("At Selected Hour", f"{hourly_diff[time_index]:,.2f} vehicle-km")
    st.sidebar.metric("Total Difference (24h)", f"{total_diff_day:,.2f} vehicle-km")

    # Line plot
    st.sidebar.markdown("**Hourly VKT Comparison**")
    df_compare = pd.DataFrame({
        "Hour": ds_2013["time"].values,
        "2013": hourly_2013,
        "2021": hourly_2021,
        "Difference": hourly_diff
    })
    fig_diff = px.line(df_compare, x="Hour", y=["2013", "2021", "Difference"],
                       labels={"value": "Total VKT", "variable": "Year"}, markers=True)
    fig_diff.update_layout(height=300, margin=dict(l=0, r=0, t=30, b=0))
    st.sidebar.plotly_chart(fig_diff, use_container_width=True)

    # Download CSV
    st.sidebar.download_button(
        "Download hourly comparison CSV",
        df_compare.to_csv(index=False),
        file_name=f"vkt_hourly_diff_2021_minus_2013_{variable}.csv",
        mime="text/csv"
    )

    # Spatial difference map
    diff_map = ds_2021[variable].isel(time=time_index) - ds_2013[variable].isel(time=time_index)
    lat_grid, lon_grid = build_latlon_grid(ds_2013["easting"].values, ds_2013["northing"].values)

    heatmap_df = pd.DataFrame({
        "lat": lat_grid.flatten(),
        "lon": lon_grid.flatten(),
        "value": diff_map.values.flatten()
    }).dropna()

    m = folium.Map(location=[-33.8, 151.2], zoom_start=9)
    HeatMap(heatmap_df[["lat", "lon", "value"]].values.tolist(), radius=8).add_to(m)

    st.title("VKT Comparison: 2021 vs 2013")
    st.subheader(f"{variable} | Hour: {time_index} | Spatial Difference (2021 - 2013)")
    st_folium(m, height=600)

    st.stop()

# --- Single-year mode ---
selected_year = selected_mode
if selected_year in ["2013", "2021"]:
    file_option = st.sidebar.selectbox("Select emission type", ["Cold Start (CS)", "Hot Start (Hot)"])
    file_path = file_paths[selected_year][file_option]
else:
    file_path = file_paths["2023"]

ds = load_dataset(file_path)
var_names = list(ds.data_vars)

# Variable/Vehicle category
if selected_year == "2023":
    vehicle_mapping = {
        "Passenger Car (PC)": "VKT_cat1",
        "Light Commercial Vehicle (LCV)": "VKT_cat3",
        "Rigid Trucks (RIG)": "VKT_cat11",
        "Articulated Trucks (ART)": "VKT_cat12",
        "Motorcycle (MC)": "VKT_cat20"
    }
    readable_names = [k for k in vehicle_mapping if vehicle_mapping[k] in var_names]
    selected_readable = st.sidebar.selectbox("Select Vehicle Category", readable_names)
    variable = vehicle_mapping[selected_readable]
    legend_label = selected_readable
else:
    variable = st.sidebar.selectbox("Select Variable", var_names)
    legend_label = variable

time_index = st.sidebar.slider("Select hour", 0, len(ds.time) - 1, 0)

# Regional summary
if selected_year == "2023":
    total_vkt = ds[variable].sum(dim=("lat", "lon"), skipna=True).values
    mean_vkt = ds[variable].mean(dim=("lat", "lon"), skipna=True).values
else:
    total_vkt = ds[variable].sum(dim=("northing", "easting"), skipna=True).values
    mean_vkt = ds[variable].mean(dim=("northing", "easting"), skipna=True).values

time_hours = ds["time"].values
total_vkt_day = np.nansum(total_vkt)
mean_vkt_day = np.nanmean(mean_vkt)

# Sidebar stats
st.sidebar.markdown("---")
st.sidebar.subheader("Daily Summary (Entire Region)")
st.sidebar.metric("Total VKT (24h)", f"{total_vkt_day:,.2f}")
st.sidebar.metric("Mean Hourly VKT", f"{mean_vkt_day:,.2f}")

fig_region = px.line(x=time_hours, y=total_vkt, labels={"x": "Hour", "y": "Total VKT"}, markers=True)
fig_region.update_layout(height=250, margin=dict(l=0, r=0, t=30, b=0))
st.sidebar.markdown("**Total VKT per Hour (Region)**")
st.sidebar.plotly_chart(fig_region, use_container_width=True)

summary_df = pd.DataFrame({"Hour": time_hours, "Total_VKT": total_vkt, "Mean_VKT": mean_vkt})
st.sidebar.download_button(
    "Download hourly summary CSV",
    summary_df.to_csv(index=False),
    file_name=f"{variable}_region_summary_{selected_year}.csv",
    mime="text/csv"
)

# Grid lat/lon
if selected_year == "2023":
    lat_vals = ds["lat"].values
    lon_vals = ds["lon"].values
    lat_grid, lon_grid = np.meshgrid(lat_vals, lon_vals, indexing="ij")
else:
    easting_vals = ds["easting"].values
    northing_vals = ds["northing"].values
    lat_grid, lon_grid = build_latlon_grid(easting_vals, northing_vals)

data_slice = ds[variable].isel(time=time_index).values
heatmap_df = pd.DataFrame({
    "lat": lat_grid.flatten(),
    "lon": lon_grid.flatten(),
    "value": data_slice.flatten()
}).dropna()

m = folium.Map(location=[-33.8, 151.2], zoom_start=7 if selected_year == "2023" else 9)
HeatMap(heatmap_df[["lat", "lon", "value"]].values.tolist(), radius=8).add_to(m)

st.title("Motor Vehicle Emission Model (VEMS) VKT dashboard")
st.subheader(f"{legend_label} | Hour: {time_index}")
col1, col2 = st.columns([2, 1])

with col1:
    click_data = st_folium(m, height=600)

with col2:
    if click_data and click_data.get("last_clicked"):
        clicked_lat = click_data["last_clicked"]["lat"]
        clicked_lon = click_data["last_clicked"]["lng"]
        st.markdown(f"**Lat:** `{clicked_lat:.4f}`, **Lon:** `{clicked_lon:.4f}`")

        dist = np.sqrt((lat_grid - clicked_lat) ** 2 + (lon_grid - clicked_lon) ** 2)
        n_idx, e_idx = np.unravel_index(np.argmin(dist), dist.shape)

        time_series = ds[variable][:, n_idx, e_idx].values
        st.subheader("Time Series")
        fig = px.line(x=time_hours, y=time_series, markers=True, labels={"x": "Hour", "y": variable})
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Summary Statistics")
        st.markdown(f"- **Total VKT (24h):** `{np.nansum(time_series):.3f}` vehicle-km")
        st.markdown(f"- **Average Hourly VKT:** `{np.nanmean(time_series):.3f}` vehicle-km")

        ts_df = pd.DataFrame({"Hour": time_hours, variable: time_series})
        st.download_button(
            "Download time series CSV",
            ts_df.to_csv(index=False),
            file_name=f"{variable}_timeseries_lat{clicked_lat:.2f}_lon{clicked_lon:.2f}_{selected_year}.csv",
            mime="text/csv"
        )
    else:
        st.info("Click on the map to see time series at that location.")

