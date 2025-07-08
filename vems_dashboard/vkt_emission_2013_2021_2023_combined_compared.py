import streamlit as st
import xarray as xr
import pandas as pd
import numpy as np
import plotly.express as px
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
from pyproj import Transformer

st.set_page_config(layout="wide", page_title="Motor Vehicle VKT & Emissions Dashboard")
st.sidebar.title("Motor Vehicle Dashboard (VKT + Emissions)")

# --- Mode Selection ---
year_options = ["2013", "2021", "2023", "Compare 2013 vs 2021"]
selected_mode = st.sidebar.selectbox("Select Mode", year_options)

# --- File Paths ---
vkt_paths = {
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

emissions_2023 = {
    "HOT": "/home/duch/mvems/emission/2023/nsw/hot_nsw_x0x1x2.nc",
    "COLD": "/home/duch/mvems/emission/2023/nsw/cold_x2y0_x2y1.nc",
    "NEPM": "/home/duch/mvems/emission/2023/nsw/nepm_nsw_x0x1x2.nc",
    "EVAPS": "/home/duch/mvems/emission/2023/nsw/evaps_nsw_x0x1x2.nc"
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

# --- Comparison Mode ---
if selected_mode == "Compare 2013 vs 2021":
    file_option = st.sidebar.selectbox("Select Emission Type", ["Cold Start (CS)", "Hot Start (Hot)"])
    ds_2013 = load_dataset(vkt_paths["2013"][file_option])
    ds_2021 = load_dataset(vkt_paths["2021"][file_option])

    common_vars = list(set(ds_2013.data_vars) & set(ds_2021.data_vars))
    if not common_vars:
        st.error("No common variables found.")
        st.stop()

    variable = st.sidebar.selectbox("Select Variable", common_vars)
    time_index = st.sidebar.slider("Select Hour", 0, len(ds_2013.time) - 1, 0)

    # Regional totals
    hourly_2013 = ds_2013[variable].sum(dim=("northing", "easting"), skipna=True).values
    hourly_2021 = ds_2021[variable].sum(dim=("northing", "easting"), skipna=True).values
    hourly_diff = hourly_2021 - hourly_2013
    total_diff_day = np.nansum(hourly_diff)

    st.sidebar.metric("Δ VKT (Selected Hour)", f"{hourly_diff[time_index]:,.2f}")
    st.sidebar.metric("Δ VKT (24h Total)", f"{total_diff_day:,.2f}")

    df_compare = pd.DataFrame({
        "Hour": ds_2013["time"].values,
        "2013": hourly_2013,
        "2021": hourly_2021,
        "Difference": hourly_diff
    })
    st.sidebar.plotly_chart(px.line(df_compare, x="Hour", y=["2013", "2021", "Difference"], markers=True), use_container_width=True)
    st.sidebar.download_button("Download CSV", df_compare.to_csv(index=False), f"vkt_diff_2013_vs_2021.csv", mime="text/csv")

    # Spatial Map
    diff_map = ds_2021[variable].isel(time=time_index) - ds_2013[variable].isel(time=time_index)
    lat_grid, lon_grid = build_latlon_grid(ds_2013["easting"].values, ds_2013["northing"].values)
    heatmap_df = pd.DataFrame({
        "lat": lat_grid.flatten(),
        "lon": lon_grid.flatten(),
        "value": diff_map.values.flatten()
    }).dropna()

    m = folium.Map(location=[-33.8, 151.2], zoom_start=9)
    HeatMap(heatmap_df[["lat", "lon", "value"]].values.tolist(), radius=8).add_to(m)
    st.title("2021 vs 2013 VKT Spatial Difference")
    st.subheader(f"{variable} | Hour {time_index}")
    st_folium(m, height=600)
    st.stop()

# --- Single-Year Mode ---
selected_year = selected_mode
if selected_year in ["2013", "2021"]:
    file_option = st.sidebar.selectbox("Select Emission Type", ["Cold Start (CS)", "Hot Start (Hot)"])
    ds = load_dataset(vkt_paths[selected_year][file_option])
    variable = st.sidebar.selectbox("Select Variable", list(ds.data_vars))
    legend_label = variable
    time_index = st.sidebar.slider("Select Hour", 0, len(ds["time"]) - 1, 0)
    data_slice = ds[variable].isel(time=time_index)
    lat_grid, lon_grid = build_latlon_grid(ds["easting"].values, ds["northing"].values)

elif selected_year == "2023":
    vkt_or_emis = st.sidebar.selectbox("Select 2023 Data Type", ["VKT", "Emission"])
    if vkt_or_emis == "VKT":
        ds = load_dataset(vkt_paths["2023"])
        vehicle_mapping = {
            "Passenger Car (PC)": "VKT_cat1",
            "Light Commercial Vehicle (LCV)": "VKT_cat3",
            "Rigid Trucks (RIG)": "VKT_cat11",
            "Articulated Trucks (ART)": "VKT_cat12",
            "Motorcycle (MC)": "VKT_cat20"
        }
        readable_names = [k for k in vehicle_mapping if vehicle_mapping[k] in ds.data_vars]
        selected_readable = st.sidebar.selectbox("Select Vehicle Category", readable_names)
        variable = vehicle_mapping[selected_readable]
        legend_label = selected_readable
        time_index = st.sidebar.slider("Select Hour", 0, len(ds.time) - 1, 0)
        data_slice = ds[variable].isel(time=time_index)
        lat_vals = ds["lat"].values
        lon_vals = ds["lon"].values
        lat_grid, lon_grid = np.meshgrid(lat_vals, lon_vals, indexing="ij")
    else:
        emis_type = st.sidebar.selectbox("Emission Type", list(emissions_2023))
        month_sel = st.sidebar.selectbox("Month", list(range(1, 13)))
        day_sel = st.sidebar.selectbox("Day Type", ["Weekday", "Weekend"])
        wdwe = 1 if day_sel == "Weekday" else 2
        ds = load_dataset(emissions_2023[emis_type])
        variable = st.sidebar.selectbox("Select Species", list(ds.data_vars))
        legend_label = f"{variable} ({emis_type})"
        time_index = st.sidebar.slider("Select Hour", 0, len(ds["Time"]) - 1, 0)
        data_slice = ds[variable].sel(Month=month_sel, WD_WE_ID=wdwe).isel(Time=time_index)
        lat_vals = ds["lat"].values
        lon_vals = ds["lon"].values
        lat_grid, lon_grid = np.meshgrid(lat_vals, lon_vals, indexing="ij")

# --- Summary Stats ---
total_val = np.nansum(data_slice.values)
mean_val = np.nanmean(data_slice.values)

st.sidebar.markdown("---")
st.sidebar.metric("Total (24h)", f"{total_val:,.2f}")
st.sidebar.metric("Mean (Grid Cell)", f"{mean_val:,.2f}")

# --- Map ---
heatmap_df = pd.DataFrame({
    "lat": lat_grid.flatten(),
    "lon": lon_grid.flatten(),
    "value": data_slice.values.flatten()
}).dropna()

m = folium.Map(location=[-33.8, 151.2], zoom_start=7 if selected_year == "2023" else 9)
HeatMap(heatmap_df[["lat", "lon", "value"]].values.tolist(), radius=8).add_to(m)

st.title("Motor Vehicle VKT & Emission Dashboard")
st.subheader(f"{legend_label} | Hour: {time_index}")
col1, col2 = st.columns([2, 1])

with col1:
    click_data = st_folium(m, height=600)

with col2:
    if click_data and click_data.get("last_clicked"):
        lat_click = click_data["last_clicked"]["lat"]
        lon_click = click_data["last_clicked"]["lng"]
        dist = np.sqrt((lat_grid - lat_click) ** 2 + (lon_grid - lon_click) ** 2)
        n_idx, e_idx = np.unravel_index(np.argmin(dist), dist.shape)
        st.markdown(f"**Lat:** `{lat_click:.4f}` | **Lon:** `{lon_click:.4f}`")

        if selected_year == "2023" and vkt_or_emis == "Emission":
            ts = ds[variable].sel(Month=month_sel, WD_WE_ID=wdwe)[:, n_idx, e_idx].values
            time_vals = ds["Time"].values
        else:
            ts = ds[variable][:, n_idx, e_idx].values
            time_vals = ds["time"].values

        fig = px.line(x=time_vals, y=ts, markers=True, labels={"x": "Hour", "y": variable})
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Summary**")
        st.markdown(f"- Total (24h): `{np.nansum(ts):.3f}`")
        st.markdown(f"- Mean per Hour: `{np.nanmean(ts):.3f}`")

        ts_df = pd.DataFrame({"Hour": time_vals, variable: ts})
        st.download_button("Download Time Series CSV", ts_df.to_csv(index=False), file_name=f"{variable}_lat{lat_click:.2f}_lon{lon_click:.2f}_{selected_year}.csv", mime="text/csv")
    else:
        st.info("Click on the map to explore time series at a location.")

