import streamlit as st
import xarray as xr
import pandas as pd
import numpy as np
import plotly.express as px
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
from pyproj import Transformer
import matplotlib.pyplot as plt
import branca.colormap as cm  # Add this import at the top

st.set_page_config(layout="wide", page_title="Motor Vehicle VKT & Emissions Dashboard")
st.sidebar.title("Motor Vehicle Dashboard (VKT + Emissions)")

# ----------------- CONFIG -----------------
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

emissions_2013 = "/home/duch/mvems/emission/2013/emission_2013_species_total_latlon.nc"
emissions_2021 = "/home/duch/mvems/emission/2021/emission_2021_species_total_latlon.nc"

# ----------------- HELPERS -----------------
@st.cache_resource
def load_dataset(path):
    return xr.open_dataset(path)

transformer = Transformer.from_crs("EPSG:32756", "EPSG:4326", always_xy=True)

@st.cache_data
def build_latlon_grid(eastings, northings):
    xx, yy = np.meshgrid(eastings, northings)
    lon, lat = transformer.transform(xx.flatten(), yy.flatten())
    return lat.reshape(xx.shape), lon.reshape(xx.shape)

# ----------------- UI -----------------
year_options = ["2013", "2021", "2023", "Compare 2013 vs 2021"]
selected_mode = st.sidebar.selectbox("Select Mode", year_options)

if selected_mode == "Compare 2013 vs 2021":
    data_type = st.sidebar.selectbox("Select Data Type", ["VKT", "Emission"] )
    if data_type == "VKT":
        file_option = st.sidebar.selectbox("Select Emission Type", ["Cold Start (CS)", "Hot Start (Hot)"])
        ds_2013 = load_dataset(vkt_paths["2013"][file_option])
        ds_2021 = load_dataset(vkt_paths["2021"][file_option])
        common_vars = list(set(ds_2013.data_vars) & set(ds_2021.data_vars))
        variable = st.sidebar.selectbox("Select Variable", common_vars)
        time_index = st.sidebar.slider("Select Hour", 0, len(ds_2013.time) - 1, 0)

        # Summary stats
        v2013 = ds_2013[variable].sum(dim=("northing", "easting"), skipna=True).values
        v2021 = ds_2021[variable].sum(dim=("northing", "easting"), skipna=True).values
        diff = v2021 - v2013
        st.sidebar.metric("Δ VKT (Selected Hour)", f"{diff[time_index]:,.2f}")
        st.sidebar.metric("Δ VKT (24h Total)", f"{np.nansum(diff):,.2f}")

        df = pd.DataFrame({"Hour": ds_2013["time"].values, "2013": v2013, "2021": v2021, "Difference": diff})
        st.sidebar.plotly_chart(px.line(df, x="Hour", y=["2013", "2021", "Difference"], markers=True), use_container_width=True)
        st.sidebar.download_button("Download CSV", df.to_csv(index=False), "vkt_diff_2013_vs_2021.csv", mime="text/csv")

        # Spatial plot
        diff_map = ds_2021[variable].isel(time=time_index) - ds_2013[variable].isel(time=time_index)
        lat_grid, lon_grid = build_latlon_grid(ds_2013["easting"].values, ds_2013["northing"].values)
        heatmap_df = pd.DataFrame({
            "lat": lat_grid.flatten(),
            "lon": lon_grid.flatten(),
            "value": diff_map.values.flatten()
        }).dropna()
        data_slice = diff_map
        # Normalize value range for the color scale
        vmin = heatmap_df["value"].min()
        vmax = heatmap_df["value"].max()

        # Create color map
        colormap = cm.LinearColormap(
            colors=["blue", "lime", "yellow", "orange", "red"],
            vmin=vmin,
            vmax=vmax,
            caption=f"{legend_label if 'legend_label' in locals() else variable} intensity"
        )
        m = folium.Map(location=[-33.8, 151.2], zoom_start=9)
        # Add HeatMap
        HeatMap(
            heatmap_df[["lat", "lon", "value"]].values.tolist(),
            radius=6,
            max_zoom=13,
            blur=8,
            #gradient={i / (len(colormap.colors) - 1): color for i, color in enumerate(colormap.colors)}
            #gradient=colormap.colors  # apply the gradient colors to match colormap
        ).add_to(m)

        # Add colorbar to map
        colormap.add_to(m)

        #HeatMap(heatmap_df[["lat", "lon", "value"]].values.tolist(), radius=8).add_to(m)
        st.title("2021 vs 2013 VKT Spatial Difference")
        st.subheader(f"{variable} | Hour {time_index}")
        st_folium(m, height=600)  
        #st.stop()

    elif data_type == "Emission":
        ds_2013 = load_dataset(emissions_2013)
        ds_2021 = load_dataset(emissions_2021)
        common_vars = list(set(ds_2013.data_vars) & set(ds_2021.data_vars))
        variable = st.sidebar.selectbox("Select Variable", sorted(common_vars))
    
        # Select Hour and Day Type
        time_index = st.sidebar.slider("Select Hour", 0, 23, 0)
        day_type = st.sidebar.selectbox("Day Type", ["Weekday", "Weekend"])
        wdwe = 0 if day_type == "Weekday" else 1

        # Summary stats
        e2013 = ds_2013[variable][:, wdwe, :, :].sum(dim=("y", "x"), skipna=True).values
        e2021 = ds_2021[variable][:, wdwe, :, :].sum(dim=("y", "x"), skipna=True).values
        diff = e2021 - e2013
        st.sidebar.metric("Δ Emission (Selected Hour)", f"{diff[time_index]:,.2f}")
        st.sidebar.metric("Δ Emission (24h Total)", f"{np.nansum(diff):,.2f}")

        df = pd.DataFrame({
            "Hour": np.arange(24),
            "2013": e2013,
            "2021": e2021,
            "Difference": diff
        })
        st.sidebar.plotly_chart(px.line(df, x="Hour", y=["2013", "2021", "Difference"], markers=True), use_container_width=True)
        st.sidebar.download_button("Download CSV", df.to_csv(index=False), "emission_diff_2013_vs_2021.csv", mime="text/csv")
        #data_slice = ds[variable][time_index, wdwe, :, :]
        dsdiff=ds_2021-ds_2013
        data_slice = dsdiff[variable][time_index, wdwe, :, :]

        # Spatial plot for selected hour
        map2013 = ds_2013[variable][time_index, wdwe, :, :]
        map2021 = ds_2021[variable][time_index, wdwe, :, :]
        diff_map = map2021 - map2013

        lat_grid = ds_2013["lat"].values
        lon_grid = ds_2013["lon"].values

        # Flatten and build heatmap
        heatmap_df = pd.DataFrame({
            "lat": lat_grid.flatten(),
            "lon": lon_grid.flatten(),
            "value": diff_map.values.flatten()
        }).dropna()

        # Convert to absolute value to show intensity of change
        #heatmap_df["value"] = heatmap_df["value"].abs()
        heatmap_df["value"] = heatmap_df["value"]
        # Normalize value range for the color scale
        vmin = heatmap_df["value"].min()
        vmax = heatmap_df["value"].max()

        # Create color map
        colormap = cm.LinearColormap(
            colors=["blue", "lime", "yellow", "orange", "red"],
            vmin=vmin,
            vmax=vmax,
            caption=f"{legend_label if 'legend_label' in locals() else variable} intensity"
        )
        m = folium.Map(location=[-33.8, 151.2], zoom_start=9)
        # Add HeatMap
        HeatMap(
            heatmap_df[["lat", "lon", "value"]].values.tolist(),
            radius=6,
            max_zoom=13,
            blur=8,
            #gradient={i / (len(colormap.colors) - 1): color for i, color in enumerate(colormap.colors)}
            #gradient=colormap.colors  # apply the gradient colors to match colormap
        ).add_to(m)

        # Add colorbar to map
        colormap.add_to(m)


        #m = folium.Map(location=[-33.8, 151.2], zoom_start=9)
        #HeatMap(heatmap_df[["lat", "lon", "value"]].values.tolist(), radius=8).add_to(m)

        st.title("2021 vs 2013 Emission Spatial Difference")
        st.subheader(f"{variable} | Hour {time_index} | {day_type}")
        st_folium(m, height=600)
        #st.stop()
        # Streamlit layout
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

            if data_type == "Emission":
                ts_2013 = ds_2013[variable][:, wdwe, n_idx, e_idx].values
                ts_2021 = ds_2021[variable][:, wdwe, n_idx, e_idx].values
                time_vals = ds_2013["time"].values
            elif data_type == "VKT":
                ts_2013 = ds_2013[variable][:, n_idx, e_idx].values
                ts_2021 = ds_2021[variable][:, n_idx, e_idx].values
                time_vals = ds_2013["time"].values
            else:
                ts_2013 = ts_2021 = np.zeros(24)  # fallback

            ts_diff = ts_2021 - ts_2013
            fig = px.line(x=time_vals, y=ts_diff, markers=True, labels={"x": "Hour", "y": variable})
            st.plotly_chart(fig, use_container_width=True)
            st.markdown(f"**Total (24h):** `{np.nansum(ts_diff):.3f}`")
            st.markdown(f"**Mean per Hour:** `{np.nanmean(ts_diff):.3f}`")
            ts_df = pd.DataFrame({"Hour": time_vals, variable: ts_diff})
            st.download_button("Download Time Series CSV", ts_df.to_csv(index=False),
                               file_name=f"{variable}_lat{lat_click:.2f}_lon{lon_click:.2f}_diff2021_2013.csv",
                               mime="text/csv")

            # Plot
            fig = px.line(
                x=time_vals,
                y=[ts_2013, ts_2021, ts_diff],
                labels={"x": "Hour", "value": variable},
                markers=True
            )
            fig.update_traces(mode='lines+markers')
            fig.update_layout(
                legend=dict(
                    title="Legend",
                    itemsizing="constant"
                ),
                xaxis_title="Hour",
                yaxis_title=variable
            )
            fig.data[0].name = "2013"
            fig.data[1].name = "2021"
            fig.data[2].name = "Difference"

            st.plotly_chart(fig, use_container_width=True)
            st.markdown(f"**Δ Total (24h):** `{np.nansum(ts_diff):.3f}`")
            st.markdown(f"**Δ Mean per Hour:** `{np.nanmean(ts_diff):.3f}`")

            # CSV
            ts_df = pd.DataFrame({
                "Hour": time_vals,
                "2013": ts_2013,
                "2021": ts_2021,
                "Difference": ts_diff
            })
            st.download_button("Download Time Series CSV", ts_df.to_csv(index=False),
                               file_name=f"{variable}_lat{lat_click:.2f}_lon{lon_click:.2f}_diff_2013_vs_2021.csv",
                               mime="text/csv")
        else:
            st.info("Click on the map to explore time series at a location.")

# ----------------- SINGLE YEAR -----------------
selected_year = selected_mode
if selected_year in ["2013", "2021"]:
    #data_type = st.sidebar.selectbox("Select Data Type", ["VKT", "Emission"] if selected_year == "2013" else ["VKT"])
    data_type = st.sidebar.selectbox("Select Data Type", ["VKT", "Emission"] )
    if data_type == "VKT":
        file_option = st.sidebar.selectbox("Emission Type", ["Cold Start (CS)", "Hot Start (Hot)"])
        ds = load_dataset(vkt_paths[selected_year][file_option])
        variable = st.sidebar.selectbox("Select Variable", list(ds.data_vars))
        time_index = st.sidebar.slider("Hour", 0, len(ds["time"]) - 1, 0)
        data_slice = ds[variable].isel(time=time_index)
        lat_grid, lon_grid = build_latlon_grid(ds["easting"].values, ds["northing"].values)
    elif data_type == "Emission":
        ds = load_dataset(emissions_2013)
        variable = st.sidebar.selectbox("Select Species", list(ds.data_vars))
        day_sel = st.sidebar.selectbox("Day Type", ["Weekday", "Weekend"])
        wdwe = 0 if day_sel == "Weekday" else 1
        time_index = st.sidebar.slider("Hour", 0, len(ds["time"]) - 1, 0)

        # Extract 2D lat/lon from file
        lat_grid = ds["lat"].values
        lon_grid = ds["lon"].values
        data_slice = ds[variable][time_index, wdwe, :, :]


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

# ----------------- Summary -----------------
total_val = np.nansum(data_slice.values)
mean_val = np.nanmean(data_slice.values)
st.sidebar.metric("Total (24h)", f"{total_val:,.2f}")
st.sidebar.metric("Mean (Grid Cell)", f"{mean_val:,.2f}")

# ----------------- Map -----------------
heatmap_df = pd.DataFrame({
    "lat": lat_grid.flatten(),
    "lon": lon_grid.flatten(),
    "value": data_slice.values.flatten()
})

# Remove zero and NaN values
heatmap_df = heatmap_df[(heatmap_df["value"] > 0) & (~np.isnan(heatmap_df["value"]))]

# Optional: Boost low values slightly for visibility
heatmap_df["value"] += 1e-6

# Normalize value range for the color scale
vmin = heatmap_df["value"].min()
vmax = heatmap_df["value"].max()

# Create color map
colormap = cm.LinearColormap(
    colors=["blue", "lime", "yellow", "orange", "red"],
    vmin=vmin,
    vmax=vmax,
    caption=f"{legend_label if 'legend_label' in locals() else variable} intensity"
)
m = folium.Map(location=[-33.8, 151.2], zoom_start=7 if selected_year == "2023" else 9)
# Add HeatMap
HeatMap(
    heatmap_df[["lat", "lon", "value"]].values.tolist(),
    radius=6,
    max_zoom=13,
    blur=8,
    #gradient={i / (len(colormap.colors) - 1): color for i, color in enumerate(colormap.colors)}
    #gradient=colormap.colors  # apply the gradient colors to match colormap
).add_to(m)

# Add colorbar to map
colormap.add_to(m)

# Create Folium map
#m = folium.Map(location=[-33.8, 151.2], zoom_start=7 if selected_year == "2023" else 9)

# Add heatmap
#HeatMap(
#    heatmap_df[["lat", "lon", "value"]].values.tolist(),
#    radius=6,
#    max_zoom=13,
#    blur=8
#).add_to(m)

if selected_year in ["2013", "2021", "2023"]:
    # Streamlit layout
    st.title("Motor Vehicle VKT & Emission Dashboard")
    st.subheader(f"{variable} | Hour: {time_index} | {day_sel if 'day_sel' in locals() else ''}")
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
            elif (selected_year == "2013" or selected_year == "2021") and data_type == "Emission":
                ts = ds[variable][:, wdwe, n_idx, e_idx].values
                time_vals = ds["time"].values
            else:
                ts = ds[variable][:, n_idx, e_idx].values
                time_vals = ds["time"].values

            fig = px.line(x=time_vals, y=ts, markers=True, labels={"x": "Hour", "y": variable})
            st.plotly_chart(fig, use_container_width=True)
            st.markdown(f"**Total (24h):** `{np.nansum(ts):.3f}`")
            st.markdown(f"**Mean per Hour:** `{np.nanmean(ts):.3f}`")
            ts_df = pd.DataFrame({"Hour": time_vals, variable: ts})
            st.download_button("Download Time Series CSV", ts_df.to_csv(index=False),
                               file_name=f"{variable}_lat{lat_click:.2f}_lon{lon_click:.2f}_{selected_year}.csv",
                               mime="text/csv")
        else:
            st.info("Click on the map to explore time series at a location.")

