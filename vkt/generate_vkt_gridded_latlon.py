import os
import re
import numpy as np
import pandas as pd
import xarray as xr
from pyproj import Transformer

# --- Configuration ---
vkt_dir = "/mnt/climate/cas/project/mems/mvec/2013"  # Your input directory
output_dir = "./gridded_output_latlon"
resolution = 1000  # in meters (1 km)

os.makedirs(output_dir, exist_ok=True)

# --- UTM to Lat-Lon Transformer (EPSG:32756 → EPSG:4326) ---
transformer = Transformer.from_crs("EPSG:32756", "EPSG:4326", always_xy=True)

# --- File selection ---
hour_files = sorted([f for f in os.listdir(vkt_dir) if re.fullmatch(r'Hour_\d{1,2}\.txt', f)])

# --- Read all coordinates to define grid extent ---
x_all, y_all = [], []
for f in hour_files:
    df = pd.read_csv(os.path.join(vkt_dir, f), delim_whitespace=True, usecols=["Xcoord", "Ycoord"])
    x_all.append(df["Xcoord"])
    y_all.append(df["Ycoord"])
x_all = pd.concat(x_all)
y_all = pd.concat(y_all)

# Snap to nearest grid edges
x_min = (x_all.min() // resolution) * resolution
x_max = (x_all.max() // resolution + 1) * resolution
y_min = (y_all.min() // resolution) * resolution
y_max = (y_all.max() // resolution + 1) * resolution

eastings = np.arange(x_min, x_max, resolution)
northings = np.arange(y_min, y_max, resolution)

# Convert grid centers to lat/lon
xe, yn = np.meshgrid(eastings, northings)
lon_grid, lat_grid = transformer.transform(xe, yn)

# --- Define variables ---
vehicle_types = ["CAR", "LCV", "RIG", "ART", "BUS", "MC"]
road_types = [1, 2, 3, 4, 5]
vkt_columns = [f"{veh}_{rt}" for rt in road_types for veh in vehicle_types]
vkt_totals = [f"VKT_{rt}" for rt in road_types]
all_vars = vkt_columns + vkt_totals

# --- Allocate 3D arrays ---
data = {var: np.full((24, len(northings), len(eastings)), np.nan, dtype=np.float32) for var in all_vars}

# --- Read and accumulate hourly data ---
for f in hour_files:
    hour = int(re.search(r'Hour_(\d{1,2})', f).group(1))
    df = pd.read_csv(os.path.join(vkt_dir, f), delim_whitespace=True)

    for _, row in df.iterrows():
        x_idx = int((row["Xcoord"] - x_min) // resolution)
        y_idx = int((row["Ycoord"] - y_min) // resolution)

        if 0 <= x_idx < len(eastings) and 0 <= y_idx < len(northings):
            for var in all_vars:
                if var in row and not pd.isna(row[var]):
                    if np.isnan(data[var][hour, y_idx, x_idx]):
                        data[var][hour, y_idx, x_idx] = 0
                    data[var][hour, y_idx, x_idx] += row[var]

# --- Create xarray Dataset ---
ds = xr.Dataset(coords={
    "time": np.arange(24),
    "lat": (("northing", "easting"), lat_grid),
    "lon": (("northing", "easting"), lon_grid),
})

for var in all_vars:
    ds[var] = (("time", "northing", "easting"), data[var])
    ds[var].attrs["units"] = "vehicle kilometers"
    ds[var].attrs["long_name"] = f"{var} aggregated to 1km grid in lat/lon"

# Add metadata
ds.attrs.update({
    "title": "Hourly Gridded Vehicle Kilometers Traveled (VKT)",
    "summary": "Hourly VKT by vehicle/road type gridded to 1km² in latitude/longitude",
    "source_crs": "EPSG:32756",
    "target_crs": "EPSG:4326",
    "resolution": "1km",
    "source": "NSW MVEC 2013",
    "Conventions": "CF-1.8"
})

# --- Save ---
output_path = os.path.join(output_dir, "vkt_allhourly_latlon_1km.nc")
ds.to_netcdf(output_path)
print(f"✅ Gridded VKT (lat/lon) saved to: {output_path}")

