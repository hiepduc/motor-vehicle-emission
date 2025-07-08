import os
import re
import pandas as pd
import numpy as np
import xarray as xr

# --- Configuration ---
vkt_dir = "/mnt/climate/cas/project/mems/mvec/2013"
resolution = 1000  # 1 km
vehicle_types = ["CAR", "LCV", "RIG", "ART", "BUS", "MC"]
road_types = [1, 2, 3, 4, 5]
vkt_columns = [f"{veh}_{rt}" for rt in road_types for veh in vehicle_types]
vkt_totals = [f"VKT_{rt}" for rt in road_types]

# --- Load all coordinates to define consistent grid ---
hour_files = sorted([f for f in os.listdir(vkt_dir) if re.fullmatch(r'Hour_\d{1,2}\.txt', f)])

x_all, y_all = [], []
for f in hour_files:
    df = pd.read_csv(os.path.join(vkt_dir, f), delim_whitespace=True, usecols=["Xcoord", "Ycoord"])
    x_all.append(df["Xcoord"])
    y_all.append(df["Ycoord"])

x_all = pd.concat(x_all)
y_all = pd.concat(y_all)

# Snap grid bounds to nearest lower multiple of resolution
x_min = (x_all.min() // resolution) * resolution
x_max = (x_all.max() // resolution + 1) * resolution
y_min = (y_all.min() // resolution) * resolution
y_max = (y_all.max() // resolution + 1) * resolution

eastings = np.arange(x_min, x_max, resolution)
northings = np.arange(y_min, y_max, resolution)

# --- Create empty data arrays ---
data = {
    col: np.full((24, len(northings), len(eastings)), np.nan, dtype=np.float32)
    for col in vkt_columns + vkt_totals
}

# --- Fill the arrays with data ---
for f in hour_files:
    hour = int(re.search(r'Hour_(\d{1,2})', f).group(1))
    df = pd.read_csv(os.path.join(vkt_dir, f), delim_whitespace=True)

    for _, row in df.iterrows():
        x_idx = int((row["Xcoord"] - x_min) // resolution)
        y_idx = int((row["Ycoord"] - y_min) // resolution)

        if 0 <= x_idx < len(eastings) and 0 <= y_idx < len(northings):
            for col in vkt_columns + vkt_totals:
                if col in row and not pd.isna(row[col]):
                    if np.isnan(data[col][hour, y_idx, x_idx]):
                        data[col][hour, y_idx, x_idx] = 0
                    data[col][hour, y_idx, x_idx] += row[col]

# --- Create xarray Dataset ---
ds = xr.Dataset(coords={
    "time": np.arange(24),
    "northing": northings,
    "easting": eastings,
})

for var, arr in data.items():
    ds[var] = (("time", "northing", "easting"), arr)
    ds[var].attrs["units"] = "vehicle kilometers"
    ds[var].attrs["long_name"] = f"{var} aggregated to 1km UTM grid"

# --- Global metadata ---
ds.attrs.update({
    "title": "Hourly Gridded Vehicle Kilometers Traveled (VKT)",
    "summary": "Hourly aggregated vehicle kilometers per 1km² UTM grid by vehicle type and road category",
    "coordinate_system": "EPSG:32756 (UTM Zone 56S)",
    "resolution": "1km",
    "source": "NSW MVEC 2013 VKT input",
    "Conventions": "CF-1.8"
})

# --- Output file ---
output_path = os.path.join(vkt_dir, "vkt_hot_allhourly_gridded_1km.nc")
ds.to_netcdf(output_path)
print(f"✅ Output saved to: {output_path}")

