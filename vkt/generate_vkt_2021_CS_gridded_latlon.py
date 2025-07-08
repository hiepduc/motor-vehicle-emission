import os
import re
import numpy as np
import pandas as pd
import xarray as xr
from pyproj import Transformer

# --- Configuration ---
vkt_dir = "/home/duch/mvems/vkt/2021"
output_file = "vkt_2021_CS_hourly_latlon_1km.nc"
resolution = 1000  # 1km
utm_zone_epsg = 32756  # UTM zone 56S

# --- Load file list ---
hour_files = sorted([f for f in os.listdir(vkt_dir) if re.fullmatch(r'Hour_\d{1,2}_CS\.txt', f)])
if not hour_files:
    raise RuntimeError("No CS files found!")

# --- Discover variable names ---
sample_df = pd.read_csv(os.path.join(vkt_dir, hour_files[0]), delim_whitespace=True)
vkt_vars = [col for col in sample_df.columns if re.match(r'd.*_(C|L)_R\d', col)]

# --- Determine domain bounds ---
x_all, y_all = [], []
for f in hour_files:
    df = pd.read_csv(os.path.join(vkt_dir, f), delim_whitespace=True, usecols=["XCoord", "YCoord"])
    x_all.append(df["XCoord"])
    y_all.append(df["YCoord"])
x = pd.concat(x_all)
y = pd.concat(y_all)
x_min, x_max = x.min(), x.max()
y_min, y_max = y.min(), y.max()

x_bins = np.arange(x_min, x_max + resolution, resolution)
y_bins = np.arange(y_min, y_max + resolution, resolution)

# --- Prepare data arrays ---
data = {
    var: np.full((24, len(y_bins), len(x_bins)), np.nan, dtype=np.float32)
    for var in vkt_vars
}

# --- Fill data into grid ---
for f in hour_files:
    hour = int(re.search(r'Hour_(\d{1,2})', f).group(1))
    df = pd.read_csv(os.path.join(vkt_dir, f), delim_whitespace=True)

    for _, row in df.iterrows():
        i = int((row["XCoord"] - x_min) // resolution)
        j = int((row["YCoord"] - y_min) // resolution)
        if 0 <= i < len(x_bins) and 0 <= j < len(y_bins):
            for var in vkt_vars:
                if not pd.isna(row[var]):
                    if np.isnan(data[var][hour, j, i]):
                        data[var][hour, j, i] = 0
                    data[var][hour, j, i] += row[var]

# --- Create lat/lon grid ---
xv, yv = np.meshgrid(x_bins, y_bins)
transformer = Transformer.from_crs(f"EPSG:{utm_zone_epsg}", "EPSG:4326", always_xy=True)
lon2d, lat2d = transformer.transform(xv, yv)

# --- Create xarray Dataset ---
ds = xr.Dataset(
    coords={
        "time": np.arange(24),
        "y": np.arange(len(y_bins)),
        "x": np.arange(len(x_bins)),
    }
)

# Add 2D lat/lon coordinates
ds["lat"] = (("y", "x"), lat2d)
ds["lon"] = (("y", "x"), lon2d)
ds["lat"].attrs["units"] = "degrees_north"
ds["lon"].attrs["units"] = "degrees_east"

# Add variables
for var, arr in data.items():
    ds[var] = (("time", "y", "x"), arr)
    ds[var].attrs["units"] = "vehicle kilometers"
    ds[var].attrs["long_name"] = f"{var} aggregated to 1km grid"

# --- Global attributes ---
ds.attrs.update({
    "title": "Hourly Gridded Cold-Start Vehicle Emissions",
    "summary": "Aggregated emissions from Hour_*_CS.txt to 1km UTM grid, converted to lat/lon",
    "coordinate_system": "EPSG:4326",
    "original_crs": f"EPSG:{utm_zone_epsg}",
    "resolution": "1km",
    "source": "NSW MVEC 2021 VKT cold-start emissions",
    "Conventions": "CF-1.8"
})

# --- Save ---
ds.to_netcdf(output_file)
print(f"✅ CS emission grid saved to: {output_file}")

