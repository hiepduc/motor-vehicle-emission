import os
import re
import pandas as pd
import numpy as np
import xarray as xr
import pyproj

# --- Configuration ---
vkt_dir = "/home/duch/mvems/vkt/2013"
resolution = 1000  # meters
epsg_code = 32756  # UTM Zone 56S (for Sydney region)
proj_utm = pyproj.CRS.from_epsg(epsg_code)
proj_latlon = pyproj.CRS.from_epsg(4326)
transformer = pyproj.Transformer.from_crs(proj_utm, proj_latlon, always_xy=True)

# --- Get all Hour_*.txt files ---
hour_files = sorted([f for f in os.listdir(vkt_dir) if re.fullmatch(r'Hour_\d{1,2}\.txt', f)])
if not hour_files:
    raise FileNotFoundError("No Hour_*.txt files found in the input directory.")

# --- Read variable names ---
sample_df = pd.read_csv(os.path.join(vkt_dir, hour_files[0]), delim_whitespace=True)
vkt_vars = [col for col in sample_df.columns if re.fullmatch(r'[A-Z]+_\d', col)]

# --- Get bounds from all files ---
x_all, y_all = [], []
for f in hour_files:
    df = pd.read_csv(os.path.join(vkt_dir, f), delim_whitespace=True, usecols=["Xcoord", "Ycoord"])
    x_all.append(df["Xcoord"])
    y_all.append(df["Ycoord"])
x = pd.concat(x_all)
y = pd.concat(y_all)
x_min, x_max = x.min(), x.max()
y_min, y_max = y.min(), y.max()

eastings = np.arange(x_min, x_max + resolution, resolution)
northings = np.arange(y_min, y_max + resolution, resolution)

# --- Convert to lat/lon grid ---
easting_grid, northing_grid = np.meshgrid(eastings, northings)
lon, lat = transformer.transform(easting_grid, northing_grid)

# --- Allocate output arrays ---
data = {var: np.full((24, lat.shape[0], lat.shape[1]), np.nan, dtype=np.float32) for var in vkt_vars}

# --- Fill values into grid ---
for f in hour_files:
    hour = int(re.search(r'Hour_(\d{1,2})', f).group(1))
    df = pd.read_csv(os.path.join(vkt_dir, f), delim_whitespace=True)
    for _, row in df.iterrows():
        i = int((row["Xcoord"] - x_min) // resolution)
        j = int((row["Ycoord"] - y_min) // resolution)
        if 0 <= i < len(eastings) and 0 <= j < len(northings):
            for var in vkt_vars:
                if var in row and not pd.isna(row[var]):
                    if np.isnan(data[var][hour, j, i]):
                        data[var][hour, j, i] = 0
                    data[var][hour, j, i] += row[var]

# --- Build xarray Dataset with (time, lat, lon) ---
ny, nx = lat.shape

ds = xr.Dataset(coords={
    "time": ("time", np.arange(24)),
    "y": ("y", np.arange(ny)),
    "x": ("x", np.arange(nx)),
    "lat": (("y", "x"), lat),
    "lon": (("y", "x"), lon),
})

for var, arr in data.items():
    ds[var] = (("time", "y", "x"), arr)
    ds[var].attrs["units"] = "vehicle kilometers"
    ds[var].attrs["long_name"] = f"{var} aggregated to 1km lat/lon grid"

# --- Metadata ---
ds.attrs.update({
    "title": "Hourly Gridded Vehicle Kilometers Traveled (VKT)",
    "summary": "Hourly VKT aggregated to 1km grid in lat/lon (WGS84)",
    "coordinate_system": "WGS84",
    "original_projection": f"EPSG:{epsg_code}",
    "resolution": "1km",
    "source": "NSW MVEC 2013 Hourly Vehicle Emission Files",
    "Conventions": "CF-1.8"
})

# --- Save NetCDF ---
ds.to_netcdf("vkt_hourly_latlon_1km.nc")
print("✅ NetCDF file created: vkt_hourly_latlon_1km.nc")

