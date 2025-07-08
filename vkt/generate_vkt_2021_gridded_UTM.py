import os
import re
import pandas as pd
import numpy as np
import xarray as xr

# Configuration
vkt_dir = "/home/duch/mvems/vkt/2021"
resolution = 1000  # meters

# Get all Hour_*.txt files (non-CS)
hour_files = sorted([f for f in os.listdir(vkt_dir) if re.fullmatch(r'Hour_\d{1,2}\.txt', f)])

# Determine UTM bounds from all files
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

# Get variable names from first file
sample_df = pd.read_csv(os.path.join(vkt_dir, hour_files[0]), delim_whitespace=True)
vkt_vars = [col for col in sample_df.columns if re.fullmatch(r'[A-Z]+_\d', col)]

# Allocate array: (time, y, x)
data = {var: np.full((24, len(northings), len(eastings)), np.nan, dtype=np.float32) for var in vkt_vars}

# Fill data
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

# Create xarray Dataset
ds = xr.Dataset(coords={
    "time": np.arange(24),
    "northing": northings,
    "easting": eastings
})

for var, arr in data.items():
    ds[var] = (("time", "northing", "easting"), arr)
    ds[var].attrs["units"] = "vehicle kilometers"
    ds[var].attrs["long_name"] = f"{var} aggregated to 1km UTM grid"

# Metadata
ds.attrs.update({
    "title": "Hourly Vehicle Emissions on 1km UTM grid",
    "coordinate_system": "EPSG:32756 (UTM Zone 56S)",
    "source": "NSW MVEC 2021",
    "Conventions": "CF-1.8"
})

# Output
ds.to_netcdf("vkt_2021_hourly_utm_1km.nc")
print("✅ Output saved to vkt_2021_hourly_utm_1km.nc")

