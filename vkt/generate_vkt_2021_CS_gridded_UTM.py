# ----------------------------------------------------------------------------------------- 
# This program read the Cold Sstart VKT 2021 hourly data files (24 files) and convert them
# to a single NetCDF file containg VKT for different vehicle types 
# Note that the names of coordinates in the hourly files are XCoord and YCcord in CS files
# rather the names Xcoord Ycoord in the hot VKT hourly files
# ----------------------------------------------------------------------------------------- 
import os
import re
import numpy as np
import pandas as pd
import xarray as xr

# --- Configuration ---
vkt_dir = "/home/duch/mvems/vkt/2021"
output_file = "vkt_2021_CS_hourly_utm_1km.nc"
resolution = 1000  # 1 km
utm_epsg = 32756  # UTM zone 56S

# --- Find input files ---
hour_files = sorted([f for f in os.listdir(vkt_dir) if re.fullmatch(r'Hour_\d{1,2}_CS\.txt', f)])
if not hour_files:
    raise RuntimeError("No CS input files found.")

# --- Get variable names ---
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

eastings = np.arange(x_min, x_max + resolution, resolution)
northings = np.arange(y_min, y_max + resolution, resolution)

# --- Prepare empty data arrays ---
data = {
    var: np.full((24, len(northings), len(eastings)), np.nan, dtype=np.float32)
    for var in vkt_vars
}

# --- Read and bin data ---
for f in hour_files:
    hour = int(re.search(r'Hour_(\d{1,2})', f).group(1))
    df = pd.read_csv(os.path.join(vkt_dir, f), delim_whitespace=True)

    for _, row in df.iterrows():
        i = int((row["XCoord"] - x_min) // resolution)
        j = int((row["YCoord"] - y_min) // resolution)
        if 0 <= i < len(eastings) and 0 <= j < len(northings):
            for var in vkt_vars:
                if not pd.isna(row[var]):
                    if np.isnan(data[var][hour, j, i]):
                        data[var][hour, j, i] = 0
                    data[var][hour, j, i] += row[var]

# --- Create xarray Dataset ---
ds = xr.Dataset(
    coords={
        "time": np.arange(24),
        "northing": northings,
        "easting": eastings
    }
)

# --- Add data variables ---
for var, arr in data.items():
    ds[var] = (("time", "northing", "easting"), arr)
    ds[var].attrs["units"] = "vehicle kilometers"
    ds[var].attrs["long_name"] = f"{var} aggregated to 1km UTM grid"

# --- Global metadata ---
ds.attrs.update({
    "title": "Hourly Gridded Cold-Start VKT in UTM",
    "summary": "Aggregated cold-start vehicle emissions from Hour_*_CS.txt into 1km UTM grid",
    "coordinate_system": f"EPSG:{utm_epsg} (UTM Zone 56S)",
    "resolution": "1km",
    "source": "NSW MVEC 2021 VKT Cold-Start",
    "Conventions": "CF-1.8"
})

# --- Save ---
ds.to_netcdf(output_file)
print(f"✅ UTM-based CS emission grid saved to: {output_file}")

