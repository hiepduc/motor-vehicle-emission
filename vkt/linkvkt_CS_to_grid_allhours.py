import os
import re
import numpy as np
import pandas as pd
import xarray as xr

# --- Configuration ---
vkt_dir = "/mnt/climate/cas/project/mems/mvec/2013"
resolution = 1000  # 1 km
output_file = "vkt_cs_emissions_gridded_hourly.nc"

# --- List of CS files ---
hour_files = sorted([f for f in os.listdir(vkt_dir) if re.fullmatch(r'Hour_(\d{1,2})_CS\.txt', f)])
assert hour_files, "No _CS files found!"

# --- Determine spatial extent ---
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

# --- Read columns from a sample file ---
sample_file = os.path.join(vkt_dir, hour_files[0])
sample_df = pd.read_csv(sample_file, delim_whitespace=True, nrows=1)
emission_columns = [col for col in sample_df.columns if re.match(r'd.*_[CL]_R[1-5]', col)]

# --- Prepare empty arrays ---
data = {
    col: np.full((24, len(northings), len(eastings)), np.nan, dtype=np.float32)
    for col in emission_columns
}

# --- Fill arrays ---
for f in hour_files:
    hour = int(re.search(r'Hour_(\d{1,2})', f).group(1))
    df = pd.read_csv(os.path.join(vkt_dir, f), delim_whitespace=True)
    
    for _, row in df.iterrows():
        i = int((row["Xcoord"] - x_min) // resolution)
        j = int((row["Ycoord"] - y_min) // resolution)
        if 0 <= i < len(eastings) and 0 <= j < len(northings):
            for col in emission_columns:
                if col in row and not pd.isna(row[col]):
                    if np.isnan(data[col][hour, j, i]):
                        data[col][hour, j, i] = 0
                    data[col][hour, j, i] += row[col]

# --- Create xarray Dataset ---
ds = xr.Dataset(coords={
    "time": np.arange(24),
    "northing": northings,
    "easting": eastings,
})

for var, arr in data.items():
    ds[var] = (("time", "northing", "easting"), arr)
    ds[var].attrs["units"] = "unknown"  # Can be updated if known (e.g., g/km)
    ds[var].attrs["long_name"] = f"{var} cold start emission aggregated to 1km UTM grid"

# --- Global metadata ---
ds.attrs.update({
    "title": "Hourly Cold Start Emissions Gridded to 1km",
    "summary": "Gridded cold start emissions by vehicle age group, type, and road type",
    "coordinate_system": "EPSG:32756 (UTM Zone 56S)",
    "resolution": "1km",
    "source": "NSW MVEC 2013 Cold Start Emission Files",
    "Conventions": "CF-1.8"
})

# --- Save ---
ds.to_netcdf(output_file)
print(f"✅ Gridded cold-start emissions saved to: {output_file}")

