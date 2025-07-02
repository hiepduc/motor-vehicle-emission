import pandas as pd
import numpy as np
import xarray as xr

# === STEP 1: Load VKT link data ===
file_path = "Hour_0.txt"  # Replace with your full path
df = pd.read_csv(file_path, delim_whitespace=True)

# === STEP 2: Define vehicle types and road types ===
vehicle_types = ['CAR', 'LCV', 'RIG', 'ART', 'BUS', 'MC']
road_types = [1, 2, 3, 4, 5]
vehicle_columns = [f"{veh}_{rt}" for rt in road_types for veh in vehicle_types]
vkt_columns = [f"VKT_{rt}" for rt in road_types]

# === STEP 3: Bin into 1 km UTM grid ===
df["easting_bin"] = (df["Xcoord"] // 1000) * 1000
df["northing_bin"] = (df["Ycoord"] // 1000) * 1000

# === STEP 4: Group by 1km grid and sum all VKT fields ===
group_cols = ["northing_bin", "easting_bin"]
sum_columns = vkt_columns + vehicle_columns
grouped = df.groupby(group_cols)[sum_columns].sum().reset_index()

# === STEP 5: Create grid axes ===
eastings = np.arange(grouped["easting_bin"].min(), grouped["easting_bin"].max() + 1000, 1000)
northings = np.arange(grouped["northing_bin"].min(), grouped["northing_bin"].max() + 1000, 1000)
e_idx = {e: i for i, e in enumerate(eastings)}
n_idx = {n: i for i, n in enumerate(northings)}

# === STEP 6: Fill arrays with gridded values ===
shape = (len(northings), len(eastings))
grid_data = {col: np.full(shape, np.nan) for col in sum_columns}

for _, row in grouped.iterrows():
    i = n_idx[row["northing_bin"]]
    j = e_idx[row["easting_bin"]]
    for col in sum_columns:
        grid_data[col][i, j] = row[col]

# === STEP 7: Create xarray Dataset ===
ds = xr.Dataset()
for col in sum_columns:
    ds[col] = (("northing", "easting"), grid_data[col])
    ds[col].attrs["units"] = "vehicle kilometers"
    ds[col].attrs["long_name"] = f"{col} aggregated to 1km UTM grid"

ds["easting"] = eastings
ds["northing"] = northings
ds["easting"].attrs["units"] = "m"
ds["northing"].attrs["units"] = "m"

# === Metadata for CMAQ compatibility ===
ds.attrs["title"] = "Gridded Vehicle Kilometers Traveled (VKT)"
ds.attrs["summary"] = "Aggregated vehicle kilometers per 1km² UTM grid by vehicle type and road category"
ds.attrs["coordinate_system"] = "EPSG:32756 (UTM Zone 56S)"
ds.attrs["resolution"] = "1km"
ds.attrs["source"] = "NSW MVEC 2013 VKT input"
ds.attrs["Conventions"] = "CF-1.8"

# === Save to NetCDF ===
output_file = "vkt_gridded_1km.nc"
ds.to_netcdf(output_file)
print(f"Saved gridded VKT to {output_file}")

