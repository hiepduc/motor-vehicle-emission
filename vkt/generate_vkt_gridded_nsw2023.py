import os
import re
import numpy as np
import pandas as pd
import xarray as xr

# Configuration
vkt_dir = "/home/duch/mvems/vkt/2023"
resolution = 0.01  # ~1 km in degrees

# Get hourly CSV files
hour_files = sorted([f for f in os.listdir(vkt_dir) if re.fullmatch(r'NSW_VKT_Hour\d{1,2}\.csv', f)])

# Get all lat/lon for grid bounds
lat_all, lon_all = [], []
cat_ids = set()
for f in hour_files:
    df = pd.read_csv(os.path.join(vkt_dir, f))
    lat_all.append(df["repr_point_lat"])
    lon_all.append(df["repr_point_lon"])
    cat_ids.update(df["Veh_Cat_ID"].unique())

lat = pd.concat(lat_all)
lon = pd.concat(lon_all)
lat_min, lat_max = lat.min(), lat.max()
lon_min, lon_max = lon.min(), lon.max()

# Grid coordinates
lat_bins = np.arange(lat_min, lat_max + resolution, resolution)
lon_bins = np.arange(lon_min, lon_max + resolution, resolution)
lat_centers = (lat_bins[:-1] + lat_bins[1:]) / 2
lon_centers = (lon_bins[:-1] + lon_bins[1:]) / 2

# Prepare output arrays: one per vehicle category
cat_ids = sorted(list(cat_ids))
data = {
    f"VKT_cat{cat_id}": np.full((24, len(lat_centers), len(lon_centers)), np.nan, dtype=np.float32)
    for cat_id in cat_ids
}

# Fill arrays
for f in hour_files:
    hour = int(re.search(r'Hour(\d{1,2})', f).group(1))
    df = pd.read_csv(os.path.join(vkt_dir, f))

    for _, row in df.iterrows():
        lat_val = row["repr_point_lat"]
        lon_val = row["repr_point_lon"]
        cat_id = int(row["Veh_Cat_ID"])
        vkt_val = row["LINK_HOURLY_VKT"]

        lat_idx = np.digitize(lat_val, lat_bins) - 1
        lon_idx = np.digitize(lon_val, lon_bins) - 1

        if 0 <= lat_idx < len(lat_centers) and 0 <= lon_idx < len(lon_centers):
            key = f"VKT_cat{cat_id}"
            if np.isnan(data[key][hour, lat_idx, lon_idx]):
                data[key][hour, lat_idx, lon_idx] = 0
            data[key][hour, lat_idx, lon_idx] += vkt_val

# Build xarray Dataset
ds = xr.Dataset(
    coords={
        "time": np.arange(24),
        "lat": lat_centers,
        "lon": lon_centers
    }
)

for key, arr in data.items():
    ds[key] = (("time", "lat", "lon"), arr)
    ds[key].attrs["units"] = "vehicle kilometers"
    ds[key].attrs["long_name"] = f"{key} aggregated to ~1km lat/lon grid"

# Metadata
ds.attrs.update({
    "title": "NSW 2023 Hourly VKT per Vehicle Category",
    "source": "NSW MVEC 2023",
    "grid_resolution": f"{resolution} degrees",
    "categories": ", ".join([f"cat{c}" for c in cat_ids]),
    "Conventions": "CF-1.8"
})

# Save
out_path = "vkt_nsw2023_hourly_gridded_latlon.nc"
ds.to_netcdf(out_path)
print(f"✅ Saved to {out_path}")

