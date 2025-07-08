import xarray as xr
import numpy as np
import os
import argparse

# --- Argument parsing ---
parser = argparse.ArgumentParser(description="Export each month to individual NetCDF files for ncview.")
parser.add_argument("input_file", help="Path to input NetCDF file")
parser.add_argument("--WD_WE_ID", type=int, default=1, help="WD_WE_ID (1=Weekday, 2=Weekend)")
parser.add_argument("--output_dir", default="monthly_exports", help="Directory to save monthly files")
args = parser.parse_args()

# --- Load dataset ---
ds = xr.open_dataset(args.input_file)

# --- Create output directory if not exists ---
os.makedirs(args.output_dir, exist_ok=True)

# --- Get all month values ---
months = ds["Month"].values if "Month" in ds.coords else np.arange(1, 13)

# --- Export loop ---
for month in months:
    print(f"Processing month: {month}")
    
    try:
        subset = ds.sel(WD_WE_ID=args.WD_WE_ID, Month=month)
    except Exception as e:
        print(f"Skipping month {month}: {e}")
        continue

    # Keep only variables with (Time, lat, lon) or (lat, lon)
    selected_vars = {}
    for var in subset.data_vars:
        if subset[var].ndim in [2, 3] and all(d in subset[var].dims for d in ["lat", "lon"]):
            selected_vars[var] = subset[var]

    if not selected_vars:
        print(f"No compatible 2D/3D variables found for month {month}")
        continue

    ds_month = xr.Dataset(selected_vars)

    # Sort lat and transpose
    if "lat" in ds_month.dims:
        ds_month = ds_month.sortby("lat")
    ds_month = ds_month.transpose("Time", "lat", "lon", ...)

    # Clean coords
    for coord in list(ds_month.coords):
        if coord not in ["Time", "lat", "lon"] and ds_month[coord].ndim != 2:
            ds_month = ds_month.drop_vars(coord)


    # Remove unused coordinates
    for coord in list(ds_month.coords):
        if coord not in ["Time", "lat", "lon"] and ds_month[coord].ndim != 2:
            ds_month = ds_month.drop_vars(coord)

    out_path = os.path.join(args.output_dir, f"month_{int(month):02d}.nc")
    ds_month.to_netcdf(out_path)
    print(f"Saved: {out_path}")

