import streamlit as st
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np

# -----------------------------
# File paths for hot/cold start
# -----------------------------
FILES = {
    "Cold Start": "/mnt/climate/cas/project/mems/mvec/2013/vkt_CS_allhourly_gridded_1km.nc",
    "Hot Start": "/mnt/climate/cas/project/mems/mvec/2013/vkt_hot_allhourly_gridded_1km.nc"
}

# ---------------------------------------
# Sidebar: Emission type and variable select
# ---------------------------------------
st.sidebar.title("Controls")

emission_type = st.sidebar.radio("Select Emission Type", list(FILES.keys()))
file_path = FILES[emission_type]

# Load dataset
ds = xr.open_dataset(file_path)

# Get all variable names (not coords)
variable_names = list(ds.data_vars)
selected_var = st.sidebar.selectbox("Select Variable", variable_names)

# Time step selection
time_steps = ds.dims["time"]
selected_hour = st.sidebar.slider("Select Hour (0-23)", 0, time_steps - 1, 0)

# ---------------------------------------
# Main Panel: Display title, plot, etc.
# ---------------------------------------
st.title("Vehicle Emission Viewer (Hot/Cold Start)")
st.markdown(f"**Emission Type:** {emission_type}")
st.markdown(f"**Variable:** `{selected_var}` at hour `{selected_hour}`")

# Extract and plot data
data = ds[selected_var].isel(time=selected_hour)

fig, ax = plt.subplots(figsize=(8, 6))
img = ax.imshow(data, origin="lower", cmap="viridis")
plt.colorbar(img, ax=ax, label="Emission Units")
ax.set_title(f"{selected_var} - Hour {selected_hour}")

st.pyplot(fig)

