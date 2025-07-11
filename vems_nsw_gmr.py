#!/usr/bin/env python3

''' This program reads in the vems emission files which are divided into 6 sections to combine them into single nsw emission files
    for hot, cold, nepm and evaporative emission files. Then emission in the  gmr domain is also generated
    Summary statisitcs are also calculated for some species
'''
import cartopy      # always use cartopy first
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import pandas as pd
from netCDF4 import Dataset
import datetime as dt  # Python standard library datetime  module
import numpy as np
from netCDF4 import Dataset  # http://code.google.com/p/netcdf4-python/
import matplotlib.pyplot as plt
import glob
import cartopy.feature as cfeature
from datetime import datetime
import os
import geopandas as gpd
import shapefile as shp
from shapely.geometry import mapping
import xarray
import utm
import rioxarray

df = gpd.read_file("nsw_state_polygon_shp/NSW_STATE_POLYGON_shp.dbf")
df.plot()
plt.show()

sf = shp.Reader("GMR_Shapefile/GMR_Shapefile.shp")
sf = shp.Reader("NSW_Shapefile_SA3/SA3_2016_AUST.shp")
plt.figure()
for shape in sf.shapeRecords():
   x = [i[0] for i in shape.shape.points[:]]
   y = [i[1] for i in shape.shape.points[:]]
   plt.plot(x,y)

plt.show()


# Read mvems files and extract the diurnal emission for weekday and weekend enission from hot, cold, evaporative and nepm

# Directory where all the emission files resides
mvemdir = '/mnt/scratch_lustre/ar_vems_scratch/vems_forecasting_V1_20231204/outputs/emvem_NSW_1_10_48_ADR/NetCDF_Profiled_final'
filepathx2y1 = mvemdir + '/NSW_sub_X_2_Y_1_4km/CB6R4_CF2/hot_emissions_vkt_grid_NSW_sub_X_2_Y_1_4km_hour_all_roadtype_all_vehcat_all_vehsize_all_fuel_all_fleet_MERI_2013_lump_CB6R4_CF2_month_hour_wewd_profiled.nc4'
#filepath = mvemdir + '/hot_emissions_vkt_grid_NSW_sub_X_2_Y_1_4km_hour_all_roadtype_all_vehcat_all_vehsize_all_fuel_all_fleet_EMVEM_lump_CB6R4_CF2_month_hour_wewd_profiled.nc4'
filepath2x2y1 = mvemdir + '/NSW_sub_X_2_Y_1_4km/CB6R4_CF2/cold_emissions_vkt_grid_NSW_sub_X_2_Y_1_4km_hour_all_roadtype_all_vehcat_all_vehsize_all_fuel_all_fleet_MERI_2013_lump_CB6R4_CF2_month_hour_wewd_profiled.nc4'
filepath3x2y1 = mvemdir + '/NSW_sub_X_2_Y_1_4km/CB6R4_CF2/nepm_emissions_vkt_grid_NSW_sub_X_2_Y_1_4km_hour_all_roadtype_all_vehcat_all_vehsize_all_fuel_all_fleet_MERI_2013_lump_CB6R4_CF2_month_hour_wewd_profiled.nc4'
filepath4x2y1 = mvemdir + '/NSW_sub_X_2_Y_1_4km/CB6R4_CF2/evaps_emissions_vkt_grid_NSW_sub_X_2_Y_1_4km_hour_all_roadtype_all_vehcat_all_vehsize_all_fuel_all_fleet_MERI_2013_lump_CB6R4_CF2_month_hour_wewd_profiled.nc4'
filepathx2y0 = mvemdir + '/NSW_sub_X_2_Y_0_4km/CB6R4_CF2/hot_emissions_vkt_grid_NSW_sub_X_2_Y_0_4km_hour_all_roadtype_all_vehcat_all_vehsize_all_fuel_all_fleet_MERI_2013_lump_CB6R4_CF2_month_hour_wewd_profiled.nc4'
filepath2x2y0 = mvemdir + '/NSW_sub_X_2_Y_0_4km/CB6R4_CF2/cold_emissions_vkt_grid_NSW_sub_X_2_Y_0_4km_hour_all_roadtype_all_vehcat_all_vehsize_all_fuel_all_fleet_MERI_2013_lump_CB6R4_CF2_month_hour_wewd_profiled.nc4'
filepath3x2y0 = mvemdir + '/NSW_sub_X_2_Y_0_4km/CB6R4_CF2/nepm_emissions_vkt_grid_NSW_sub_X_2_Y_0_4km_hour_all_roadtype_all_vehcat_all_vehsize_all_fuel_all_fleet_MERI_2013_lump_CB6R4_CF2_month_hour_wewd_profiled.nc4'
filepath4x2y0 = mvemdir + '/NSW_sub_X_2_Y_0_4km/CB6R4_CF2/evaps_emissions_vkt_grid_NSW_sub_X_2_Y_0_4km_hour_all_roadtype_all_vehcat_all_vehsize_all_fuel_all_fleet_MERI_2013_lump_CB6R4_CF2_month_hour_wewd_profiled.nc4'
filepathx1y1 = mvemdir + '/NSW_sub_X_1_Y_1_4km/CB6R4_CF2/hot_emissions_vkt_grid_NSW_sub_X_1_Y_1_4km_hour_all_roadtype_all_vehcat_all_vehsize_all_fuel_all_fleet_MERI_2013_lump_CB6R4_CF2_month_hour_wewd_profiled.nc4'
filepath2x1y1 = mvemdir + '/NSW_sub_X_1_Y_1_4km/CB6R4_CF2/cold_emissions_vkt_grid_NSW_sub_X_1_Y_1_4km_hour_all_roadtype_all_vehcat_all_vehsize_all_fuel_all_fleet_MERI_2013_lump_CB6R4_CF2_month_hour_wewd_profiled.nc4'
filepath3x1y1 = mvemdir + '/NSW_sub_X_1_Y_1_4km/CB6R4_CF2/nepm_emissions_vkt_grid_NSW_sub_X_1_Y_1_4km_hour_all_roadtype_all_vehcat_all_vehsize_all_fuel_all_fleet_MERI_2013_lump_CB6R4_CF2_month_hour_wewd_profiled.nc4'
filepath4x1y1 = mvemdir + '/NSW_sub_X_1_Y_1_4km/CB6R4_CF2/evaps_emissions_vkt_grid_NSW_sub_X_1_Y_1_4km_hour_all_roadtype_all_vehcat_all_vehsize_all_fuel_all_fleet_MERI_2013_lump_CB6R4_CF2_month_hour_wewd_profiled.nc4'
filepathx1y0 = mvemdir + '/NSW_sub_X_1_Y_0_4km/CB6R4_CF2/hot_emissions_vkt_grid_NSW_sub_X_1_Y_0_4km_hour_all_roadtype_all_vehcat_all_vehsize_all_fuel_all_fleet_MERI_2013_lump_CB6R4_CF2_month_hour_wewd_profiled.nc4'
filepath2x1y0 = mvemdir + '/NSW_sub_X_1_Y_0_4km/CB6R4_CF2/cold_emissions_vkt_grid_NSW_sub_X_1_Y_0_4km_hour_all_roadtype_all_vehcat_all_vehsize_all_fuel_all_fleet_MERI_2013_lump_CB6R4_CF2_month_hour_wewd_profiled.nc4'
filepath3x1y0 = mvemdir + '/NSW_sub_X_1_Y_0_4km/CB6R4_CF2/nepm_emissions_vkt_grid_NSW_sub_X_1_Y_0_4km_hour_all_roadtype_all_vehcat_all_vehsize_all_fuel_all_fleet_MERI_2013_lump_CB6R4_CF2_month_hour_wewd_profiled.nc4'
filepath4x1y0 = mvemdir + '/NSW_sub_X_1_Y_0_4km/CB6R4_CF2/evaps_emissions_vkt_grid_NSW_sub_X_1_Y_0_4km_hour_all_roadtype_all_vehcat_all_vehsize_all_fuel_all_fleet_MERI_2013_lump_CB6R4_CF2_month_hour_wewd_profiled.nc4'
filepathx0y1 = mvemdir + '/NSW_sub_X_0_Y_1_4km/CB6R4_CF2/hot_emissions_vkt_grid_NSW_sub_X_0_Y_1_4km_hour_all_roadtype_all_vehcat_all_vehsize_all_fuel_all_fleet_MERI_2013_lump_CB6R4_CF2_month_hour_wewd_profiled.nc4'
filepath2x0y1 = mvemdir + '/NSW_sub_X_0_Y_1_4km/CB6R4_CF2/cold_emissions_vkt_grid_NSW_sub_X_0_Y_1_4km_hour_all_roadtype_all_vehcat_all_vehsize_all_fuel_all_fleet_MERI_2013_lump_CB6R4_CF2_month_hour_wewd_profiled.nc4'
filepath3x0y1 = mvemdir + '/NSW_sub_X_0_Y_1_4km/CB6R4_CF2/nepm_emissions_vkt_grid_NSW_sub_X_0_Y_1_4km_hour_all_roadtype_all_vehcat_all_vehsize_all_fuel_all_fleet_MERI_2013_lump_CB6R4_CF2_month_hour_wewd_profiled.nc4'
filepath4x0y1 = mvemdir + '/NSW_sub_X_0_Y_1_4km/CB6R4_CF2/evaps_emissions_vkt_grid_NSW_sub_X_0_Y_1_4km_hour_all_roadtype_all_vehcat_all_vehsize_all_fuel_all_fleet_MERI_2013_lump_CB6R4_CF2_month_hour_wewd_profiled.nc4'
filepathx0y0 = mvemdir + '/NSW_sub_X_0_Y_0_4km/CB6R4_CF2/hot_emissions_vkt_grid_NSW_sub_X_0_Y_0_4km_hour_all_roadtype_all_vehcat_all_vehsize_all_fuel_all_fleet_MERI_2013_lump_CB6R4_CF2_month_hour_wewd_profiled.nc4'
filepath2x0y0 = mvemdir + '/NSW_sub_X_0_Y_0_4km/CB6R4_CF2/cold_emissions_vkt_grid_NSW_sub_X_0_Y_0_4km_hour_all_roadtype_all_vehcat_all_vehsize_all_fuel_all_fleet_MERI_2013_lump_CB6R4_CF2_month_hour_wewd_profiled.nc4'
filepath3x0y0 = mvemdir + '/NSW_sub_X_0_Y_0_4km/CB6R4_CF2/nepm_emissions_vkt_grid_NSW_sub_X_0_Y_0_4km_hour_all_roadtype_all_vehcat_all_vehsize_all_fuel_all_fleet_MERI_2013_lump_CB6R4_CF2_month_hour_wewd_profiled.nc4'
filepath4x0y0 = mvemdir + '/NSW_sub_X_0_Y_0_4km/CB6R4_CF2/evaps_emissions_vkt_grid_NSW_sub_X_0_Y_0_4km_hour_all_roadtype_all_vehcat_all_vehsize_all_fuel_all_fleet_MERI_2013_lump_CB6R4_CF2_month_hour_wewd_profiled.nc4'


print(filepathx2y1)
ccdf=Dataset(filepathx2y1,'r')
ccdf2=Dataset(filepath2x2y1,'r')
ccdf3=Dataset(filepath3x2y1,'r')

# Clip the netcdf file using NSW shape mask (df)
df.plot()
plt.show()
nc_hotx2y1 = xarray.open_dataset(filepathx2y1)
nc_hotx2y0 = xarray.open_dataset(filepathx2y0)
nc_coldx2y1 = xarray.open_dataset(filepath2x2y1)
nc_coldx2y0 = xarray.open_dataset(filepath2x2y0)
nc_nepmx2y1 = xarray.open_dataset(filepath3x2y1)
nc_nepmx2y0 = xarray.open_dataset(filepath3x2y0)
nc_evapsx2y1 = xarray.open_dataset(filepath4x2y1)
nc_evapsx2y0 = xarray.open_dataset(filepath4x2y0)
#clip_ncx2y1 = nc_hotx2y1.ACET.rio.set_crs("GDA94").rio.clip(df.geometry.apply(mapping), df.crs, all_touched = True)
#clip_ncx2y0 = nc_hotx2y0.ACET.rio.set_crs("GDA94").rio.clip(df.geometry.apply(mapping), df.crs, all_touched = True)
#clip_ncx2y1[1,12,0,:,:].transpose("lat", "lon").plot()  # select month =1, hour =12, weekday (0)
#plt.show()
#clip_ncx2y0[1,12,0,:,:].transpose("lat", "lon").plot()  # select month =1, hour =12, weekday (0)
#plt.show()

nc_hotx1y1 = xarray.open_dataset(filepathx1y1)
nc_hotx1y0 = xarray.open_dataset(filepathx1y0)
#nc_coldx1y1 = xarray.open_dataset(filepath2x1y1)
#nc_coldx1y0 = xarray.open_dataset(filepath2x1y0)
nc_nepmx1y1 = xarray.open_dataset(filepath3x1y1)
nc_nepmx1y0 = xarray.open_dataset(filepath3x1y0)
nc_evapsx1y1 = xarray.open_dataset(filepath4x1y1)
nc_evapsx1y0 = xarray.open_dataset(filepath4x1y0)

nc_hotx0y1 = xarray.open_dataset(filepathx0y1)
nc_hotx0y0 = xarray.open_dataset(filepathx0y0)
#nc_coldx0y1 = xarray.open_dataset(filepath2x0y1)
#nc_coldx0y0 = xarray.open_dataset(filepath2x0y0)
nc_nepmx0y1 = xarray.open_dataset(filepath3x0y1)
nc_nepmx0y0 = xarray.open_dataset(filepath3x0y0)
nc_evapsx0y1 = xarray.open_dataset(filepath4x0y1)
nc_evapsx0y0 = xarray.open_dataset(filepath4x0y0)

nc_hotx2y1.to_netcdf("nc_hotx2y1.nc")  # write to netcdf file
nc_hotx2y0.to_netcdf("nc_hotx2y0.nc")
nc_hotx1y1.to_netcdf("nc_hotx1y1.nc")  # write to netcdf file
nc_hotx1y0.to_netcdf("nc_hotx1y0.nc")
nc_hotx0y1.to_netcdf("nc_hotx0y1.nc")  # write to netcdf file
nc_hotx0y0.to_netcdf("nc_hotx0y0.nc")
dsx2 = xarray.open_mfdataset("nc_hotx2*.nc", concat_dim=['lat'], combine='nested')
dsx1 = xarray.open_mfdataset("nc_hotx1*.nc", concat_dim=['lat'], combine='nested')
dsx0 = xarray.open_mfdataset("nc_hotx0*.nc", concat_dim=['lat'], combine='nested')
nc_coldx2y1.to_netcdf("nc_coldx2y1.nc")  # write to netcdf file
nc_coldx2y0.to_netcdf("nc_coldx2y0.nc")
#nc_coldx1y1.to_netcdf("nc_coldx1y1.nc")  # write to netcdf file
#nc_coldx1y0.to_netcdf("nc_coldx1y0.nc")
#nc_coldx0y1.to_netcdf("nc_coldx0y1.nc")  # write to netcdf file
#nc_coldx0y0.to_netcdf("nc_coldx0y0.nc")
dsx2cold = xarray.open_mfdataset("nc_coldx2*.nc", concat_dim=['lat'], combine='nested')
#dsx1cold = xarray.open_mfdataset("nc_coldx1*.nc", concat_dim=['lat'], combine='nested')
#dsx0cold = xarray.open_mfdataset("nc_coldx0*.nc", concat_dim=['lat'], combine='nested')
nc_nepmx2y1.to_netcdf("nc_nepmx2y1.nc")  # write to netcdf file
nc_nepmx2y0.to_netcdf("nc_nepmx2y0.nc")
nc_nepmx1y1.to_netcdf("nc_nepmx1y1.nc")  # write to netcdf file
nc_nepmx1y0.to_netcdf("nc_nepmx1y0.nc")
nc_nepmx0y1.to_netcdf("nc_nepmx0y1.nc")  # write to netcdf file
nc_nepmx0y0.to_netcdf("nc_nepmx0y0.nc")
dsx2nepm = xarray.open_mfdataset("nc_nepmx2*.nc", concat_dim=['lat'], combine='nested')
dsx1nepm = xarray.open_mfdataset("nc_nepmx1*.nc", concat_dim=['lat'], combine='nested')
dsx0nepm = xarray.open_mfdataset("nc_nepmx0*.nc", concat_dim=['lat'], combine='nested')
nc_evapsx2y1.to_netcdf("nc_evapsx2y1.nc")  # write to netcdf file
nc_evapsx2y0.to_netcdf("nc_evapsx2y0.nc")
nc_evapsx1y1.to_netcdf("nc_evapsx1y1.nc")  # write to netcdf file
nc_evapsx1y0.to_netcdf("nc_evapsx1y0.nc")
nc_evapsx0y1.to_netcdf("nc_evapsx0y1.nc")  # write to netcdf file
nc_evapsx0y0.to_netcdf("nc_evapsx0y0.nc")
dsx2evaps = xarray.open_mfdataset("nc_evapsx2*.nc", concat_dim=['lat'], combine='nested')
dsx1evaps = xarray.open_mfdataset("nc_evapsx1*.nc", concat_dim=['lat'], combine='nested')
dsx0evaps = xarray.open_mfdataset("nc_evapsx0*.nc", concat_dim=['lat'], combine='nested')
# Select January weekday data
dsx2 = dsx2.rename({"Hour":"Time"})
dsx2.sortby("lat").transpose("Time","Month", "WD_WE_ID", "lat", "lon",...).to_netcdf("hot_x2y0_x2y1.nc")
dsx2.sel(Month=1,WD_WE_ID=1).transpose("Time","lat", "lon",...).sortby("lat").to_netcdf("hot_x2y0_x2y1_janwd.nc")
dsx2cold=dsx2cold.rename({"Hour":"Time"})
dsx2cold.sortby("lat").transpose("Time","Month", "WD_WE_ID","lat", "lon",...).to_netcdf("cold_x2y0_x2y1.nc")
dsx2cold.sel(Month=1,WD_WE_ID=1).transpose("Time","lat", "lon",...).sortby("lat").to_netcdf("cold_x2y0_x2y1_janwd.nc")
dsx2nepm=dsx2nepm.rename({"Hour":"Time"})
dsx2nepm.sortby("lat").transpose("Time","Month", "WD_WE_ID","lat", "lon",...).to_netcdf("nepm_x2y0_x2y1.nc")
dsx2nepm.sel(Month=1,WD_WE_ID=1).transpose("Time","lat", "lon",...).sortby("lat").to_netcdf("nepm_x2y0_x2y1_janwd.nc")
dsx2evaps=dsx2evaps.rename({"Hour":"Time"})
dsx2evaps.sortby("lat").transpose("Time","Month", "WD_WE_ID","lat", "lon",...).to_netcdf("evaps_x2y0_x2y1.nc")
dsx2evaps.sel(Month=1,WD_WE_ID=1).transpose("Time","lat", "lon",...).sortby("lat").to_netcdf("evaps_x2y0_x2y1_janwd.nc")
#dsx1.ACET[1,12,0,:,:].transpose("lat", "lon").sortby("lat").plot()
#plt.show()
dsx1=dsx1.rename({"Hour":"Time"})
dsx1.sortby("lat").transpose("Time","Month", "WD_WE_ID","lat", "lon",...).to_netcdf("hot_x1y0_x1y1.nc")
dsx1.sel(Month=1,WD_WE_ID=1).transpose("Time","lat", "lon",...).sortby("lat").to_netcdf("hot_x1y0_x1y1_janwd.nc")
#dsx1cold.rename({"Hour":"Time"}).to_netcdf("cold_x1y0_x1y1.nc")
#dsx1cold.sel(Month=1,WD_WE_ID=1).transpose("Hour","lat", "lon",...).sortby("lat").to_netcdf("coldx1y0_x1y1.nc")
dsx1nepm=dsx1nepm.rename({"Hour":"Time"})
dsx1nepm.sortby("lat").transpose("Time","Month", "WD_WE_ID","lat", "lon",...).to_netcdf("nepm_x1y0_x1y1.nc")
dsx1nepm.sel(Month=1,WD_WE_ID=1).transpose("Time","lat", "lon",...).sortby("lat").to_netcdf("nepm_x1y0_x1y1_janwd.nc")
dsx1evaps=dsx1evaps.rename({"Hour":"Time"})
dsx1evaps.sortby("lat").transpose("Time","Month", "WD_WE_ID","lat", "lon",...).to_netcdf("evaps_x1y0_x1y1.nc")
dsx1evaps.sel(Month=1,WD_WE_ID=1).transpose("Time","lat", "lon",...).sortby("lat").to_netcdf("evaps_x1y0_x1y1_janwd.nc")
#dsx0.ACET[1,12,0,:,:].transpose("lat", "lon").sortby("lat").plot()
#plt.show()
dsx0=dsx0.rename({"Hour":"Time"})
dsx0.sortby("lat").transpose("Time","Month", "WD_WE_ID","lat", "lon",...).to_netcdf("hot_x0y0_x0y1.nc")
dsx0.sel(Month=1,WD_WE_ID=1).transpose("Time","lat", "lon",...).sortby("lat").to_netcdf("hot_x0y0_x0y1_janwd.nc")
#dsx0cold.rename({"Hour":"Time"}).transpose("Hour","lat", "lon",...).sortby("lat").to_netcdf("cold_x0y0_x0y1.nc")
#dsx0cold.sel(Month=1,WD_WE_ID=1).transpose("Hour","lat", "lon",...).sortby("lat").to_netcdf("cold_x0y0_x0y1_janwd.nc")
dsx0nepm=dsx0nepm.rename({"Hour":"Time"})
dsx0nepm.sortby("lat").transpose("Time", "Month", "WD_WE_ID","lat", "lon",...).to_netcdf("nepm_x0y0_x0y1.nc")
dsx0nepm.sel(Month=1,WD_WE_ID=1).transpose("Time","lat", "lon",...).sortby("lat").to_netcdf("nepm_x0y0_x0y1_janwd.nc")
dsx0evaps=dsx0evaps.rename({"Hour":"Time"})
dsx0evaps.sortby("lat").transpose("Time","Month", "WD_WE_ID","lat", "lon",...).to_netcdf("evaps_x0y0_x0y1.nc")
dsx0evaps.sel(Month=1,WD_WE_ID=1).transpose("Time","lat", "lon",...).sortby("lat").to_netcdf("evaps_x0y0_x0y1_janwd.nc")

dsx1x2 = xarray.open_mfdataset(["hot_x1y0_x1y1_janwd.nc", "hot_x2y0_x2y1_janwd.nc"], concat_dim=['lon'], combine='nested')
#dsx1x2cold = xarray.open_mfdataset(["cold_x1y0_x1y1_janwd.nc", "cold_x2y0_x2y1_janwd.nc"], concat_dim=['lon'], combine='nested')
dsx1x2nepm = xarray.open_mfdataset(["nepm_x1y0_x1y1_janwd.nc", "nepm_x2y0_x2y1_janwd.nc"], concat_dim=['lon'], combine='nested')
dsx1x2evaps = xarray.open_mfdataset(["evaps_x1y0_x1y1_janwd.nc", "evaps_x2y0_x2y1_janwd.nc"], concat_dim=['lon'], combine='nested')
#dsx1x2.ACET[12,:,:].transpose("lat", "lon").sortby("lat").plot()
#plt.show()
dsx1x2.to_netcdf("hot_nsw_x1x2_janwd.nc")
#dsx1x2cold.to_netcdf("cold_nsw_x1x2.nc")
dsx1x2nepm.to_netcdf("nepm_nsw_x1x2_janwd.nc")
dsx1x2evaps.to_netcdf("evaps_nsw_x1x2_janwd.nc")

#dsx0x1x2 = xarray.open_mfdataset("hotx*.nc", concat_dim=['lon'], combine='nested')
dsx0x1x2 = xarray.open_mfdataset(["hot_x0y0_x0y1_janwd.nc", "hot_x1y0_x1y1_janwd.nc", "hot_x2y0_x2y1_janwd.nc"], concat_dim=['lon'], combine='nested')
dsx0x1x2year = xarray.open_mfdataset(["hot_x0y0_x0y1.nc", "hot_x1y0_x1y1.nc", "hot_x2y0_x2y1.nc"], concat_dim=['lon'], combine='nested')
#dsx0x1x2cold = xarray.open_mfdataset(["coldx0y0_x0y1.nc", "coldx1y0_x1y1.nc", "coldx2y0_x2y1.nc"], concat_dim=['lon'], combine='nested')
dsx0x1x2nepm = xarray.open_mfdataset(["nepm_x0y0_x0y1_janwd.nc", "nepm_x1y0_x1y1_janwd.nc", "nepm_x2y0_x2y1_janwd.nc"], concat_dim=['lon'], combine='nested')
dsx0x1x2nepmyear = xarray.open_mfdataset(["nepm_x0y0_x0y1.nc", "nepm_x1y0_x1y1.nc", "nepm_x2y0_x2y1.nc"], concat_dim=['lon'], combine='nested')
dsx0x1x2evaps = xarray.open_mfdataset(["evaps_x0y0_x0y1_janwd.nc", "evaps_x1y0_x1y1_janwd.nc", "evaps_x2y0_x2y1_janwd.nc"], concat_dim=['lon'], combine='nested')
dsx0x1x2evapsyear = xarray.open_mfdataset(["evaps_x0y0_x0y1.nc", "evaps_x1y0_x1y1.nc", "evaps_x2y0_x2y1.nc"], concat_dim=['lon'], combine='nested')
#dsx0x1x2.NO2[12,:,:].transpose("lat", "lon").sortby("lat").plot()
#plt.show()
dsx0x1x2.to_netcdf("hot_nsw_x0x1x2_janwd.nc")
dsx0x1x2year.to_netcdf("hot_nsw_x0x1x2.nc")
#dsx0x1x2cold.to_netcdf("cold_nsw_x0x1x2.nc")
dsx0x1x2nepm.to_netcdf("nepm_nsw_x0x1x2_janwd.nc")
dsx0x1x2nepmyear.to_netcdf("nepm_nsw_x0x1x2.nc")
dsx0x1x2evaps.to_netcdf("evaps_nsw_x0x1x2_janwd.nc")
dsx0x1x2evapsyear.to_netcdf("evaps_nsw_x0x1x2.nc")

# Clip the vems emission within the GMR
min_lon = 149.835205
min_lat = -34.66993
max_lon = 152.1400795
max_lat = -32.25476

# UTM coordinate in EDMS 2013 is  UTM zone 56S GDA94, in metre
# xmin=149.8295
# xmax=152.1395
# ymin=-34.6735
# ymax=-32.2165
# Grid Easting from........................210 to 419
# equivalent to cells......................1 to 210
# Grid Northing from.......................6159 to 6431
# equivalent to cells......................1 to 273

import utm
utm.to_latlon(210000, 6159000, 56, northern=False)
utm.to_latlon(419000, 6431000, 56, northern=False)

ds_gmr_hot = dsx0x1x2.sel(lat=slice(min_lat,max_lat), lon=slice(min_lon, max_lon))
ds_gmr_hotyear = dsx0x1x2year.sel(lat=slice(min_lat,max_lat), lon=slice(min_lon, max_lon))
#ds_gmr_cold = dsx0x1x2cold.sel(lat=slice(min_lat,max_lat), lon=slice(min_lon, max_lon))
ds_gmr_nepm = dsx0x1x2nepm.sel(lat=slice(min_lat,max_lat), lon=slice(min_lon, max_lon))
ds_gmr_nepmyear = dsx0x1x2nepmyear.sel(lat=slice(min_lat,max_lat), lon=slice(min_lon, max_lon))
ds_gmr_evaps = dsx0x1x2evaps.sel(lat=slice(min_lat,max_lat), lon=slice(min_lon, max_lon))
ds_gmr_evapsyear = dsx0x1x2evapsyear.sel(lat=slice(min_lat,max_lat), lon=slice(min_lon, max_lon))
dsx2cold = xarray.open_dataset("cold_x2y0_x2y1_janwd.nc")
dsx2_gmr_cold = dsx2cold.sel(lat=slice(min_lat,max_lat), lon=slice(min_lon, max_lon))
dsx2coldyear = xarray.open_dataset("cold_x2y0_x2y1.nc")
dsx2_gmr_coldyear = dsx2coldyear.sel(lat=slice(min_lat,max_lat), lon=slice(min_lon, max_lon))

# Save netcdf files of GMR hot, cold, evap, nepm emission
ds_gmr_hot.to_netcdf("hot_gmr_x0x1x2_janwd.nc")
ds_gmr_hotyear.to_netcdf("hot_gmr_x0x1x2.nc")
#ds_gmr_cold.to_netcdf("cold_gmr_x0x1x2.nc")
dsx2_gmr_cold.to_netcdf("cold_gmr_x2_janwd.nc")
dsx2_gmr_coldyear.to_netcdf("cold_gmr_x2.nc")
ds_gmr_nepm.to_netcdf("nepm_gmr_x0x1x2_janwd.nc")
ds_gmr_nepmyear.to_netcdf("nepm_gmr_x0x1x2.nc")
ds_gmr_evaps.to_netcdf("evaps_gmr_x0x1x2_janwd.nc")
ds_gmr_evapsyear.to_netcdf("evaps_gmr_x0x1x2.nc")

# Plot the emission of some species i the GMR
ds_gmr_hot.NO2[12,:,:].plot()
plt.show()
ds_gmr_hot.PM10[12,:,:].plot()
plt.show()
#ds_gmr_cold["PM2.5"][12,:,:].plot()
#plt.show()
#dsx2_gmr_cold["PM2.5"][12,:,:].plot()
#plt.show()
ds_gmr_nepm["PM2.5"][12,:,:].plot()
plt.show()

ds_gmr_evaps["ISOP"][12,:,:].plot()
plt.show()

# Statistics of emission (total)

np.sum(ds_gmr_hot.NO2,axis=(1,2))
# GMR
np.sum(ds_gmr_hot.NO2,axis=(1,2)).plot()
plt.show()
np.sum(ds_gmr_hot.NO2,axis=(1,2)).compute()
np.sum(ds_gmr_hot.NO2,axis=(1,2)).sum().compute()

#np.sum(ds_gmr_cold.NO2,axis=(1,2)).plot()
#plt.show()
#np.sum(ds_gmr_cold.NO2,axis=(1,2)).compute()
#np.sum(ds_gmr_cold.NO2,axis=(1,2)).sum().compute()

np.sum(dsx2_gmr_cold.NO2,axis=(1,2)).plot()
plt.show()
np.sum(dsx2_gmr_cold.NO2,axis=(1,2)).compute()
np.sum(dsx2_gmr_cold.NO2,axis=(1,2)).sum().compute()

np.sum(ds_gmr_hot["PM2.5"],axis=(1,2)).plot()
plt.show()
np.sum(ds_gmr_hot["PM2.5"],axis=(1,2)).compute()
np.sum(ds_gmr_hot["PM2.5"],axis=(1,2)).sum().compute()

#np.sum(ds_gmr_cold["PM2.5"],axis=(1,2)).plot()
#plt.show()
#np.sum(ds_gmr_cold["PM2.5"],axis=(1,2)).compute()
#np.sum(ds_gmr_cold["PM2.5"],axis=(1,2)).sum().compute()

#np.sum(dsx2_gmr_cold["PM2.5"],axis=(1,2)).plot()
#plt.show()
#np.sum(dsx2_gmr_cold["PM2.5"],axis=(1,2)).compute()
#np.sum(dsx2_gmr_cold["PM2.5"],axis=(1,2)).sum().compute()

np.sum(ds_gmr_nepm["PM2.5"],axis=(1,2)).plot()
plt.show()
np.sum(ds_gmr_nepm["PM2.5"],axis=(1,2)).compute()
np.sum(ds_gmr_nepm["PM2.5"],axis=(1,2)).sum().compute()

# NSW
np.sum(dsx0x1x2.NO2,axis=(1,2)).plot()
plt.show()
np.sum(dsx0x1x2.NO2,axis=(1,2)).compute()
np.sum(dsx0x1x2.NO2,axis=(1,2)).sum().compute()
np.sum(dsx0x1x2["PM2.5"],axis=(1,2)).sum().compute()

#np.sum(dsx0x1x2cold.NO2,axis=(1,2)).plot()
#plt.show()
#np.sum(dsx0x1x2cold.NO2,axis=(1,2)).compute()
#np.sum(dsx0x1x2cold.NO2,axis=(1,2)).sum().compute()
#np.sum(dsx0x1x2cold["PM2.5"],axis=(1,2)).sum().compute()

np.sum(dsx0x1x2nepm.PM10,axis=(1,2)).plot()
plt.show()
np.sum(dsx0x1x2nepm.PM10,axis=(1,2)).compute()
np.sum(dsx0x1x2nepm.PM10,axis=(1,2)).sum().compute()
np.sum(dsx0x1x2nepm["PM2.5"],axis=(1,2)).sum().compute()

np.sum(dsx0x1x2evaps.TOL,axis=(1,2)).plot()
plt.show()
np.sum(dsx0x1x2evaps.ISOP,axis=(1,2)).compute()
np.sum(dsx0x1x2evaps.ISOP,axis=(1,2)).sum().compute()
np.sum(dsx0x1x2evaps["TOL"],axis=(1,2)).sum().compute()

#no2jan = ccdf.variables['NO2'][1,:,1,:,:]  # [month, hour, wd_we_id, lon, lat]    wd_wk_id = 0 (weekday) and 1 (weeend)
#no2jan2 = ccdf2.variables['NO2'][1,:,1,:,:]
no2jan = ccdf.variables['NO2'][1,:,0,:,:]  # [month, hour, wd_we_id, lon, lat]    wd_wk_id = 0 (weekday) and 1 (weeend)
no2jan2 = ccdf2.variables['NO2'][1,:,0,:,:]
pm25jan = ccdf3.variables['PM2.5'][1,:,1,:,:]
#pm25jan = ccdf3.variables['PM2.5'][1,:,0,:,:]
#no2janall = no2jan + no2jan2 + no2jan3
#no2janall = no2jan + no2jan2
no2jan.shape

#no2jan = ccdf.variables['NO2'][1,6,1,:,:]
#no2jantot =  np.sum(no2jan,axis=(1,2))/np.sum(no2jan,axis=(0,1,2))
no2jantot =  np.sum(no2jan,axis=(1,2)) + np.sum(no2jan2,axis=(1,2))
#no2jantot =  np.sum(no2jan,axis=(1,2)) 
#no2jantot2 =  np.sum(no2jan2,axis=(1,2))/np.sum(no2jan2,axis=(0,1,2))
#no2jantot3 =  np.sum(no2jan3,axis=(1,2))/np.sum(no2jan3,axis=(0,1,2))
#x = np.array([datetime(2013, 1, 1, i, 0) for i in range(24)])
x = np.array([i for i in range(24)])
plt.plot(x, no2jantot)
#plt.title("Jan 2013 NO2 hot + cold emission weekend")
plt.title("Jan 2013 NO2 hot + cold emission weekday")
plt.show()

pm25jantot =   np.sum(pm25jan,axis=(1,2))
plt.plot(x, pm25jantot)
plt.title("Jan 2013 PM2.5 NEPM emission (kg/hour) weekend")
plt.show()

# Check the emission of species by plotting spatially
lats = ccdf.variables['lat']
lons = ccdf.variables['lon']
extent = [136.0, 161.0, -45.0, -20.0]
#co_whe = species_dict['CO']  # select CO emission

fig=plt.figure()
ax = plt.axes(projection=ccrs.PlateCarree())
#cp=ax.contourf(lons[:], lats[:], whe[:,:], extent=extent)  # too big and causes segmentation fault
#cp=ax.contourf(lons[::10], lats[::10], no2jan[::10,::10], extent=extent)
cp=ax.contourf(lats[:], lons[:], no2jan[1,:,:], extent=extent)
fig.colorbar(cp)
ax.set_title('NO2 Contour Plot')
ax.set_xlabel('lon')
ax.set_ylabel('lat')
ax.coastlines()
ax.add_feature(cfeature.STATES)
#ax.add_feature(cfeature.BORDERS)
#ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.RIVERS)
plt.xticks([135,140,145,150,155,160])
plt.yticks([-45,-40,-35,-30,-25,-20])
ax.set_extent([135, 161, -45, -20], crs=ccrs.PlateCarree())
plt.show()





