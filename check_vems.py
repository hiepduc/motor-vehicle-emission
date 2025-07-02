#!/usr/bin/env python3

''' This program reads in the speciation tool motor vehicle files
    to determine the diurnal pattern of 
    each species.
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

# First read the netCDF CB6 EDMS data as provided by speciation tool
#
#sptCB6dir = '/mnt/scratch_lustre/ar_soup_scratch/Speciation_tools/EDMS_Extraction/6-on-road_KgHr/petrol-exhaust/CB6R3_AE7/netcdf_output_grid/'
#sptCB6dir2 = '/mnt/scratch_lustre/ar_soup_scratch/Speciation_tools/EDMS_Extraction/6-on-road_KgHr/diesel-exhaust/CB6R3_AE7/netcdf_output_grid/'
sptCB6dir = '/mnt/climate/cas/ar_data/EDMS/2013/EDMS_Extraction/6-on-road_KgHr/petrol-exhaust/CB6R3_AE7/netcdf_output_grid/'
sptCB6dir2 = '/mnt/climate/cas/ar_data/EDMS/2013/EDMS_Extraction/6-on-road_KgHr/diesel-exhaust/CB6R3_AE7/netcdf_output_grid/'
#petrolexhwday = sptCB6dir + 'petrol-exhaust_Lump_pd_CB6R3_AE7_Jan_WeekDayAmount.nc4'
petrolexhwday = sptCB6dir + 'petrol-exhaust_Lump_pd_CB6R3_AE7_Jan_WeekEndAmount.nc4'
petrolid = Dataset(petrolexhwday, 'r')
#dieselexhwday = sptCB6dir2 + 'diesel-exhaust_Lump_pd_CB6R3_AE7_Jan_WeekDayAmount.nc4'
dieselexhwday = sptCB6dir2 + 'diesel-exhaust_Lump_pd_CB6R3_AE7_Jan_WeekEndAmount.nc4'
dieselid = Dataset(dieselexhwday, 'r')
# List of all CB6 species
cb6_list = ['ALD2','ALDX','CH4','CO','ETHA','FORM','IOLE','ISOP','IVOC','NH3','NO','NO2','NVOL','OLE','PAR','PM10','PM2.5','SO2','TOL','UNR']
mol_wgtcb6 = [44.05,43.65,16.04,28.0,30.07,30.03,56.11,68.2,137.19,17.0,30.0,46.0,1.0,27.65,14.43,1.0,1.0,64.1,92.14,50.49]
len(cb6_list)
#factorem = 6.81943e-7*3000*3000   # To convert from ppm/min per m2 to g/sec/grid cell (3km x 3km)
#factorm = 1/factorem              # To convert from g/s/gridcell (3kmx3km) to ppm/min/m2
# using zip() to convert 2 lists to dictionary
mol_weight = dict(zip(cb6_list, mol_wgtcb6))
mol_weight.keys()    # List all the keys in the dictionary

# Extract from the netcdf file, the hourly emission of each species
emiss_dict = {}
diurnal_dict = {}
emiss_dict2= {}
diurnal_dict2 = {}
for specs in cb6_list:
   emiss_dict[specs] = petrolid.variables[specs] 
   emiss_dict2[specs] = dieselid.variables[specs] 
#   diurnal_dict[specs] = np.sum(emiss_dict[specs],axis=(1,2))/np.sum(emiss_dict[specs],axis=(0,1,2))
#   diurnal_dict2[specs] = np.sum(emiss_dict2[specs],axis=(1,2))/np.sum(emiss_dict2[specs],axis=(0,1,2))
   diurnal_dict[specs] = np.sum(emiss_dict[specs],axis=(1,2))
   diurnal_dict2[specs] = np.sum(emiss_dict2[specs],axis=(1,2))

#co_cb6 = wheid.variables['CO']
#tot_co = np.sum(co_cb6, axis=1)   # Sum over lat (northing) 273
#tot_coall = np.sum(tot_co, axis = 1)  # sum over lon (easting) 210

#tot_coall = np.sum(co_cb6, axis =(1,2)) # sum over lat and lon, the result is 24 hourly data 
#tot_co_norm = tot_coall/np.sum(tot_coall)  # normalise the diurnal

# Check the diurnal by plotting
tot_no2all = diurnal_dict['NO2'] + diurnal_dict2['NO2']
#x = np.array([datetime(2013, 7, 1, i, 0) for i in range(24)])
x = np.array([i for i in range(24)])
plt.plot(x, tot_no2all)
plt.title("Jan 2013 NO2 emission EDMS weekend (petrol + diesel)")
plt.show()

# Check the diurnal by plotting
tot_pm25all = diurnal_dict['PM2.5'] + diurnal_dict2['PM2.5']
#x = np.array([datetime(2013, 7, 1, i, 0) for i in range(24)])
x = np.array([i for i in range(24)])
plt.plot(x, tot_pm25all)
plt.title("Jan 2013 PM2.5 emission EDMS weekend (petrol + diesel)")
plt.show()

# Read mvems files and extract the diurnal emission for weekday and weekend enission from hot, cold, evaporative and nepm

# Directory where all the emission files resides
mvemdir = '/mnt/scratch_lustre/ar_vems_scratch/vems_forecasting_V1_20231204/outputs/emvem_NSW_1_10_48_ADR/NetCDF_Profiled_final/NSW_sub_X_2_Y_1_4km/CB6R4_CF2'
filepath = mvemdir + '/hot_emissions_vkt_grid_NSW_sub_X_2_Y_1_4km_hour_all_roadtype_all_vehcat_all_vehsize_all_fuel_all_fleet_MERI_2013_lump_CB6R4_CF2_month_hour_wewd_profiled.nc4'
#filepath = mvemdir + '/hot_emissions_vkt_grid_NSW_sub_X_2_Y_1_4km_hour_all_roadtype_all_vehcat_all_vehsize_all_fuel_all_fleet_EMVEM_lump_CB6R4_CF2_month_hour_wewd_profiled.nc4'
filepath2 = mvemdir + '/cold_emissions_vkt_grid_NSW_sub_X_2_Y_1_4km_hour_all_roadtype_all_vehcat_all_vehsize_all_fuel_all_fleet_MERI_2013_lump_CB6R4_CF2_month_hour_wewd_profiled.nc4'
filepath3 = mvemdir + '/nepm_emissions_vkt_grid_NSW_sub_X_2_Y_1_4km_hour_all_roadtype_all_vehcat_all_vehsize_all_fuel_all_fleet_MERI_2013_lump_CB6R4_CF2_month_hour_wewd_profiled.nc4'
print(filepath)
ccdf=Dataset(filepath,'r')
ccdf2=Dataset(filepath2,'r')
ccdf3=Dataset(filepath3,'r')

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






