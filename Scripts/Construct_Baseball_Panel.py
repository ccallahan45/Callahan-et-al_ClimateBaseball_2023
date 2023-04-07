# Construct panel of baseball game statistics and nearest-cell temperatures
#### Christopher Callahan
#### Christopher.W.Callahan.GR@dartmouth.edu

#### Mechanics
# Dependencies

import xarray as xr
import numpy as np
import sys
import os
import datetime
import pandas as pd
from sklearn import linear_model
import statsmodels.api as sm
import statsmodels.formula.api as smf


# Data location
loc_logs = "../Data/RetrosheetGameLogs/"
loc_era5 = "/path/to/era5/data/" #"/dartfs-hpc/rc/lab/C/CMIG/Data/Observations/ERA5/daily/"
loc_gpcp = "/path/to/gpcp/data/" #"/dartfs-hpc/rc/lab/C/CMIG/Data/Observations/GPCP/daily/"
loc_hadisd = "/path/to/hadisd/data/" #"/dartfs-hpc/rc/lab/C/CMIG/ccallahan/Baseball/Data/HadISD/"
loc_out = "../Data/Panel/"

# Years
y1 = 1954
y2 = 2019


#### Analysis

# read in game log data
gamelogs = pd.read_csv(loc_logs+"GameLogs_Temp_"+str(y1)+"-"+str(y2)+".csv",index_col=0)
panel = gamelogs.loc[gamelogs["year"].values>=y1,:]

# drop panel if ballparks not in CONUS
ids = panel["park_id"].values
parks_full = np.array(sorted(np.unique(ids)))

panel1 = panel.loc[(ids!="SJU01") & (ids!="MNT01") & (ids!="SYD01") & (ids!="TOK01") & (ids!="HON01"),:].reset_index()
panel1["lon_360"] = panel1["lon"].values % 360


# get latitude and longitude arrays
lat_full = panel1["lat"].values
lon_full = panel1["lon"].values

lat_min = 22
lat_max = 50
lon_min = -120
lon_max = -65
lon_min_360 = 240
lon_max_360 = 295


print(panel1.dome_status)
x = panel1.dome_status.values
print(len(x))
print(len(x[~np.isnan(x)]))

# Get HadISD park timeseries
print("getting HadISD park-level data",flush=True)
y1_hadisd = 1954
y2_hadisd = 2019
parks_hadisd = np.array(sorted([x[0:5] for x in os.listdir(loc_hadisd) if str(y1_hadisd)+"-"+str(y2_hadisd) in x]))
panel1["tmax_hadisd"] = np.full(len(lon_full),np.nan)
panel1["tmean_hadisd"] = np.full(len(lon_full),np.nan)
panel1["tmin_hadisd"] = np.full(len(lon_full),np.nan)
panel1["precip_hadisd"] = np.full(len(lon_full),np.nan)
panel1["windspeed_hadisd"] = np.full(len(lon_full),np.nan)
panel1["twmax_hadisd"] = np.full(len(lon_full),np.nan)
panel1["twmean_hadisd"] = np.full(len(lon_full),np.nan)
panel1["rh_hadisd"] = np.full(len(lon_full),np.nan)
panel1["q_hadisd"] = np.full(len(lon_full),np.nan)
panel1["slp_hadisd"] = np.full(len(lon_full),np.nan)
panel1["vapor_pressure_hadisd"] = np.full(len(lon_full),np.nan)
#panel1["density_hadisd"] = np.full(len(lon_full),np.nan)

vrs = ["tmax","tmean","tmin","precip","windspeed","twmax","twmean","rh","q","slp","vapor_pressure"] #,"density"
for p in parks_hadisd:
    if p in ids:
        print(p)
        hadisd_data = xr.open_dataset(loc_hadisd+p+"_HadISD_baseball_park_weather_"+str(y1_hadisd)+"-"+str(y2_hadisd)+".nc")
        panel_p = panel1.loc[panel1.park_id==p,:].reset_index()
        park_dates = np.array([pd.to_datetime(str(panel_p.iloc[x,:].year)+"-"+str(panel_p.iloc[x,:].month)+"-"+str(panel_p.iloc[x,:].day)) for x in panel_p.index])
        hadisd_park_time = hadisd_data.sel(time=park_dates)
        for v in vrs:
            if v in hadisd_park_time.data_vars:
                panel1.loc[panel1.park_id==p,v+"_hadisd"] = hadisd_park_time.data_vars[v].values



# get GPCP precip
print("reading gpcp precip",flush=True)
panel1["precip_gpcp"] = np.full(len(lon_full),np.nan)

y1_gpcp = 1997
y2_gpcp = 2015

for yy in np.arange(y1_gpcp,y2_gpcp+1,1):
    print(yy)

    # read in files for that year
    gpcp_y = xr.open_mfdataset(loc_gpcp+"gpcp_1dd_v1.2_p1d."+str(yy)+"*.nc",concat_dim="time").precip.load()
    lat_gpcp = gpcp_y.lat
    lon_gpcp = gpcp_y.lon

    for p in parks_full:
        # get lat and lon
        inds = (panel1.park_id==p)&(panel1.year==yy)
        if np.any(inds):
            panel_p = panel1.loc[inds,:].reset_index()
            lat_ind = np.argmin(np.abs(lat_gpcp.values - panel_p.lat[0]))
            lon_ind = np.argmin(np.abs(lon_gpcp.values - panel_p.lon[0]))
            precip_park = gpcp_y[:,lat_ind,lon_ind]
            tm_p = np.array([pd.to_datetime(str(panel_p.iloc[x,:].year)+"-"+str(panel_p.iloc[x,:].month)+"-"+str(panel_p.iloc[x,:].day)) for x in panel_p.index])
            panel1.loc[inds,"precip_gpcp"] = precip_park.loc[tm_p].values

# Get ERA5
print("reading ERA5 temperature",flush=True)
panel1["tmax_era5"] = np.full(len(lon_full),np.nan)
panel1["tmin_era5"] = np.full(len(lon_full),np.nan)

y1_era5 = 1979
y2_era5 = 2019

for yy in np.arange(y1_era5,y2_era5+1,1):
    print(yy)

    # max
    era_file = xr.open_dataset(loc_era5+"tasmax_"+str(yy)+".nc")
    tmax_in = xr.DataArray(era_file.data_vars["mx2t"])
    tmax = tmax_in[(tmax_in.time.dt.month>=3) & (tmax_in.time.dt.month<=10),:,:].loc[:,lat_max:lat_min,lon_min_360:lon_max_360]
    lat_era5 = tmax.coords["latitude"].values
    lon_era5 = tmax.coords["longitude"].values

    # min
    era_file = xr.open_dataset(loc_era5+"tasmin_"+str(yy)+".nc")
    tmin_in = xr.DataArray(era_file.data_vars["mn2t"])
    tmin = tmin_in[(tmin_in.time.dt.month>=3) & (tmin_in.time.dt.month<=10),:,:].loc[:,lat_max:lat_min,lon_min_360:lon_max_360]

    for p in parks_full:
        # get lat and lon
        inds = (panel1.park_id==p)&(panel1.year==yy)
        if np.any(inds):
            panel_p = panel1.loc[inds,:].reset_index()
            lat_ind = np.argmin(np.abs(lat_era5 - panel_p.lat[0]))
            lon_ind = np.argmin(np.abs(lon_era5 - panel_p.lon_360[0]))
            tmax_park = tmax[:,lat_ind,lon_ind]
            tmin_park = tmin[:,lat_ind,lon_ind]
            tm_p = np.array([pd.to_datetime(str(panel_p.iloc[x,:].year)+"-"+str(panel_p.iloc[x,:].month)+"-"+str(panel_p.iloc[x,:].day)) for x in panel_p.index])
            panel1.loc[inds,"tmax_era5"] = tmax_park.loc[tm_p].values
            panel1.loc[inds,"tmin_era5"] = tmin_park.loc[tm_p].values



print("other housekeeping",flush=True)

# Check if parks or teams don't appear very often
parks = list(set(panel1["park_id"].values))
pks = np.array([])
for t in np.arange(0,len(parks),1):
    panel_park = panel1.loc[panel1["park_id"] == parks[t],:]
    ngames = len(panel_park["year"].values)
    if ngames < 10:
        print(parks[t])
    else:
        pks = np.append(pks,parks[t])

ids = panel1["park_id"].values
panel2 = panel1.loc[(ids!="LBV01") & (ids!="LAS01") & (ids!="FTB01") & (ids!="WIL02"),:].reset_index()

teams = list(set(panel2["visitor"].values))
for t in np.arange(0,len(teams),1):
    panel_team = panel2.loc[panel2["visitor"] == teams[t],:]
    ngames = len(panel_team["year"].values)
    if ngames < 10:
        print(teams[t])


# Set unique numeric code for each park
panel3 = panel2.drop(columns=["level_0","index"])
panel3["parknum"] = np.zeros(len(panel3["year"].values))

ids_uq = list(set(panel3["park_id"].values))
for ii in np.arange(0,len(ids_uq),1):
    panel3.loc[panel3["park_id"].values == ids_uq[ii],"parknum"] = ii+1


# Set unique numeric code for each team

panel3["visitornum"] = np.zeros(len(panel3["visitor"].values))
panel3["homenum"] = np.zeros(len(panel3["home"].values))
visitors_uq = list(set(panel3["visitor"].values))
home_uq = list(set(panel3["home"].values))
for ii in np.arange(0,len(visitors_uq),1):
    panel3.loc[panel3["visitor"].values == visitors_uq[ii],"visitornum"] = ii+1
for ii in np.arange(0,len(home_uq),1):
    panel3.loc[panel3["home"].values == home_uq[ii],"homenum"] = ii+1


# Add team-year fixed effect

panel3["visitor_year"] = (panel3["year"].values*100 + panel3["visitornum"].values).astype(int)
panel3["home_year"] = (panel3["year"].values*100 + panel3["homenum"].values).astype(int)


# Add day-of-year effect

dayofyear = np.zeros(len(panel3["year"].values))
for ii in np.arange(0,len(panel3["year"].values),1):
    day = panel3.loc[ii,"day"]
    mon = panel3.loc[ii,"month"]
    yr = panel3.loc[ii,"year"]
    datetime = pd.to_datetime(str(day)+"/"+str(mon)+"/"+str(yr),dayfirst=True)
    doy = datetime.dayofyear
    dayofyear[ii] = doy

panel3["dayofyear"] = dayofyear


# Finally, incorporate dome information
# houston astrodome
# seattle kingdome
# olympic stadium, montreal
# metrodome, mpls
# tropicana field, st petersburg (tampa)
# miller park (MIL)
# minute maid park (HOU)
# marlins park (MIA)
# rogers centre (TOR)
# chase field (phoenix)
# t-mobile field (SEA)

dome_codes = ["HOU02","HOU03","MIA02","MIL06","SEA02","STP01","TOR02","PHO01","SEA03","MIN03","MON02"] #sea03 - "safeco field"?

dome = np.zeros(len(panel3["year"].values))
for ii in np.arange(0,len(dome),1):
    park = panel3.loc[ii,"park_id"]
    if park in dome_codes:
        dome[ii] = 1
panel3["dome"] = dome

# write out
fname = loc_out+"baseball_climate_data_"+str(y1)+"-"+str(y2)+".csv"
panel3.to_csv(fname,index=False)
print(fname,flush=True)

