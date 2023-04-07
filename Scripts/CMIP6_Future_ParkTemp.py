# Historical and future temperature at baseball stadiums
#### Christopher Callahan
#### Christopher.W.Callahan.GR@dartmouth.edu

# Dependencies

import xarray as xr
import numpy as np
import sys
import os
import datetime
import pandas as pd
from scipy import signal, stats
import statsmodels.api as sm
from statsmodels.formula.api import ols as reg
from functools import reduce

# locations
loc_cmip6 = "/path/to/cmip6/data/"
loc_logs = "../Data/RetrosheetGameLogs/"
loc_baseball_data = "../Data/"
loc_out = "../Data/CMIP6/BallparkTx/"

# years
y1_hist = 1950
y2_hist = 2014
y1_ssp = 2015
y2_ssp = 2100
y1 = 1950
y2 = 2100
y1_panel = 1954
y2_panel = 2019
mon1 = 3
mon2 = 10

# leap days
def xr_strip_leapdays(x):
    # This function removes leap days (February 29) from a timeseries x
    # The timeseries x is assumed to have a functional "time" coordinate in the xarray style
    x_noleap = x.sel(time=~((x.time.dt.month == 2) & (x.time.dt.day == 29)))
    return(x_noleap)

# Read panel
gamelogs = pd.read_csv(loc_logs+"GameLogs_Temp_"+str(y1_panel)+"-"+str(y2_panel)+".csv",index_col=0)
panel = gamelogs.loc[gamelogs["year"].values>=y1_panel,:]
ids = panel["park_id"].values
panel = panel.loc[(ids!="SJU01") & (ids!="MNT01") & (ids!="SYD01") & (ids!="TOK01") & (ids!="HON01"),:].reset_index()
parks = np.unique(panel["park_id"].values)
pks_final = np.array([])
for t in np.arange(0,len(parks),1):
    panel_park = panel.loc[panel["park_id"] == parks[t],:]
    ngames = len(panel_park["year"].values)
    if ngames >= 10:
        pks_final = np.append(pks_final,parks[t])

# now read overall park locations
parkloc = pd.read_csv(loc_baseball_data+"ParkLocations.csv")
# final parks -- just the ones that have lots of game data
parkloc_final = parkloc.loc[[x in pks_final for x in parkloc.PARKID.values],:]
# arrays
parks = parkloc_final["PARKID"].values
park_lat = parkloc_final["Latitude"].values
park_lon = parkloc_final["Longitude"].values
park_alt = parkloc_final["Altitude"].values

# new grid
res = 1
lon = np.arange(1,359.0+res,res)
latmin = 10.0
latmax = 70.0
lat = np.arange(latmin,latmax+res,res)

## experiments
exps = ["ssp370"] #["ssp126","ssp245","ssp370","ssp585"]

# hist models
hist_models = np.array([x for x in sorted(os.listdir(loc_cmip6+"historical/tasmax_day/")) if (x.endswith(".nc"))])
hist_models_prefix = np.array([x.split("_")[2]+"_"+x.split("_")[4] for x in hist_models])

for exp in exps:

    print(exp,flush=True)

    # ssp models
    ssp_models = np.array([x for x in sorted(os.listdir(loc_cmip6+exp+"/tasmax_day/")) if (x.endswith(".nc"))])
    ssp_models_prefix = np.array([x.split("_")[2]+"_"+x.split("_")[4] for x in ssp_models])

    # intersection of all three
    models = reduce(np.intersect1d,(hist_models_prefix,ssp_models_prefix))
    models_exclude = [] #["CNRM-CM6-1","FGOALS-g3","GFDL-CM4","GFDL-ESM4","IPSL-CM6A-LR","MPI-ESM1-2-LR","MPI-ESM1-2-HR"]
    models = np.array([x for x in models if x.split("_")[0] not in models_exclude])
    #existing_data = np.array([x.split("_")[0]+"_"+x.split("_")[1]+"_historical-"+exp for x in os.listdir(loc_out)])
    #models = np.array([x for x in models if x not in existing_data])
    print(models)

    # loop through models
    for m in models:
        print(m,flush=True)
        mname = m.split("_")[0]
        mreal = m.split("_")[1]

        model_hist = mname+"_historical_"+mreal
        model_ssp = mname+"_"+exp+"_"+mreal


        # read tasmax
        print("reading hist data",flush=True)
        tx_ds_hist = xr.open_mfdataset(loc_cmip6+"historical/tasmax_day/"+"tasmax_day"+"_"+model_hist+"*.nc",
                                    concat_dim="time")
        tm_hist = tx_ds_hist.coords["time"].load()
        lat_hist = tx_ds_hist.coords["lat"].load()
        tm_hist_ind = (tm_hist.dt.year>=y1_hist)&(tm_hist.dt.year<=y2_hist)
        lat_hist_ind = (lat_hist>=latmin)&(lat_hist<=latmax)
        tx_hist = tx_ds_hist.tasmax[tm_hist_ind,lat_hist_ind,:].load()
        if tx_hist.max()>200:
            tx_hist = tx_hist-273.15
        tx_hist_noleap = xr_strip_leapdays(tx_hist)

        print("reading ssp data",flush=True)
        tx_ds_ssp = xr.open_mfdataset(loc_cmip6+exp+"/tasmax_day/"+"tasmax_day"+"_"+model_ssp+"*.nc",
                                    concat_dim="time")
        tm_ssp = tx_ds_ssp.coords["time"].load()
        lat_ssp = tx_ds_ssp.coords["lat"].load()
        tm_ssp_ind = (tm_ssp.dt.year>=y1_ssp)&(tm_ssp.dt.year<=y2_ssp)
        lat_ssp_ind = (lat_ssp>=latmin)&(lat_ssp<=latmax)
        tx_ssp = tx_ds_ssp.tasmax[tm_ssp_ind,lat_ssp_ind,:].load()
        if tx_ssp.max()>200:
            tx_ssp = tx_ssp-273.15
        tx_ssp_noleap = xr_strip_leapdays(tx_ssp)

        # concat
        tx_final = xr.concat([tx_hist_noleap,tx_ssp_noleap],dim="time")

        # regrid
        if np.amin(tx_final.coords["lon"].values)<0:
            tx_final.coords["lon"] = tx_final.coords["lon"].values % 360
        tx_regrid = tx_final.interp(lat=lat,lon=lon)
        del(tx_final)

        # establish calendar
        tm = xr.cftime_range(start=str(y1)+"-01-01",end=str(y2)+"-12-31",
                                freq="D",calendar="noleap")
        if len(tx_regrid.time.values)<len(tm):
            continue
        tx_regrid.coords["time"] = tm

        # create data -- park x time
        print("calculating park-level data",flush=True)
        park_tx = xr.DataArray(np.full((len(parks),len(tm)),np.nan),
                                coords=[parks,tm],dims=["park","time"])

        # loop through parks and calculate points around each park index
        for p in np.arange(0,len(parks),1):
            latp = park_lat[p]
            lonp = np.mod(park_lon[p],360)
            lat_ind = np.argmin(np.abs(lat - latp))
            lon_ind = np.argmin(np.abs(lon - lonp))
            # three values in both lat and lon directions, then mean
            park_tx[p,:] = tx_regrid[:,(lat_ind-1):(lat_ind+2),(lon_ind-1):(lon_ind+2)].mean(dim=["lat","lon"])

        # combine into ds and write out
        park_ds = xr.Dataset({"tx":(["park","time"],park_tx)},
                               coords={"park":(["park"],parks),
                                        "time":(["time"],tm)})

        park_ds.attrs["creation_date"] = str(datetime.datetime.now())
        park_ds.attrs["created_by"] = "Christopher Callahan, Christopher.W.Callahan.GR@dartmouth.edu"
        park_ds.attrs["variable_description"] = "Daily tx from CMIP6 historical and "+exp
        park_ds.attrs["created_from"] = os.getcwd()+"/CMIP6_Future_ParkTemp.py"
        park_ds.attrs["spatial_averaging"] = "9 grid cells closest to park location"

        fname_out = loc_out+m+"_historical_"+exp+"_baseball_park_tx_"+str(y1)+"-"+str(y2)+".nc"
        park_ds.to_netcdf(fname_out,mode="w")
        print(fname_out,flush=True)
