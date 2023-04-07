# Generating future synthetic baseball seasons
#### Christopher Callahan
#### Christopher.W.Callahan.GR@dartmouth.edu

# dependencies

import xarray as xr
import numpy as np
import sys
import os
import datetime
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.gridspec as gridspec
import seaborn as sns
from matplotlib import colors

# data locations
loc_out = "../Data/SyntheticSeasons/"
loc_panel = "../Data/Panel/"

# Years
y1 = 2015
y2 = 2100

# Read panel
y1_panel = 1954
y2_panel = 2019
y1_final = 1962 # full adoption of 162-game season
data = pd.read_csv(loc_panel+"baseball_climate_data_"+str(y1_panel)+"-"+str(y2_panel)+".csv")
data["tx"] = data.tmax_hadisd
data["total_hr"] = data.visitor_hr + data.home_hr
data = data.loc[data.year.values >= y1_final,:]

# climatology
y1_clm = 2000
y2_clm = 2019
data_clm = data.loc[(data.year.values>=y1_clm)&(data.year.values<=y2_clm),:]

# Set up data for synthetic seasons
np.random.seed(100)
ngame = np.arange(1,81+1,1)
yrs = np.arange(y1,y2+1,1)
nboot = 250 # number of random realizations
parks_uq = np.unique(data_clm.loc[data_clm.year.values==2018,:].park_id.values)
night_fraction1 = (data_clm.groupby("park_id").agg("mean"))["daynight"].reset_index()
night_fraction = night_fraction1.iloc[[x in parks_uq for x in night_fraction1.park_id.values],:].reset_index().drop(columns="index")

dome_fraction1 = (data_clm.dropna(subset=["dome_status"]).groupby("park_id").agg("mean"))["dome_status"].reset_index()
dome_fraction = dome_fraction1.iloc[[x in parks_uq for x in dome_fraction1.park_id.values],:].reset_index().drop(columns="index")
# ARL02 has a dome in the future (post-2020), but not the past
# so we'll set it equal to average of the rest of the retractable roof stadiums
dome_foravg = dome_fraction.dome_status.values
dome_fraction.loc[dome_fraction.park_id=="ARL02","dome_status"] = np.mean(dome_foravg[(dome_foravg!=0)&(dome_foravg!=1)])
print(dome_fraction)
sys.exit()
parkgames_dates = xr.DataArray(np.zeros((nboot,len(parks_uq),len(ngame),len(yrs))),
                               coords=[np.arange(1,nboot+1,1),parks_uq,ngame,yrs],
                               dims=["boot","park","ngame","year"])
parkgames_daynight = xr.DataArray(np.zeros((nboot,len(parks_uq),len(ngame),len(yrs))),
                               coords=[np.arange(1,nboot+1,1),parks_uq,ngame,yrs],
                               dims=["boot","park","ngame","year"])
parkgames_dome = xr.DataArray(np.zeros((nboot,len(parks_uq),len(ngame),len(yrs))),
                               coords=[np.arange(1,nboot+1,1),parks_uq,ngame,yrs],
                               dims=["boot","park","ngame","year"])



# loop through realizations, set schedules
for b in np.arange(0,nboot,1):
    if np.mod(b,50)==0:
        print(b)

    for pp in np.arange(0,len(parks_uq),1):
        night_frac_p = night_fraction.loc[night_fraction.park_id==parks_uq[pp],"daynight"].values[0]
        dome_frac_p = dome_fraction.loc[dome_fraction.park_id==parks_uq[pp],"dome_status"].values[0]
        night_number = int(night_frac_p*len(ngame))
        dome_number = int(dome_frac_p*len(ngame))

        for yy in np.arange(0,len(yrs),1):
            y = yrs[yy]
            dates_regular_year = pd.date_range(start=str(y)+"-04-01",end=str(y)+"-09-30",freq="D")
            doy_year = np.array([x.dayofyear for x in dates_regular_year])
            doy_random = sorted(np.random.choice(doy_year,len(ngame),replace=False))
            parkgames_dates[b,pp,:,yy] = doy_random

            night_indices = np.random.choice(ngame-1,night_number,replace=False)
            parkgames_daynight[b,pp,night_indices,yy] = 1

            dome_indices = np.random.choice(ngame-1,dome_number,replace=False)
            parkgames_dome[b,pp,dome_indices,yy] = 1

# combine into ds and write out
games_ds = xr.Dataset({"game_dates":(["park","ngame","year","boot"],parkgames_dates.transpose("park","ngame","year","boot")),
                       "game_daynight":(["park","ngame","year","boot"],parkgames_daynight.transpose("park","ngame","year","boot")),
                       "game_dome":(["park","ngame","year","boot"],parkgames_dome.transpose("park","ngame","year","boot"))},
                       coords={"park":(["park"],parks_uq),
                               "year":(["year"],yrs),
                               "ngame":(["ngame"],ngame),
                               "boot":(["boot"],np.arange(1,nboot+1,1))})

games_ds.attrs["creation_date"] = str(datetime.datetime.now())
games_ds.attrs["created_by"] = "Christopher Callahan, Christopher.W.Callahan.GR@dartmouth.edu"
games_ds.attrs["variable_description"] = "game_dates: days-of-year of games; game_daynight: 0 for day game, 1 for night game; game_dome: 0 for no dome, 1 for dome"
games_ds.attrs["created_from"] = os.getcwd()+"/Generate_Future_Seasons.py"

fname_out = loc_out+"synthetic_baseball_seasons_dates_daynight_domes_"+str(y1)+"-"+str(y2)+".nc"
games_ds.to_netcdf(fname_out,mode="w")
print(fname_out,flush=True)
