# Timeseries of weather data at each park from HadISD data
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
from scipy import spatial
from scipy import stats

# Data location
loc_logs = "../Data/RetrosheetGameLogs/"
loc_baseball_data = "../Data/"
loc_hadisd_metadata = "/path/to/hadisd/metadata/"
loc_hadisd_main = "/path/to/hadisd/data/"
loc_hadisd_derived = "/path/to/hadisd/derived-data/"
loc_out = "../Data/HadISD/"

#### Analysis
y1_final = 1954
y2_final = 2019

# Read panel
gamelogs = pd.read_csv(loc_logs+"GameLogs_Temp_"+str(y1_final)+"-"+str(y2_final)+".csv",index_col=0)
panel = gamelogs.loc[gamelogs["year"].values>=y1_final,:]
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

print(len(parks))

# Get HadISD metadata
station_info = pd.read_csv(loc_hadisd_metadata+"station_list.csv",
                          header=None,
                           names=["station","lat","lon","altitude"])
station_info["location"] = [np.array([station_info.iloc[x,:].loc["lat"],station_info.iloc[x,:].loc["lon"]]) for x in station_info.index]

# Now get X number of stations that are closest, in euclidean distance, to each park
n_neighbors = 5
# stations are relatively dense in north america

import warnings
warnings.filterwarnings("ignore",category=FutureWarning,message="'base' in .resample()")

vrs = ["tx","tm","tn","p","ws","twx","twm","rh","q","slp","vapor_pres","density"]
# tmax, tmean, tmin, precip, wet bulb max,
# wet bulb mean, relative humidity, specific humidity
y1_full = 1931
y2_full = 2021
full_time = pd.date_range(start=str(y1_full)+"-01-01",end=str(y2_full)+"-12-31",freq="D")
final_time = pd.date_range(start=str(y1_final)+"-01-01",end=str(y2_final)+"-12-31",freq="D")

parks_already_done = [] # np.array([x[0:5] for x in os.listdir(loc_out)])
for p in np.arange(0,len(parks),1):
    idp = parks[p]
    print(str(idp)+" ("+str(p)+")",flush=True)

    if idp not in parks_already_done:

        latp = park_lat[p]
        lonp = park_lon[p]
        park_loc = np.array([latp,lonp])
        station_info_park = station_info.copy()

        # find N closest stations
        station_locs = station_info_park.location.values
        station_info_park["distance"] = [spatial.distance.euclidean(park_loc,x) for x in station_locs]
        station_info_park_sorted = station_info_park.sort_values(by="distance")
        station_ids = station_info_park_sorted.iloc[0:n_neighbors,:].loc[:,"station"].values
        station_distances = station_info_park_sorted.iloc[0:n_neighbors,:].loc[:,"distance"].values

        # create array for data and loop -- not all stations appear to have the same time info
        # so xr.open_mfdataset gets tough
        full_data = xr.DataArray(np.full((len(vrs),len(station_ids),len(full_time)),np.nan),
                                 coords=[vrs,station_ids,full_time],
                                 dims=["var","station","time"])
        distance_wgts = xr.DataArray(1.0/station_distances,coords=[station_ids],dims=["station"])

        # get individual station data
        #print("getting individual stations",flush=True)
        for x in station_ids:
            print(x,flush=True)
            # first the main file
            ds = xr.open_dataset(loc_hadisd_main+"hadisd.3.1.2.202106p_19310101-20210701_"+x+".nc")
            t = ds.temperatures
            t = t.where(t>-500,np.nan) # fill values nan
            precip = ds.precip1_depth
            precip = precip.where(t>-500,np.nan)
            wind = ds.windspeeds.where(t>-500,np.nan)
            
            #x1 = ds.precip1_depth
            #x2 = ds.precip12_depth
            #x3 = ds.precip24_depth
            
            # now the derived humidity file
            ds2 = xr.open_dataset(loc_hadisd_derived+"hadisd.3.1.2.202106p_19310101-20210701_"+x+"_humidity.nc")
            rh = ds2.relative_humidity
            q = ds2.specific_humidity
            tw = ds2.wet_bulb_temperature
            slp = ds2.slp
            vapor_pres = ds2.vapor_pressure
            saturation_vapor_pres = ds2.saturation_vapor_pressure
            
            # convert to daily
            tx_daily = t.resample(time="D").max(dim="time")
            tn_daily = t.resample(time="D").min(dim="time")
            tm_daily = t.resample(time="D").mean(dim="time")
            del(t)
            p_daily = precip.resample(time="D").sum(dim="time")
            del(precip)
            rh_daily = rh.resample(time="D").mean(dim="time")
            del(rh)
            ws_daily = wind.resample(time="D").mean(dim="time")
            del(wind)
            q_daily = q.resample(time="D").mean(dim="time")
            del(q)
            twx_daily = tw.resample(time="D").max(dim="time")
            twm_daily = tw.resample(time="D").mean(dim="time")
            # other
            vp_daily = vapor_pres.resample(time="D").mean(dim="time")
            slp_daily = slp.resample(time="D").mean(dim="time")
            del([vapor_pres,slp])
            
            # density?
            pv = vp_daily*1.0
            pd = slp_daily - pv
            rd = 287.058
            rv = 461.495
            ## this calc is wrong because T needs to be in K not C
            density = (pd/(rd*tm_daily)) + (pv/(rv*tm_daily))

            # add to main data
            full_data.loc["tx",x,tx_daily.time] = tx_daily
            full_data.loc["tn",x,tn_daily.time] = tn_daily
            full_data.loc["tm",x,tm_daily.time] = tm_daily
            full_data.loc["p",x,p_daily.time] = p_daily
            full_data.loc["ws",x,ws_daily.time] = ws_daily
            full_data.loc["rh",x,rh_daily.time] = rh_daily
            full_data.loc["q",x,q_daily.time] = q_daily
            full_data.loc["twx",x,twx_daily.time] = twx_daily
            full_data.loc["twm",x,twm_daily.time] = twm_daily
            full_data.loc["slp",x,slp_daily.time] = slp_daily
            full_data.loc["vapor_pres",x,vp_daily.time] = vp_daily
            full_data.loc["density",x,density.time] = density

        # weight by distance and average, create final variables
        tmax = full_data.loc["tx",:,str(y1_final)+"-01-01":str(y2_final)+"-12-31"].weighted(distance_wgts).mean(dim="station")
        tmean = full_data.loc["tm",:,str(y1_final)+"-01-01":str(y2_final)+"-12-31"].weighted(distance_wgts).mean(dim="station")
        tmin = full_data.loc["tn",:,str(y1_final)+"-01-01":str(y2_final)+"-12-31"].weighted(distance_wgts).mean(dim="station")
        precip = full_data.loc["p",:,str(y1_final)+"-01-01":str(y2_final)+"-12-31"].weighted(distance_wgts).mean(dim="station")
        windspeed = full_data.loc["ws",:,str(y1_final)+"-01-01":str(y2_final)+"-12-31"].weighted(distance_wgts).mean(dim="station")
        twmax = full_data.loc["twx",:,str(y1_final)+"-01-01":str(y2_final)+"-12-31"].weighted(distance_wgts).mean(dim="station")
        twmean = full_data.loc["twm",:,str(y1_final)+"-01-01":str(y2_final)+"-12-31"].weighted(distance_wgts).mean(dim="station")
        rh = full_data.loc["rh",:,str(y1_final)+"-01-01":str(y2_final)+"-12-31"].weighted(distance_wgts).mean(dim="station")
        q = full_data.loc["q",:,str(y1_final)+"-01-01":str(y2_final)+"-12-31"].weighted(distance_wgts).mean(dim="station")
        slp = full_data.loc["slp",:,str(y1_final)+"-01-01":str(y2_final)+"-12-31"].weighted(distance_wgts).mean(dim="station")
        vp = full_data.loc["vapor_pres",:,str(y1_final)+"-01-01":str(y2_final)+"-12-31"].weighted(distance_wgts).mean(dim="station")
        density = full_data.loc["density",:,str(y1_final)+"-01-01":str(y2_final)+"-12-31"].weighted(distance_wgts).mean(dim="station")

        # create dataset and write out
        park_ds = xr.Dataset({"tmax":(["time"],tmax),
                              "tmean":(["time"],tmean),
                               "tmin":(["time"],tmin),
                               "precip":(["time"],precip),
                               "windspeed":(["time"],windspeed),
                               "twmax":(["time"],twmax),
                               "twmean":(["time"],twmean),
                               "rh":(["time"],rh),
                               "q":(["time"],q),
                               "slp":(["time"],slp),
                               "vapor_pressure":(["time"],vp),
                               "density":(["time"],density)},
                               coords={"time":(["time"],final_time)})

        park_ds.attrs["creation_date"] = str(datetime.datetime.now())
        park_ds.attrs["created_by"] = "Christopher Callahan, Christopher.W.Callahan.GR@dartmouth.edu"
        park_ds.attrs["variable_description"] = "tmax, tmean, tmin, precip, windspeed, wet bulb max, wet bulb mean, rh, q, slp, vapor_pres, density"
        park_ds.attrs["created_from"] = os.getcwd()+"/Process_Park_HadISD_Data.py"
        park_ds.attrs["station_averaging"] = str(n_neighbors)+" nearest neighbors, inverse distance weighting"

        fname_out = loc_out+idp+"_HadISD_baseball_park_weather_"+str(y1_final)+"-"+str(y2_final)+".nc"
        park_ds.to_netcdf(fname_out,mode="w")
        print(fname_out,flush=True)
