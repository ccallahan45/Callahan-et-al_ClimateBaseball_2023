# Changes in home run number due to future climate change
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
from rasterio import features
from affine import Affine
import geopandas as gp
import descartes
from scipy import signal, stats
import statsmodels.api as sm
from statsmodels.formula.api import ols as reg
from functools import reduce

# Data locations
loc_cmip6_tx = "../Data/CMIP6/BallparkTx/"
loc_cmip6_gmst = "../Data/CMIP6/GMST/"
loc_panel = "../Data/Panel/"
loc_seasons = "../Data/SyntheticSeasons/"
loc_reg = "../Data/RegressionResults/"
loc_hadisd = "../Data/HadISD/"
loc_out = "../Data/HRProjections/"

# years
y1_cmip6 = 1950
y2_cmip6 = 2100
y1_future = 2020
y2_future = 2100
y1_panel = 1954
y2_panel = 2019
y1_clm = 2000  # climatology/baseline
y2_clm = 2019
mon1 = 3
mon2 = 10

print('getting initial data',flush=True)

# read panel data
panel = pd.read_csv(loc_panel+"baseball_climate_data_"+str(y1_panel)+"-"+str(y2_panel)+".csv")
#print(panel)

# regression coefficients
coefs_df = pd.read_csv(loc_reg+"homeruns_tx_poisson_gametypes.csv",index_col=0)
# resample to generate distribution
np.random.seed(100)
nboot = 1000
coef_day_nodome = np.random.normal(loc=coefs_df.loc[coefs_df.types=="day_nodome","beta"].values[0],
                                   scale=coefs_df.loc[coefs_df.types=="day_nodome","se"].values[0],
                                   size=nboot)
coef_day_dome = np.random.normal(loc=coefs_df.loc[coefs_df.types=="day_dome","beta"].values[0],
                                   scale=coefs_df.loc[coefs_df.types=="day_dome","se"].values[0],
                                   size=nboot)
coef_night_nodome = np.random.normal(loc=coefs_df.loc[coefs_df.types=="night_nodome","beta"].values[0],
                                   scale=coefs_df.loc[coefs_df.types=="night_nodome","se"].values[0],
                                   size=nboot)
coef_night_dome = np.random.normal(loc=coefs_df.loc[coefs_df.types=="night_dome","beta"].values[0],
                                   scale=coefs_df.loc[coefs_df.types=="night_dome","se"].values[0],
                                   size=nboot)

# read synthetic seasons
y1_seasons = 2015
y2_seasons = 2100
seasons = xr.open_dataset(loc_seasons+"synthetic_baseball_seasons_dates_daynight_domes_"+str(y1_seasons)+"-"+str(y2_seasons)+".nc")

#print(seasons.game_dome.sel(park="STP01").sel(boot=1).mean(dim="ngame"))
#sys.exit()

# read HadISD park temps for climatology
y1_hadisd = 1954
y2_hadisd = 2019
parks_final = sorted(np.unique(panel.loc[panel.year==2018,"park_id"].values))
park_temp_f = np.array([loc_hadisd+x for x in sorted(os.listdir(loc_hadisd)) if (str(y1_hadisd)+"-"+str(y2_hadisd)+".nc" in x)&(x[0:5] in parks_final)])
park_temp_obs = xr.open_mfdataset(park_temp_f,combine="nested",concat_dim="park").data_vars["tmax"].load()
park_temp_obs.coords["park"] = parks_final
park_temp_clm = park_temp_obs[:,(park_temp_obs.time.dt.year>=y1_clm)&(park_temp_obs.time.dt.year<=y2_clm)]
park_temp_clm.coords["dayofyear"] = xr.DataArray(park_temp_clm.time.dt.dayofyear,coords=[park_temp_clm.time],dims=["time"])
park_temp_clm_doy = park_temp_clm.groupby("dayofyear").mean(dim="time")


# dome
# adding the rangers stadium -- globe life field -- because the new one got a dome in 2020
# but the new one is just like across the street from the old one
# so it doesn't require us to change the location
dome_codes = ["ARL02","HOU02","HOU03","MIA02","MIL06","SEA02","STP01","TOR02","PHO01","SEA03","MIN03","MON02"] #sea03 - "safeco field"?
dome = xr.DataArray(np.zeros(len(parks_final)),coords=[parks_final],dims=["park"])
for j in np.arange(0,len(parks_final),1):
    if parks_final[j] in dome_codes:
        dome[j] = 1.0

dome_inverse = np.abs(dome - 1.0)
#print(dome.mean())

# GMST smoothing length
smooth_len = 30 # 30-year smoothing for GMST

# experiments
exps = ["ssp126","ssp245","ssp370","ssp585"] #

for ee in np.arange(2,len(exps),1):
    e = exps[ee]
    print(e,flush=True)

    # reading models
    # initial broad set of models
    model_tx_files = np.array([x for x in sorted(os.listdir(loc_cmip6_tx)) if ("historical_"+e in x)&(x.endswith(".nc"))&(str(y1_cmip6)+"-"+str(y2_cmip6) in x)])
    model_tx_prefix = np.array([x.split("_")[0]+"_"+x.split("_")[1] for x in model_tx_files])
    # GMST models
    model_gmst_files = np.array([x for x in sorted(os.listdir(loc_cmip6_gmst)) if ("historical-"+e in x)&(x.endswith(".nc"))&("gmst_annual_1850-2100.nc" in x)])
    model_gmst_prefix = np.array([x.split("_")[0]+"_"+x.split("_")[1] for x in model_gmst_files])

    # now the final intersection of those two
    models = reduce(np.intersect1d,(model_tx_prefix,model_gmst_prefix))
    #models_list = np.array([loc_cmip6_tx+x for x in model_files])
    #models = np.array([x.split("_")[0]+"_"+x.split("_")[1] for x in model_files])

    # inverse weights for realizations
    mdlnames = xr.DataArray(np.array([x.split("_")[0] for x in models]),coords=[models],dims=["model"])
    mdlwgts = xr.DataArray(np.array([1.0/sum(mdlnames==x) for x in mdlnames]),coords=[models],dims=["model"])
    mdl_p = mdlwgts/np.sum(mdlwgts)

    print("constructing monte carlo parameter samples",flush=True)
    ## now establish monte carlo/uncertainty distributions
    n_mc = 250
    mc_ind_mdl = np.zeros(n_mc) # climate model
    mc_ind_reg = np.zeros(n_mc) # regression coefficient
    mc_ind_seas = np.zeros(n_mc) # schedule
    uncertainty = np.arange(1,n_mc+1,1)

    for n in uncertainty:
        if (n==1)|(np.mod(n,50.0)==0):
            print(n)
        # climate models -- inversely weight models by number of realizations per model
        mc_ind_mdl[n-1] = int(np.random.choice(np.arange(0,len(models),1),size=1,p=mdl_p.values))
        # regression bootstrap ind
        mc_ind_reg[n-1] = int(np.random.choice(np.arange(0,nboot,1),size=1))
        # schedule
        mc_ind_seas[n-1] = int(np.random.choice(np.arange(0,len(seasons.boot.values),1),size=1))

    # now several scenarios
    # one where the day/night frequency is the same as historical
    # one where it's all night games
    # and one where its all day games
    for k in [0,1]: 
        if k == 0:
            dn_name = "daynight-dome-hist"
        if k == 1:
            dn_name = "daynight-dome-allnight"
        #if k == 2:
        #    dn_name = "daynight-dome-allnight-alldome"
        print(dn_name,flush=True)

        # now actually do the monte carlo samples
        print("looping through monte carlo samples and calculating home runs",flush=True)
        yrs = np.arange(y1_future,y2_future+1,1)
        hr_diff = xr.DataArray(np.full((len(parks_final),len(yrs),n_mc),np.nan),
                            coords=[parks_final,yrs,uncertainty],
                            dims=["park","year","uncertainty"])

        # add in GMST data for ease of analysis
        # smoothed
        gmst_anom_smth = xr.DataArray(np.full((len(yrs),n_mc),np.nan),
                                coords=[yrs,uncertainty],
                                dims=["year","uncertainty"])
        # not smoothed
        gmst_anom = xr.DataArray(np.full((len(yrs),n_mc),np.nan),
                                coords=[yrs,uncertainty],
                                dims=["year","uncertainty"])

        # get the models that we end up using
        #model_list_mc = xr.DataArray(np.full(n_mc,np.nan),
        #                            coords=[uncertainty],dims=["uncertainty"])
        model_list_mc = []
        for n in uncertainty:
            print(n,flush=True)
            # seasons

            season_dates_mc = seasons.game_dates[:,:,:,int(mc_ind_seas[n-1])].loc[:,:,yrs]
            season_daynight_mc = seasons.game_daynight[:,:,:,int(mc_ind_seas[n-1])].loc[:,:,yrs]
            season_dome_mc = seasons.game_dome[:,:,:,int(mc_ind_seas[n-1])].loc[:,:,yrs]

            # read model data
            model_list_mc.append(models[int(mc_ind_mdl[n-1])])
            mdl_tx_in = xr.open_dataset(loc_cmip6_tx+models[int(mc_ind_mdl[n-1])]+"_historical_"+e+"_baseball_park_tx_"+str(y1_cmip6)+"-"+str(y2_cmip6)+".nc").data_vars["tx"]
            mdl_tx_abs = mdl_tx_in.loc[parks_final,:] 
            mdl_tx_abs.coords["year"] = xr.DataArray(mdl_tx_abs.time.dt.year.values,
                                                coords=[mdl_tx_abs.time],dims=["time"])
            mdl_tx_abs.coords["dayofyear"] = xr.DataArray(mdl_tx_abs.time.dt.dayofyear.values,
                                                coords=[mdl_tx_abs.time],dims=["time"])
            # eliminate bias by calculating change relative to climatology period
            mdl_tx_clm = mdl_tx_abs.loc[:,str(y1_clm)+"-01-01":str(y2_clm)+"-12-31"].groupby("dayofyear").mean(dim="time")
            mdl_tx_delta = mdl_tx_abs.loc[:,str(y1_future)+"-01-01":str(y2_future)+"-12-31"].groupby("dayofyear") - mdl_tx_clm

            del([mdl_tx_abs,mdl_tx_clm])

            # read and add GMST
            mdl_gmst = xr.open_dataset(loc_cmip6_gmst+models[int(mc_ind_mdl[n-1])]+"_historical-"+e+"_gmst_annual_1850-2100.nc").data_vars["gmst_anom"]
            gmst_anom_smooth = mdl_gmst.rolling(time=smooth_len,min_periods=smooth_len,center=True).mean()
            gmst_anom_smth[:,n-1] = gmst_anom_smooth.loc[y1_future:y2_future].values
            gmst_anom[:,n-1] = mdl_gmst.loc[y1_future:y2_future].values
            del(mdl_gmst)

            # get coefficients
            coef_day_nodome_mc = coef_day_nodome[int(mc_ind_reg[n-1])]
            coef_day_dome_mc = coef_day_dome[int(mc_ind_reg[n-1])]
            coef_night_nodome_mc = coef_night_nodome[int(mc_ind_reg[n-1])]
            coef_night_dome_mc = coef_night_dome[int(mc_ind_reg[n-1])]

            # mdl_tx --> park (30) x time (29565)
            # season_dates --> park (30) x ngame (81) x year (81)

            # loop through years, maybe?
            # dome_inverse --> 30

            # check time elapsed
            import time
            start = time.time()

            for pp in np.arange(0,len(parks_final),1):
                #print(parks_final[pp],flush=True)
                #print(dome[pp].values)
                #   print(parks_final[pp],flush=True)
                for yy in np.arange(0,len(yrs),1):
                    #print(yrs[yy],flush=True)
                    #dome0 = dome[pp]
                    season_dates_y = season_dates_mc[pp,:,yy]
                    if dn_name == "daynight-dome-hist":
                        night = season_daynight_mc[pp,:,yy].values
                        dm = season_dome_mc[pp,:,yy].values
                    elif dn_name == "daynight-dome-allnight":
                        night = np.full(len(season_daynight_mc[pp,:,yy].values),1.0)
                        dm = season_dome_mc[pp,:,yy].values
                    elif dn_name == "daynight-dome-allnight-alldome":
                        night = np.full(len(season_daynight_mc[pp,:,yy].values),1.0)
                        if dome[pp].values==1:
                            dm = np.full(len(season_dome_mc[pp,:,yy].values),1.0)
                        else:
                            dm = season_dome_mc[pp,:,yy].values

                    #print(dome)
                    ## climatological dayofyear temperatures
                    tx_clm_dates_mean = park_temp_clm_doy[:,[x in season_dates_y for x in park_temp_clm_doy.dayofyear.values]].loc[parks_final[pp],:]

                    ## model temps
                    mdl_tx_y = mdl_tx_delta[pp,mdl_tx_delta.coords["year"]==yrs[yy]]
                    mdl_tx_y.coords["time"] = mdl_tx_y.dayofyear.values

                    ## add delta onto climatological temperatures
                    mdl_tx_dates = mdl_tx_y.loc[season_dates_y.values] + tx_clm_dates_mean.values
                    mdl_tx_final = mdl_tx_y.loc[season_dates_y.values].values

                    # daynight
                    day = np.abs(night - 1.0)
                    ndm = np.abs(dm - 1.0)

                    #calc
                    hr_diff_day_nodome = np.exp(mdl_tx_dates*coef_day_nodome_mc*day*ndm) - np.exp(tx_clm_dates_mean.values*coef_day_nodome_mc*day*ndm)
                    hr_diff_night_nodome = np.exp(mdl_tx_dates*coef_night_nodome_mc*night*ndm) - np.exp(tx_clm_dates_mean.values*coef_night_nodome_mc*night*ndm)
                    hr_diff_day_dome = np.exp(mdl_tx_dates*coef_day_dome_mc*day*dm) - np.exp(tx_clm_dates_mean.values*coef_day_dome_mc*day*dm)
                    hr_diff_night_dome = np.exp(mdl_tx_dates*coef_night_dome_mc*night*dm) - np.exp(tx_clm_dates_mean.values*coef_night_dome_mc*night*dm)

                    hr = np.sum(hr_diff_day_nodome.values)+np.sum(hr_diff_night_nodome.values)+np.sum(hr_diff_day_dome.values)+np.sum(hr_diff_night_dome.values)
                    hr_diff[pp,yy,n-1] = hr

            print(hr_diff.sel(uncertainty=n).sum(dim="park"))
            end = time.time()
            print((end-start)/60.,flush=True)
            #sys.exit()

        # add to dataset and write out
        model_list_out = xr.DataArray(model_list_mc,coords=[uncertainty],dims=["uncertainty"])
        hr_ds = xr.Dataset({"hr_difference":(["park","year","uncertainty"],hr_diff),
                            "gmst_anom":(["year","uncertainty"],gmst_anom),
                            "gmst_anom_smoothed":(["year","uncertainty"],gmst_anom_smth),
                            "model_list":(["uncertainty"],model_list_out)},
                                    coords={"park":(["park"],parks_final),
                                            "year":(["year"],yrs),
                                            "uncertainty":(["uncertainty"],uncertainty)})

        hr_ds.attrs["creation_date"] = str(datetime.datetime.now())
        hr_ds.attrs["created_by"] = "Christopher Callahan, Christopher.W.Callahan.GR@dartmouth.edu"
        hr_ds.attrs["variable_description"] = "Changing home runs due to global warming and "+str(smooth_len)+"-year smoothed GMST"
        hr_ds.attrs["created_from"] = os.getcwd()+"/CMIP6_HR_Projections.py"
        hr_ds.attrs["dims"] = "park, year, uncertainty"
        hr_ds.attrs["uncertainty"] = str(n_mc)+" monte carlo samples"

        fname_out = loc_out+"CMIP6_historical-"+e+"_homeruns_"+dn_name+"_"+str(y1_future)+"-"+str(y2_future)+".nc"
        hr_ds.to_netcdf(fname_out,mode="w")
        print(fname_out,flush=True)
