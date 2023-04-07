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
loc_out = "../Data/HRAttribution/"

# warnings
import warnings
warnings.filterwarnings("ignore",category=FutureWarning,message="'base' in .resample()")


# years
y1_cmip6 = 1950
y2_cmip6 = 2019
y1_panel = 1954
y2_panel = 2019
y1_final = 1962
y2_final = 2019
y1_clm = 2000  # climatology/baseline
y2_clm = 2019
mon1 = 3
mon2 = 10

print('getting initial data',flush=True)

# read panel data
panel1 = pd.read_csv(loc_panel+"baseball_climate_data_"+str(y1_panel)+"-"+str(y2_panel)+".csv")
panel = panel1.loc[(panel1.year.values>=y1_final)&(panel1.year.values<=y2_final),:]
panel['total_hr'] = panel["visitor_hr"] + panel["home_hr"]
#print(panel)

# modify dome status
# where dome == 0 (where there's no dome on the park)
# we know dome status = 0
panel.loc[panel.dome==0,"dome_status"] = 0

# construct xarray dataarrays with dates and parks
print("assembling obs data",flush=True)
dates_full = pd.date_range(start=str(y1_final)+"-01-01",end=str(y2_final)+"-12-31",freq="D")
parks = sorted(np.unique(panel.loc[:,"park_id"].values))

# exclude some parks
parks_final = np.array([x for x in parks if x not in ["LON01","OMA01"]])

hr_obs = xr.DataArray(np.full((len(parks_final),len(dates_full)),np.nan),
                    coords=[parks_final,dates_full],dims=["park","time"])
tx_obs = xr.DataArray(np.full((len(parks_final),len(dates_full)),np.nan),
                    coords=[parks_final,dates_full],dims=["park","time"])
daynight_obs = xr.DataArray(np.full((len(parks_final),len(dates_full)),np.nan),
                    coords=[parks_final,dates_full],dims=["park","time"])
dome_obs = xr.DataArray(np.full((len(parks_final),len(dates_full)),np.nan),
                    coords=[parks_final,dates_full],dims=["park","time"])

#print(hr_obs)
# loop through parks
for p in parks_final:
    print(p)
    panel_p = panel.loc[panel.park_id==p,:]
    dates = np.array([pd.to_datetime(str(i)+"-"+str(j)+"-"+str(k)) for i, j, k in zip(panel_p.year.values,panel_p.month.values,panel_p.day.values)])
    hr_obs.loc[p,dates] = panel_p.total_hr.values
    tx_obs.loc[p,dates] = panel_p.tmax_hadisd.values
    daynight_obs.loc[p,dates] = panel_p.daynight.values
    dome_obs.loc[p,dates] = panel_p.dome_status.values

hr_obs_mons = hr_obs[:,(hr_obs.time.dt.month>=mon1)&(hr_obs.time.dt.month<=mon2)]
tx_obs_mons = tx_obs[:,(tx_obs.time.dt.month>=mon1)&(tx_obs.time.dt.month<=mon2)]
daynight_obs_mons = daynight_obs[:,(daynight_obs.time.dt.month>=mon1)&(daynight_obs.time.dt.month<=mon2)]
dome_obs_mons = dome_obs[:,(dome_obs.time.dt.month>=mon1)&(dome_obs.time.dt.month<=mon2)]
# set nan to 1 in dome_obs -- assume dome closed
dome_obs_mons = dome_obs_mons.where(~np.isnan(dome_obs_mons),1.0)

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

# dome
dome_codes = ["HOU02","HOU03","MIA02","MIL06","SEA02","STP01","TOR02","PHO01","SEA03","MIN03","MON02"] #sea03 - "safeco field"?
dome = xr.DataArray(np.zeros(len(parks_final)),coords=[parks_final],dims=["park"])
for j in np.arange(0,len(parks_final),1):
    if parks_final[j] in dome_codes:
        dome[j] = 1.0
dome_inverse = np.abs(dome - 1.0)

# domes for specific games
dm = dome_obs_mons*1.0
ndm = np.abs(dm - 1.0)
#print(dome.mean())


## read climate model data
model_files = np.array([x for x in sorted(os.listdir(loc_cmip6_tx)) if ("historical_natural" in x)&(x.endswith(".nc"))&(str(y1_cmip6)+"-"+str(y2_cmip6) in x)])
models = np.array([x.split("_")[0]+"_"+x.split("_")[1] for x in model_files])
model_files_in = np.array([loc_cmip6_tx+x for x in model_files])

# inverse weights for realizations
mdlnames = xr.DataArray(np.array([x.split("_")[0] for x in models]),coords=[models],dims=["model"])
mdlwgts = xr.DataArray(np.array([1.0/sum(mdlnames==x) for x in mdlnames]),coords=[models],dims=["model"])
mdl_p = mdlwgts/np.sum(mdlwgts)


print("constructing monte carlo parameter samples",flush=True)
## now establish monte carlo/uncertainty distributions
n_mc = 1000 # for consistency with the 250*4 MC distributions
mc_ind_mdl = np.zeros(n_mc) # climate model
mc_ind_reg = np.zeros(n_mc) # regression coefficient
uncertainty = np.arange(1,n_mc+1,1)

for n in uncertainty:
    if (n==1)|(np.mod(n,100.0)==0):
        print(n)
    # climate models -- inversely weight models by number of realizations per model
    mc_ind_mdl[n-1] = int(np.random.choice(np.arange(0,len(models),1),size=1,p=mdl_p.values))
    # regression bootstrap ind
    mc_ind_reg[n-1] = int(np.random.choice(np.arange(0,nboot,1),size=1))


print("iterating through uncertainty and calculating attribution",flush=True)
yrs = np.arange(y1_final,y2_final+1,1)

hr_diff = xr.DataArray(np.full((len(parks_final),len(yrs),n_mc),np.nan),
                    coords=[parks_final,yrs,uncertainty],
                    dims=["park","year","uncertainty"])

for n in uncertainty:
    print(n,flush=True)

    import time
    start = time.time()

    # get coefficients
    coef_day_nodome_mc = coef_day_nodome[int(mc_ind_reg[n-1])]
    coef_day_dome_mc = coef_day_dome[int(mc_ind_reg[n-1])]
    coef_night_nodome_mc = coef_night_nodome[int(mc_ind_reg[n-1])]
    coef_night_dome_mc = coef_night_dome[int(mc_ind_reg[n-1])]

    # read model data
    mdl_tx_ds = xr.open_dataset(loc_cmip6_tx+models[int(mc_ind_mdl[n-1])]+"_historical_natural_baseball_park_tx_"+str(y1_cmip6)+"-"+str(y2_cmip6)+".nc")
    mdl_tx_hist = mdl_tx_ds.data_vars["tx_hist"].loc[parks_final,str(y1_final)+"-01-01":str(y2_final)+"-12-31"]
    mdl_tx_nat = mdl_tx_ds.data_vars["tx_histnat"].loc[parks_final,str(y1_final)+"-01-01":str(y2_final)+"-12-31"]

    # model difference
    mdl_diff = mdl_tx_hist - mdl_tx_nat
    del([mdl_tx_hist,mdl_tx_nat])

    # slice by months
    mdl_diff_mons = mdl_diff[:,(mdl_diff.time.dt.month>=mon1)&(mdl_diff.time.dt.month<=mon2)]
    #del([mdl_diff,tx_obs,hr_obs,daynight_obs])

    # calculate counterfactual temp
    mdl_diff_mons.coords["time"] = tx_obs_mons.coords["time"]
    tx_cf = tx_obs_mons - mdl_diff_mons

    del(mdl_diff_mons)

    #print(tx_obs_mons)
    #print(tx_cf)
    #print(hr_obs_mons)
    #print(daynight_obs_mons)
    # dome --> 0 for no dome, 1 for dome
    # dome_inverse --> 0 for dome, 1 for no dome
    #print(dome)
    #print(dome_inverse)

    daynight_inverse = np.abs(daynight_obs_mons - 1.0) # 1 for day, 0 for night

    hr_diff_day_nodome = np.exp(tx_obs_mons*daynight_inverse*ndm*coef_day_nodome_mc) - np.exp(tx_cf*daynight_inverse*ndm*coef_day_nodome_mc)
    hr_diff_day_nodome_yr = hr_diff_day_nodome.resample(time="YS").sum(dim="time")

    hr_diff_night_nodome = np.exp(tx_obs_mons*daynight_obs_mons*ndm*coef_night_nodome_mc) - np.exp(tx_cf*daynight_obs_mons*ndm*coef_night_nodome_mc)
    hr_diff_night_nodome_yr = hr_diff_night_nodome.resample(time="YS").sum(dim="time")

    hr_diff_day_dome = np.exp(tx_obs_mons*daynight_inverse*dm*coef_day_dome_mc) - np.exp(tx_cf*daynight_inverse*dm*coef_day_dome_mc)
    hr_diff_day_dome_yr = hr_diff_day_dome.resample(time="YS").sum(dim="time")

    hr_diff_night_dome = np.exp(tx_obs_mons*daynight_obs_mons*dm*coef_night_dome_mc) - np.exp(tx_cf*daynight_obs_mons*dm*coef_night_dome_mc)
    hr_diff_night_dome_yr = hr_diff_night_dome.resample(time="YS").sum(dim="time")

    hr_diff_total = hr_diff_day_nodome_yr+hr_diff_night_nodome_yr+hr_diff_day_dome_yr+hr_diff_night_dome_yr

    hr_diff[:,:,n-1] = hr_diff_total.values

    end = time.time()
    #print((end-start)/60.,flush=True)
    #print(hr_diff[:,:,n-1].sum(dim="park"))

print(hr_diff.sum(dim="park").mean(dim="uncertainty"))
print(hr_diff.sum(dim="park").std(dim="uncertainty"))

# add to dataset and write out
hr_ds = xr.Dataset({"hr_difference":(["park","year","uncertainty"],hr_diff)},
                            coords={"park":(["park"],parks_final),
                                    "year":(["year"],yrs),
                                    "uncertainty":(["uncertainty"],uncertainty)})

hr_ds.attrs["creation_date"] = str(datetime.datetime.now())
hr_ds.attrs["created_by"] = "Christopher Callahan, Christopher.W.Callahan.GR@dartmouth.edu"
hr_ds.attrs["variable_description"] = "Changing home runs due to historical global warming"
hr_ds.attrs["created_from"] = os.getcwd()+"/CMIP6_HR_Attribution.py"
hr_ds.attrs["dims"] = "park, year, uncertainty"
hr_ds.attrs["uncertainty"] = str(n_mc)+" monte carlo samples"

fname_out = loc_out+"CMIP6_historical-natural_homeruns_"+str(y1_final)+"-"+str(y2_final)+".nc"
hr_ds.to_netcdf(fname_out,mode="w")
print(fname_out,flush=True)
