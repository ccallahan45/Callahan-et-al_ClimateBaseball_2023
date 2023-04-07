# Historical and future global mean surface temperature
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

# warnings
import warnings
warnings.filterwarnings("ignore",category=FutureWarning,message="'base' in")

# locations
loc_cmip6 = "/path/to/cmip6/data/"
loc_out = "../Data/CMIP6/GMST/"

# years
y1_clm = 1850
y2_clm = 1900

# years
y1_hist = 1850
y2_hist = 2014
y1_ssp = 2015
y2_ssp = 2100

# monthly to yearly
def monthly_to_yearly_mean(x):

        # calculate annual mean from monthly data
        # after weighting for the difference in month length
        # x must be data-array with time coord
        # xarray must be installed

        # x_yr = x.resample(time="YS").mean(dim="time") is wrong
        # because it doesn't weight for the # of days in each month

        days_in_mon = x.time.dt.days_in_month
        wgts = days_in_mon.groupby("time.year")/days_in_mon.groupby("time.year").sum()
        ones = xr.where(x.isnull(),0.0,1.0)
        x_sum = (x*wgts).resample(time="YS").sum(dim="time")
        ones_out = (ones*wgts).resample(time="YS").sum(dim="time")
        return(x_sum/ones_out)

# new grid
res = 2
lon_new = np.arange(1,359+res,res)
lat_new = np.arange(-89,89+res,res)

latshape = len(lat_new)
lonshape = len(lon_new)

# spatial weights
wgt = xr.DataArray(np.zeros((latshape,lonshape)),
                coords=[lat_new,lon_new],dims=["lat","lon"])
for ll in np.arange(0,lonshape,1):
    wgt[:,ll] = np.cos(np.radians(lat_new))

# experiments
exps = ["ssp126","ssp245","ssp370","ssp585"] #

# hist models
hist_models = np.array([x for x in sorted(os.listdir(loc_cmip6+"historical/tas_Amon/")) if (x.endswith(".nc"))])
hist_models_prefix = np.array([x.split("_")[2]+"_"+x.split("_")[4] for x in hist_models])

# loop through models and calc
for exp in exps:

    print(exp,flush=True)

    # ssp models
    ssp_models = np.array([x for x in sorted(os.listdir(loc_cmip6+exp+"/tas_Amon/")) if (x.endswith(".nc"))])
    ssp_models_prefix = np.array([x.split("_")[2]+"_"+x.split("_")[4] for x in ssp_models])

    # intersection
    models = reduce(np.intersect1d,(hist_models_prefix,ssp_models_prefix))
    print(models)

    models_exclude = []
    #models = np.array([x for x in models if x.split("_")[0] not in models_exclude])
    #existing_data = np.array([x.split("_")[0]+"_"+x.split("_")[1]+"_historical-"+exp for x in os.listdir(loc_out)])
    #models = np.array([x for x in models if x not in existing_data])

    # loop through models
    for m in models:
        print(m,flush=True)
        mname = m.split("_")[0]
        mreal = m.split("_")[1]

        if mname in models_exclude:
            continue

        model_hist = mname+"_historical_"+mreal
        model_ssp = mname+"_"+exp+"_"+mreal

        # read tas data
        print("reading hist data",flush=True)
        tas_ds_hist = xr.open_mfdataset(loc_cmip6+"historical/tas_Amon/"+"tas_Amon"+"_"+model_hist+"*.nc",concat_dim="time")
        tm_hist = tas_ds_hist.coords["time"].load()
        tm_hist_ind = (tm_hist.dt.year>=y1_hist)&(tm_hist.dt.year<=y2_hist)
        tas_hist = tas_ds_hist.tas[tm_hist_ind,:,:].load()
        if tas_hist.max()>200:
            tas_hist = tas_hist-273.15

        print("reading ssp data",flush=True)
        tas_ds_ssp = xr.open_mfdataset(loc_cmip6+exp+"/tas_Amon/"+"tas_Amon"+"_"+model_ssp+"*.nc",concat_dim="time")
        tm_ssp = tas_ds_ssp.coords["time"].load()
        tm_ssp_ind = (tm_ssp.dt.year>=y1_ssp)&(tm_ssp.dt.year<=y2_ssp)
        tas_ssp = tas_ds_ssp.tas[tm_ssp_ind,:,:].load()
        if tas_ssp.max()>200:
            tas_ssp = tas_ssp-273.15

        # concat
        tas = xr.concat([tas_hist,tas_ssp],dim="time")
        if (("latitude" in tas.coords)&("longitude" in tas.coords)):
            tas = tas.rename({"latitude":"lat","longitude":"lon"})
        del([tas_hist,tas_ssp])

        print("regridding and resampling to yearly",flush=True)
        # regrid
        if np.amin(tas.coords["lon"].values)<0:
            tas.coords["lon"] = tas.coords["lon"].values % 360
        tas_regrid = tas.interp(lat=lat_new,lon=lon_new)
        time_new = pd.date_range(start=str(y1_hist)+"-01-01",end=str(y2_ssp)+"-12-31",freq="MS")
        if len(tas_regrid.time.values)<len(time_new):
            continue
        tas_regrid.coords["time"] = pd.date_range(start=str(y1_hist)+"-01-01",end=str(y2_ssp)+"-12-31",freq="MS")
        del(tas)

        # yearly
        tas_regrid_yr = monthly_to_yearly_mean(tas_regrid)
        del(tas_regrid)

        # gmst
        print("calculating GMST and writing out",flush=True)
        gmst = tas_regrid_yr.weighted(wgt).mean(dim=["lat","lon"])
        gmst.coords["time"] = gmst.time.dt.year.values
        gmst_anom = gmst - gmst.loc[y1_clm:y2_clm].mean(dim="time")
        time_coord = gmst_anom.coords["time"]

        gmst_ds = xr.Dataset({"gmst":(["time"],gmst),
                              "gmst_anom":(["time"],gmst_anom)},
                              coords={"time":(["time"],time_coord)})

        gmst_ds.attrs["creation_date"] = str(datetime.datetime.now())
        gmst_ds.attrs["created_by"] = "Christopher Callahan, Christopher.W.Callahan.GR@dartmouth.edu"
        gmst_ds.attrs["variable_description"] = "Annual global mean surface temperature, and anomaly relative to "+str(y1_clm)+"-"+str(y2_clm)
        gmst_ds.attrs["created_from"] = os.getcwd()+"/CMIP6_GMST.py"
        gmst_ds.attrs["dims"] = "time"
        gmst_ds.attrs["grid"] = "regridded to 2x2 regular grid"
        gmst_ds.attrs["weights"] = "cos(lat)"

        fname_out = loc_out+m+"_historical-"+exp+"_gmst_annual_"+str(y1_hist)+"-"+str(y2_ssp)+".nc"
        gmst_ds.to_netcdf(fname_out,mode="w")
        print(fname_out,flush=True)
