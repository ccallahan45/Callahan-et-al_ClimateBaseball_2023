# Assembling CMIP6 historical/nat ballpark tx data
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
loc_in = "../Data/CMIP6/BallparkTx/"
loc_out = "../Data/Panel/"

# years
y1 = 1950
y2 = 2019

# get list of models
#mdl_files = [x for x in os.listdir(loc_in) if x.endswith(".nc")]
mdls = [x.split("_")[0]+"_"+x.split("_")[1] for x in os.listdir(loc_in) if (x.endswith(".nc"))&("historical_natural" in x)&(str(y1)+"-"+str(y2) in x)]

# read in all the data along a new model dimension
files_in = sorted([loc_in+x for x in os.listdir(loc_in) if x.endswith(".nc")&("historical_natural" in x)])
tx_ds = xr.open_mfdataset(files_in,combine="nested",concat_dim="model")
tx_ds.coords["model"] = mdls

# extract data variables
tx_hist = tx_ds.tx_hist.load()
tx_histnat = tx_ds.tx_histnat.load()
print(tx_hist)

# create dataframe
mon1 = 3
mon2 = 10
ind = (tx_hist.time.dt.month>=mon1)&(tx_hist.time.dt.month<=mon2)
tx_hist_stack = tx_hist[:,:,ind].stack(index=("model","park","time"))
tx_df = tx_hist_stack.to_dataframe().reset_index()

# add hist nat
tx_histnat_stack = tx_histnat[:,:,ind].stack(index=("model","park","time"))
tx_df["tx_histnat"] = tx_histnat_stack.values
tx_df["tx_diff"] = tx_df.tx_hist.values - tx_df.tx_histnat.values

# add year month day as separate cols
tx_df["year"] = tx_hist_stack.time.dt.year.values
tx_df["month"] = tx_hist_stack.time.dt.month.values
tx_df["day"] = tx_hist_stack.time.dt.day.values

# reorder cols
tx_df = tx_df[["model","park","time","year","month","day","tx_hist","tx_histnat","tx_diff"]]

# monthly to smooth out daily-scale variability
tx_df_monthly = tx_df.groupby(["model","park","year","month"]).mean().reset_index().drop(columns="day")
print(tx_df_monthly)

# write out
fname = loc_out+"CMIP6_historical_natural_ballpark_tx_monthly_"+str(y1)+"-"+str(y2)+".csv"
tx_df_monthly.to_csv(fname)
print(fname)
