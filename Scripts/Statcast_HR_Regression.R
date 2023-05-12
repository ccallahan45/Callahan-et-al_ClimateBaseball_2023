# Relationship between weather and home runs using statcast data
#### Christopher Callahan
#### Christopher.W.Callahan.GR@dartmouth.edu

rm(list=ls())
library(ggplot2)
library(margins)
library(sandwich)
library(stargazer)
library(lfe)
library(fixest)
library(tidyverse)
library(cowplot)
library(lemon)
library(gridExtra)
library(texreg)

#install.packages("xxxx",repo = 'https://mac.R-project.org')

# preparation
loc_panel <- "../Data/Panel/"
loc_save <- "../Data/RegressionResults/"

# read and save panel data
y1 <- 1954
y2 <- 2019
data <- read.csv(paste(loc_panel,"baseball_climate_data_",y1,"-",y2,".csv",sep=""))

y1_final <- 1962 # full adoption of 162-game season
data %>% filter(year>=y1_final) -> data

# generate t and t_squared
data$tx <- data$tmax_hadisd #data$tmax_era5
data$tx2 <- (data$tx)**2
data$game_id <- paste(data$park_id,data$year,data$dayofyear,sep="_")

# create new variables
data$total_hr <- data$visitor_hr + data$home_hr

# dome status -- fix NAs
# for all parks that just don't have them
# list dome status as zero
data[which(data$dome==0),"dome_status"] <- 0

# and list all dome statuses as 1 
# in the trop since it's a fixed dome
data[which(data$park_id=="STP01"),"dome_status"] <- 1

# how many obs do we have of dome status from parks w domes?
n_dome_park <- dim(data %>% filter(dome==1))[1]
n_dome_park_status <- dim(data %>% filter(dome==1,!is.na(dome_status)))[1]
print(100*n_dome_park_status/n_dome_park)
# 82%, not bad


# limit data
data_day_nodome <- data[(data$dome_status==0)&(data$daynight==0),]
data_day_dome <- data[(data$dome_status==1)&(data$daynight==0),]
data_night_nodome <- data[(data$dome_status==0)&(data$daynight==1),]
data_night_dome <- data[(data$dome_status==1)&(data$daynight==1),]
data_nodome <- data[(data$dome_status==0),]
data_dome <- data[(data$dome_status==1),]


### regressions

## poisson with fixed effects

total_mdl <- fepois(total_hr ~ tx | parknum + year + dayofyear,
                    data_nodome,cluster=~parknum+year)
print(summary(total_mdl))
r2(total_mdl,'pr2',full_names = TRUE)

total_mdl2 <- fepois(total_hr ~ tx | parknum + year + dayofyear,
                     data_nodome[(data_nodome$year>=2015)&(data_nodome$year<=2019),],
                     cluster=~parknum+year)
print(summary(total_mdl2))



#### statcast data!

#### make some plots for model choice


y1_sc <- 2015
y2_sc <- 2019
statcast_data <- read.csv(paste(loc_panel,"statcast_battedball_panel_",y1_sc,"-",y2_sc,".csv",sep=""))

statcast_data$tx <- statcast_data$tmax_hadisd #data$tmax_era5
statcast_data$tx2 <- (statcast_data$tx)**2
statcast_data$ws <- statcast_data$windspeed_hadisd
statcast_data[which(statcast_data$ws < 0),"ws"] <- NA

# dome status -- fix NAs
# for all parks that just don't have them
# list dome status as zero
statcast_data[which(statcast_data$dome==0),"dome_status"] <- 0

# and list all dome statuses as 1 
# in the trop since it's a fixed dome
statcast_data[which(statcast_data$park_id=="STP01"),"dome_status"] <- 1

# and other vars
statcast_data$hr_boolean <- as.numeric(statcast_data$events=="home_run")
statcast_data$launch_angle2 <- (statcast_data$launch_angle)**2
statcast_data$speed_norm <- statcast_data$launch_speed - mean(statcast_data$launch_speed,na.rm=T)
statcast_data$angle_norm <- statcast_data$launch_angle - mean(statcast_data$launch_angle,na.rm=T)
statcast_data$angle_norm2 <- (statcast_data$angle_norm**2)
statcast_data$game_id <- paste(statcast_data$park_id,statcast_data$year,statcast_data$dayofyear,sep="_")
statcast_data %>% rename(distance = hit_distance_sc) -> statcast_data


# limit data
statcast_data_nodome <- statcast_data[(statcast_data$dome_status==0),]
statcast_data_dome <- statcast_data[(statcast_data$dome_status==1),]

n_total <- dim(statcast_data %>% filter(!is.na(hr_boolean),!is.na(tx),!is.na(angle_norm),
                                        !is.na(speed_norm),!is.na(dayofyear),
                                        !is.na(year),!is.na(park_id),!is.na(dome_status)))[1]


# test by game
statcast_data_nodome %>% group_by(park_id,year,dayofyear,dome_status,game_id) %>%
  summarize(total_hr = sum(hr_boolean),
            mean_tx = mean(tx),
            n = n()) %>%
  filter(!is.na(year)) -> statcast_data_bygame

## how many fly balls in a game?
mean(statcast_data_bygame$n)

fepois(total_hr ~ mean_tx | park_id + year + dayofyear,statcast_data_bygame,
       cluster=~park_id+year)

# overall hr_boolean w/ logit
tx_hr_statcast <- feglm(hr_boolean ~ tx | park_id + year + dayofyear,
                        statcast_data_nodome,cluster=~game_id,
                        family=binomial(link="logit"))
print(tx_hr_statcast)
AIC(tx_hr_statcast)
print(100*(exp(coef(tx_hr_statcast)["tx"])-1))


# tx plus controls w/ logit
tx_hr_statcast_ctrl <- feglm(hr_boolean ~ tx + angle_norm + angle_norm2 + speed_norm | park_id + year + dayofyear,
                             statcast_data_nodome,family=binomial(link="logit"))
print(tx_hr_statcast_ctrl)
print(100*(exp(coef(tx_hr_statcast_ctrl)["tx"])-1))

# HR boolean w/ OLS 
tx_hr_statcast_ols <- feols(hr_boolean ~ tx | park_id + year + dayofyear,
                            statcast_data_nodome,cluster=~game_id)
print(tx_hr_statcast_ols)
print(100*coef(tx_hr_statcast_ols)["tx"]/mean(statcast_data_nodome$hr_boolean,na.rm=T))

# HR boolean plus controls w/ OLS
tx_hr_statcast_ols_ctrl <- feols(hr_boolean ~ tx + angle_norm + angle_norm2 + speed_norm | park_id + year + dayofyear,
                                 statcast_data_nodome,cluster=~game_id)
print(tx_hr_statcast_ols_ctrl)
print(100*coef(tx_hr_statcast_ols_ctrl)["tx"]/mean(statcast_data_nodome$hr_boolean,na.rm=T))
## with interaction, effect at average launch speed 

tx_hr_prob_ols <- data.frame(beta=numeric(0),se=numeric(0),
                             ci2_5=numeric(0),ci97_5=numeric(0))
tx_hr_prob_ols[1,"beta"] <- as.numeric(coef(tx_hr_statcast_ols_ctrl)["tx"])
tx_hr_prob_ols[1,"se"] <- as.numeric(se(tx_hr_statcast_ols_ctrl)["tx"])
tx_hr_prob_ols[1,"ci2_5"] <- confint(tx_hr_statcast_ols_ctrl)["tx",1]
tx_hr_prob_ols[1,"ci97_5"] <- confint(tx_hr_statcast_ols_ctrl)["tx",2]
write.csv(tx_hr_prob_ols,paste0(loc_save,"hr_probability_speed_angle_ols.csv"))


