# Relationship between weather and home runs
#### Christopher Callahan
#### Christopher.W.Callahan.GR@dartmouth.edu

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
data$tn <- data$tmin_hadisd
data$tn2 <- (data$tn)**2
data$tx_rs <- data$temp_retrosheet
data$tx2_rs <- (data$tx_rs)**2
data$p <- data$precip_hadisd
data$tw <- data$twmax_hadisd
data$tw2 <- (data$tw)**2

# limit wind fillvalue
data$ws <- data$windspeed_hadisd
data[which(data$ws < 0),"ws"] <- NA

# calculate density
pv <- data$vapor_pressure_hadisd*100.0 # hPa to Pa
pd <- data$slp_hadisd*100.0 - pv
rd <- 287.058
rv <- 461.495
tk <- data$tmean_hadisd + 273.15
data$density <- (pd/(rd*tk)) + (pv/(rv*tk))
data$density_norm <- (data$density - mean(data$density,na.rm=T))/sd(data$density,na.rm=T)
data$tx_norm <- (data$tx - mean(data$tx,na.rm=T))/sd(data$tx,na.rm=T)
data$tmean_norm <- (data$tmean_hadisd - mean(data$tmean_hadisd,na.rm=T))/sd(data$tmean_hadisd,na.rm=T)

# create new variables
data$total_hr <- data$visitor_hr + data$home_hr

# for each game, what is the average home temperature for the H and V teams?
visitingteams <- unique(data$visitor_year) #unique(data$visitor)
data$visitor_avg_t <- numeric(dim(data)[1])
for (vv in c(1:length(visitingteams))){
  v <- visitingteams[vv]
  data %>% filter(home_year==v | visitor_year==v) %>% summarize(meant=mean(tx)) -> avg_t
  #print(as.vector(unlist(avg_t)))
  data[data$visitor_year==v,"visitor_avg_t"] <- as.vector(unlist(avg_t))
}


data %>% group_by(home) %>%
  mutate(home_avg_t=mean(tx)) -> data
data <- data.frame(data)
data$visitor_avg_t_cen <- (data$visitor_avg_t) - mean(data$visitor_avg_t,na.rm=T)
data$home_avg_t_cen <- (data$home_avg_t) - mean(data$home_avg_t,na.rm=T)

# dome status -- fix NAs
# for all parks that just don't have them
# list dome status as zero
data[which(data$dome==0),"dome_status"] <- 0

# and list all dome statuses as 1 
# in the trop since it's a fixed dome
data[which(data$park_id=="STP01"),"dome_status"] <- 1

# how many obs do we have of dome status from parks w domes?
n_dome_park <- dim(data %>% filter(!is.na(total_hr),!is.na(tx),!is.na(parknum),!is.na(year),!is.na(dayofyear),dome==1))[1]
n_dome_park_status <- dim(data %>% 
                            filter(!is.na(total_hr),!is.na(tx),!is.na(parknum),!is.na(year),!is.na(dayofyear),dome==1,!is.na(dome_status)))[1]
print(100*n_dome_park_status/n_dome_park)
# 82%, not bad

n_dome_park1 <- dim(data %>% filter(!is.na(total_hr),!is.na(tx),!is.na(parknum)))[1]
n_dome_park_status1 <- dim(data %>% 
                            filter(!is.na(total_hr),!is.na(tx),!is.na(parknum),!is.na(dome_status)))[1]
print(100*n_dome_park_status1/n_dome_park1)


# limit data
data_day_nodome <- data[(data$dome_status==0)&(data$daynight==0),]
data_day_dome <- data[(data$dome_status==1)&(data$daynight==0),]
data_night_nodome <- data[(data$dome_status==0)&(data$daynight==1),]
data_night_dome <- data[(data$dome_status==1)&(data$daynight==1),]
data_nodome <- data[(data$dome_status==0),]
data_dome <- data[(data$dome_status==1),]

#### main poisson regression
#### coefficients for all different classes of games

game_types <- c("all","all_nodome","all_dome","day_nodome",
                "night_nodome","day_dome","night_dome")
df_types <- data.frame(types=game_types,
                       beta=numeric(length(game_types)),
                       se=numeric(length(game_types)),
                       ci2_5=numeric(length(game_types)),
                       ci97_5=numeric(length(game_types)),
                       n=numeric(length(game_types)))
for (j in c(1:length(game_types))){
  print(j)
  t <- game_types[j]
  if (t=="all"){
    dat <- data
  } else if (t=="all_nodome"){
    dat <- data_nodome
  } else if (t=="all_dome"){
    dat <- data_dome
  } else if (t=="day_nodome"){
    dat <- data_day_nodome
  } else if (t=="night_nodome"){
    dat <- data_night_nodome
  } else if (t=="day_dome") {
    dat <- data_day_dome
  } else if (t=="night_dome") { 
    dat <- data_night_dome
  }
  mdl <- fepois(total_hr ~ tx | parknum + year + dayofyear,
                dat,cluster=~parknum+year)
  n <- nobs(mdl)
  df_types[j,"beta"] <- as.numeric(coef(mdl)["tx"])
  df_types[j,"se"] <- as.numeric(se(mdl)["tx"])
  df_types[j,"ci2_5"] <- confint(mdl)["tx",1]
  df_types[j,"ci97_5"] <- confint(mdl)["tx",2]
  df_types[j,"n"] <- n
}

write.csv(df_types,paste0(loc_save,"homeruns_tx_poisson_gametypes.csv"))



#### placebo test for each game type
nboot <- 1000
game_types <- c("all","all_nodome","all_dome","day_nodome",
                "night_nodome","day_dome","night_dome")

df_types_placebo <- data.frame(types=game_types,
                       beta=numeric(length(game_types)),
                       se=numeric(length(game_types)),
                       ci2_5=numeric(length(game_types)),
                       ci97_5=numeric(length(game_types)))

coefs <- matrix(data=NA,nrow=length(game_types),ncol=nboot)

for (n in c(1:nboot)){
  print(n)
  data %>% group_by(parknum) %>%
    mutate(tx_rand=sample(tx)) -> dat
  for (j in c(1:length(game_types))){
    t <- game_types[j]
    if (t=="all"){
      df <- dat
    } else if (t=="all_nodome"){
      df <- dat[dat$dome_status==0,]
    } else if (t=="all_dome"){
      df <- dat[dat$dome_status==1,]
    } else if (t=="day_nodome"){
      df <- dat[(dat$dome_status==0)&(dat$daynight==0),]
    } else if (t=="night_nodome"){
      df <- dat[(dat$dome_status==0)&(dat$daynight==1),]
    } else if (t=="day_dome") {
      df <- dat[(dat$dome_status==1)&(dat$daynight==0),]
    } else if (t=="night_dome") { 
      df <- dat[(dat$dome_status==1)&(dat$daynight==1),]
    }
    
    mdl <- fepois(total_hr ~ tx_rand | parknum + year + dayofyear,df,notes=FALSE)
    coefs[j,n] <- as.numeric(coef(mdl)["tx_rand"])
  }
}

for (j in c(1:length(game_types))){
  df_types_placebo[j,"beta"] <- mean(coefs[j,])
  df_types_placebo[j,"se"] <- sd(coefs[j,])
  df_types_placebo[j,"ci2_5"] <- as.numeric(quantile(coefs[j,],0.025))
  df_types_placebo[j,"ci97_5"] <- as.numeric(quantile(coefs[j,],0.975))
}

write.csv(df_types_placebo,paste0(loc_save,"homeruns_tx_poisson_gametypes_placebo_parkgroup.csv"))




#### now coefficients for different time periods

time_periods <- c(1962,1982,2000,2015)
dn <- c("both","day","night")
time_period_str <- vector(mode='character',length=length(time_periods))
df_time <- data.frame(y1=rep(time_periods,each=length(dn)),
                      daynight=rep(dn,length(time_periods)),
                       beta=numeric(length(rep(time_periods,each=length(dn)))),
                       se=numeric(length(rep(time_periods,each=length(dn)))),
                       ci2_5=numeric(length(rep(time_periods,each=length(dn)))),
                       ci97_5=numeric(length(rep(time_periods,each=length(dn)))))
for (j in c(1:length(time_periods))){
  print(j)
  y_1 <- time_periods[j]
  if (y_1<1995){y_2 <- time_periods[j+1]-1}else{y_2 <- (y2+1)}
  time_period_str[j] <- paste0(y_1,"-",y_2)
  for (k in c(1:length(dn))){
    data_nodome %>% filter(year>=y_1,year<y_2) -> dat_y
    if (dn[k]=="both"){dat<-dat_y}else(dat<-dat_y[dat_y$daynight==(k-2),])
    mdl <- fepois(total_hr ~ tx | parknum + year + dayofyear,
                  dat,cluster=~parknum+year)
    ind <- (df_time$y1==y_1)&(df_time$daynight==dn[k])
    df_time[ind,"beta"] <- as.numeric(coef(mdl)["tx"])
    df_time[ind,"se"] <- as.numeric(se(mdl)["tx"])
    df_time[ind,"ci2_5"] <- confint(mdl)["tx",1]
    df_time[ind,"ci97_5"] <- confint(mdl)["tx",2]
  }
}

write.csv(df_time,paste0(loc_save,"homeruns_tx_poisson_time_periods.csv"))


#### Coefficients for each year
years <- unique(data$year)
years_df <- data.frame(year=years,
                      beta=numeric(length(years)),
                      se=numeric(length(years)),
                      ci2_5=numeric(length(years)),
                      ci97_5=numeric(length(years)))

for (y in years){
  mdl <- fepois(total_hr ~ tx | parknum + year + dayofyear,
                data_nodome[data_nodome$year==y,],
                cluster=~park_id,notes=FALSE)
  print(as.numeric(coef(mdl)["tx"]))
  print(confint(mdl)["tx",1])
  years_df[years_df$year==y,"beta"] <- as.numeric(coef(mdl)["tx"])
  years_df[years_df$year==y,"se"] <- as.numeric(se(mdl)["tx"])
  years_df[years_df$year==y,"ci2_5"] <- confint(mdl)["tx",1]
  years_df[years_df$year==y,"ci97_5"] <- confint(mdl)["tx",2]
  years_df[years_df$year==y,"n"] <- nobs(mdl)
}


write.csv(years_df,paste0(loc_save,"homeruns_tx_poisson_years.csv"))


#### Coefficients for each park

park_id_uq <- sort(unique(data[data$year==2018,"park_id"]))
park_df <- data.frame(park=park_id_uq,
                      dome=numeric(length(park_id_uq)),
                      beta=numeric(length(park_id_uq)),
                      se=numeric(length(park_id_uq)),
                      ci2_5=numeric(length(park_id_uq)),
                      ci97_5=numeric(length(park_id_uq)))

for (p in park_id_uq){
  mdl <- fepois(total_hr ~ tx | parknum + year + dayofyear,
                data[data$park_id==p,],
                cluster=~year,notes=FALSE)
  dm <- mean(data[data$park_id==p,'dome'])
  park_df[park_df$park==p,"dome"] <- dm
  park_df[park_df$park==p,"beta"] <- as.numeric(coef(mdl)["tx"])
  park_df[park_df$park==p,"se"] <- as.numeric(se(mdl)["tx"])
  park_df[park_df$park==p,"ci2_5"] <- confint(mdl)["tx",1]
  park_df[park_df$park==p,"ci97_5"] <- confint(mdl)["tx",2]
  park_df[park_df$park==p,"n"] <- nobs(mdl)
}

# the ones that have negative relationships all have domes! 
write.csv(park_df,paste0(loc_save,"homeruns_tx_poisson_parks.csv"))



#### Poisson vs. negative binomial

## model 1: poisson
mdl1 <- fepois(total_hr ~ tx | parknum + year + dayofyear,data_nodome,
               cluster=~parknum+year)

## model 2: neg bin
mdl2 <- fenegbin(total_hr ~ tx | parknum + year + dayofyear,data_nodome,
               cluster=~parknum+year)

texreg(list(mdl1,mdl2),digits=4,stars=c(0.001,0.01,0.05))



#### Dome/no dome, day/night


## model 1: no dome, day
mdl1 <- fepois(total_hr ~ tx | parknum + year + dayofyear,data_day_nodome,
               cluster=~parknum+year)

## model 2: no dome, night
mdl2 <- fepois(total_hr ~ tx | parknum + year + dayofyear,data_night_nodome,
               cluster=~parknum+year)

## model 3: dome, day
mdl3 <- fepois(total_hr ~ tx | parknum + year + dayofyear,data_day_dome,
               cluster=~parknum+year)

## model 4: dome, night
mdl4 <- fepois(total_hr ~ tx | parknum + year + dayofyear,data_night_dome,
               cluster=~parknum+year)

texreg(list(mdl1,mdl2,mdl3,mdl4),digits=4,stars=c(0.001,0.01,0.05))



#### density vs. temperature


## model 1: temp
mdl1 <- fepois(total_hr ~ tmean_norm | parknum + year + dayofyear,data_nodome,
               cluster=~parknum+year)

## model 2: density
mdl2 <- fepois(total_hr ~ density_norm | parknum + year + dayofyear,data_nodome,
                 cluster=~parknum+year)

## model 3: both
mdl3 <- fepois(total_hr ~ tmean_norm + density_norm | parknum + year + dayofyear,data_nodome,
               cluster=~parknum+year)

texreg(list(mdl1,mdl2,mdl3),digits=4,stars=c(0.001,0.01,0.05))




#### Visitor, home, and other temperature datasets    

## model 1: hadisd (main)
mdl1 <- fepois(total_hr ~ tx | parknum + year + dayofyear,data_nodome,
               cluster=~parknum+year)

## model 2: visitor
mdl2 <- fepois(visitor_hr ~ tx | parknum + year + dayofyear,data_nodome,
               cluster=~parknum+year)

## model 3: home
mdl3 <- fepois(home_hr ~ tx | parknum + year + dayofyear,data_nodome,
               cluster=~parknum+year)

## model 4: era5
mdl4 <- fepois(total_hr ~ tmax_era5 | parknum + year + dayofyear,data_nodome,
               cluster=~parknum+year)

## model 5: retrosheet
mdl5 <- fepois(total_hr ~ tx_rs | parknum + year + dayofyear,data_nodome,
               cluster=~parknum+year)

texreg(list(mdl1,mdl2,mdl3,mdl4,mdl5),digits=4,stars=c(0.001,0.01,0.05))




#### other weather vars

## model 1: main
mdl1 <- fepois(total_hr ~ tx | parknum + year + dayofyear,data_nodome,
               cluster=~parknum+year)

## model 2: precip
mdl2 <- fepois(total_hr ~ tx + precip_gpcp | parknum + year + dayofyear,data_nodome,
               cluster=~parknum+year)

## model 3: humidity
mdl3 <- fepois(total_hr ~ tx + rh_hadisd | parknum + year + dayofyear,data_nodome,
               cluster=~parknum+year)

## model 4: wind
mdl4 <- fepois(total_hr ~ tx + ws | parknum + year + dayofyear,data_nodome,
               cluster=~parknum+year)

texreg(list(mdl1,mdl2,mdl3,mdl4),digits=4,stars=c(0.001,0.01,0.05))


