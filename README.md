# Global warming, home runs, and the future of America's pastime

Data and replication code for "Global warming, home runs, and the future of America's pastime," published in the _Bulletin of the American Meteorological Society_ by Christopher Callahan, Nate Dominy, Jerry DeSilva, and Justin Mankin. The paper is available [here](https://journals.ametsoc.org/view/journals/bams/104/5/BAMS-D-22-0235.1.xml).

### Overview

The repository is organized into **Scripts/**, **Figures/**, and **Data/** folders.

- **Scripts/**: All code required to reproduce the findings of our work is included in this folder. 

- **Figures/**: The final figures from the main text are included in this folder.

- **Data/**: This folder includes intermediate and processed summary data that enable replication of most the figures and numbers cited in the text. Some of the files for the climate model projections are quite large, so they are not provided here. Should you desire any of this underlying data, feel free to contact me at _Christopher.W.Callahan.GR (at) dartmouth (dot) edu_ and I will be happy to organize a mass data transfer.


Finally, this repository includes game log and event file data from [Retrosheet](https://www.retrosheet.org/) as well as data on individual batted balls from [Statcast](https://baseballsavant.mlb.com/csv-docs). Please credit these organizations if you use any of this data for any reason. 

### Details

Each script performs a component of the analysis as follows:

- `Process_Park_HadISD_Data.py` assembles park-level time series of weather variables from the HadISD weather station data.
- `Process_Retrosheet_Temp.ipynb` extracts gametime temperature data from the Retrosheet event files.
- `Construct_Baseball_Panel.py` combines the baseball and weather data into a panel dataset for the regression analysis. `Process_Statcast_Distance_Data.ipynb` performs a similar function for the more recent Statcast batted-ball-level data.
- `Temp_HR_Regression.R` performs the core regression analysis and `Statcast_HR_Regression.R` performs the supplemental Statcast-based regression.
- `CMIP6_Historical_ParkTemp.py` and `CMIP6_Future_ParkTemp.py` construct past and future park-level time series of temperature from the CMIP6 climate models. `CMIP6_GMST.py` calculates from global mean temperature from the same models.
- `Generate_Future_Seasons.py` produces a set of randomly generated plausible baseball seasons for the future projections.
- `CMIP6_HR_Attribution.py` performs the attribution of home runs to historical global warming and `CMIP6_HR_Projections.py` performs the future projections of home runs over the 21st century.
- `Fig1.ipynb`, `Fig2.ipynb`, and `Fig3.ipynb` plot the main text figures. `Plot_TimePeriod_Fig.ipynb` and `Plot_RH_Density.ipynb` plot the supplementary figures. 

