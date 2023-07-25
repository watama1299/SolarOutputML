## Code reference: https://medium.com/@akashprabhakar427/solar-power-forecasting-using-machine-learning-and-deep-learning-61d6292693de

## Import packages
import pandas as pd


## Read csv files
in_file = pd.read_csv('FR_to_interpolate.csv')
# in_file.style


## Interpolate data
out_file = in_file.copy()
out_file['P_GEN_MIN'].interpolate(method='polynomial', order=2, inplace=True)
out_file['P_GEN_MAX'].interpolate(method='polynomial', order=2, inplace=True)
out_file['SolarRad'].interpolate(method='polynomial', order=2, inplace=True)
out_file.interpolate(method='linear', inplace=True)
# out_file.style


## Output interpolated data to new csv file
out_file.to_csv('FR_to_interpolate_done.csv')