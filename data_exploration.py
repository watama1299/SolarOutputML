## Import packages
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


## Read csv files
pv_combined = pd.read_csv('YMCA_combined.csv')
print(pv_combined.describe())


## Data analysis
## Using scatter plot and regression line to find correlation
## Regression line reference: https://towardsdatascience.com/seaborn-pairplot-enhance-your-data-understanding-with-a-single-plot-bf2f44524b22
p1 = sns.pairplot(pv_combined,
                  y_vars=['TempOut','OutHum','DewPt'], 
                  x_vars=['P_GEN_MIN','P_GEN_MAX'], 
                  kind='reg',
                  plot_kws={'line_kws':{'color':'red'}})

p2 = sns.pairplot(pv_combined,
                  y_vars=['WindSpeed','WindRun','WindChill'], 
                  x_vars=['P_GEN_MIN','P_GEN_MAX'],  
                  kind='reg',
                  plot_kws={'line_kws':{'color':'red'}})

p3 = sns.pairplot(pv_combined,
                  y_vars=['HeatIndex','THWIndex'], 
                  x_vars=['P_GEN_MIN','P_GEN_MAX'],  
                  kind='reg',
                  plot_kws={'line_kws':{'color':'red'}})

p4 = sns.pairplot(pv_combined,
                  y_vars=['Bar','Rain','RainRate'], 
                  x_vars=['P_GEN_MIN','P_GEN_MAX'],  
                  kind='reg',
                  plot_kws={'line_kws':{'color':'red'}})

p5 = sns.pairplot(pv_combined,
                  y_vars=['SolarRad','SolarEnergy','HiSolarRad'], 
                  x_vars=['P_GEN_MIN','P_GEN_MAX'],  
                  kind='reg',
                  plot_kws={'line_kws':{'color':'red'}})
plt.show()

## Using heat map to find correlation
h1 = sns.heatmap(pv_combined.corr(), annot=True)
plt.show()