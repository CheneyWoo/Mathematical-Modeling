from __future__ import print_function
import pandas as pd
import numpy as np
from scipy import  stats
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.graphics.api import qqplot
from numpy.random import randn

dta=[44.234 ,45.114 ,46.630 ,48.737 ,49.914 ,50.708 ,51.836 ,53.577 ,54.971 ,57.106 ,57.815 ,57.639 ,57.678 ,58.075 ,59.924 ,61.458 ,62.412 ,63.36259079,62.41079712,64.967 ,66.068 ,67.169 ];

dta=np.array(dta,dtype=np.float);

dta=pd.Series(dta);
dta.index = pd.Index(sm.tsa.datetools.dates_from_range('1996','2017'));
dta.plot(figsize=(12,8));
#plt.show();

fig = plt.figure(figsize=(12,8));
ax1= fig.add_subplot(111);
diff1 = dta.diff(1);
diff1.plot(ax=ax1);

#plt.show();

diff1= dta.diff(1);
fig = plt.figure(figsize=(12,8));
ax1=fig.add_subplot(211);
fig = sm.graphics.tsa.plot_acf(dta,lags=20,ax=ax1);
ax2 = fig.add_subplot(212);
fig = sm.graphics.tsa.plot_pacf(dta,lags=20,ax=ax2);
#plt.show();

try:
    arma_mod00 = sm.tsa.ARMA(dta,(0,0)).fit();
except ValueError,e:
    print ("Error");
else:
    print(arma_mod00.aic);

try:
    arma_mod01 = sm.tsa.ARMA(dta,(0,1)).fit();
except ValueError,e:
    print ("Error");
else:
    print(arma_mod01.aic);

try:
    arma_mod02 = sm.tsa.ARMA(dta,(0,2)).fit();
except ValueError,e:
    print ("Error");
else:
    print(arma_mod02.aic);

try:
    arma_mod10 = sm.tsa.ARMA(dta,(1,0)).fit();
except ValueError,e:
    print ("Error");
else:
    print(arma_mod10.aic);

try:
    arma_mod11 = sm.tsa.ARMA(dta,(1,1)).fit();
except ValueError,e:
    print ("Error");
else:
    print(arma_mod11.aic);

try:
    arma_mod12 = sm.tsa.ARMA(dta,(1,2)).fit();
except ValueError,e:
    print ("Error");
else:
    print(arma_mod12.aic);

try:
    arma_mod20 = sm.tsa.ARMA(dta,(2,0)).fit();
except ValueError,e:
    print ("Error");
else:
    print(arma_mod20.aic);

try:
    arma_mod21 = sm.tsa.ARMA(dta,(2,1)).fit();
except ValueError,e:
    print ("Error");
else:
    print(arma_mod21.aic);

try:
    arma_mod22 = sm.tsa.ARMA(dta,(2,2)).fit();
except ValueError,e:
    print ("Error");
else:
    print(arma_mod22.aic);


predict_dta = arma_mod10.predict('2017', '2020', dynamic=True);
print(predict_dta);

fig, ax = plt.subplots(figsize=(12, 8));
ax = dta.ix['1996':].plot(ax=ax);
fig = arma_mod10.plot_predict('2017', '2020', dynamic=True, ax=ax, plot_insample=False);
plt.show();
