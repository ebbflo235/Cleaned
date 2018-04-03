### read writeup for this code at https://www.ebbflow.net/blogger/post/1782978500575192718

import pandas as p
import numpy as np
import scipy.stats as s
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
os.getcwd()
os.chdir('/home/jonathan/Desktop/Auto_trade_scripts')

csv = "vixcurrent.csv"
vix = p.read_csv(csv,index_col = 0)
vix = vix[vix.VIX != 0]

vix = vix.iloc[:,0:2]
vix = vix.astype(float)

#1 day backward looking return
vix['SP500_ret'] = vix['SP500']/vix['SP500'].shift(1)

#252 day forward looking return
vix['252day_SP500_ret'] = (vix['SP500']/vix['SP500'].shift(252)).shift(-252)

#60 day backward looking return
vix['3Month_SP500_ret'] = (vix['SP500']/vix['SP500'].shift(60)).shift(-60)
vix['Delta_VIX'] = vix['VIX'] - vix['VIX'].shift(1)
vix['21day_Delta_VIX'] = vix['VIX'] - vix['VIX'].shift(21)

# find min and max of delta vix for rolling 50 days
# same idea as translate function below except with pandas operations
vix['rolling_max'] = vix['21day_Delta_VIX'].rolling(252).max()
vix['rolling_min'] = vix['21day_Delta_VIX'].rolling(252).min()
vix['spread'] = vix['rolling_max'] - vix['rolling_min']
vix['scaled'] = (vix['21day_Delta_VIX'] - vix['rolling_min']) / vix['spread']

# this can be changed (output of function)
vix['mapped'] = 1 + (vix['scaled']*49)

# top values
fifty = vix[vix['mapped']==50]
fifty_dropna = fifty[~np.isnan(fifty['252day_SP500_ret'])]
# geometric mean and sd, top values
top_geo = s.gmean(fifty_dropna['252day_SP500_ret'])
top_sd = np.std(fifty_dropna['252day_SP500_ret'])
top_SR = (top_geo-1) / top_sd

#all values
all_ret = vix[~np.isnan(vix['252day_SP500_ret'])]
all_geo = s.gmean(all_ret['252day_SP500_ret'])
all_sd = np.std(all_ret['252day_SP500_ret'])
all_SR = (all_geo-1) / all_sd

#temporarily changing axis
fifty_dropna.index = p.to_datetime(fifty_dropna.index).year 
all_ret.index = p.to_datetime(all_ret.index).year

cut_length = len(all_ret.index) - len(all_ret[~np.isnan(all_ret['mapped'])])
scatter = all_ret[cut_length:len(all_ret)]

######################################################################################
###GRAPH - RUN ALL AT ONCE ###########################################################
######################################################################################
red_patch = mpatches.Patch(color='red', label='All Returns')
blue_patch = mpatches.Patch(color='blue', label='Selected Trades')
plt.scatter(scatter.index,scatter['252day_SP500_ret']-1,color='r')
plt.scatter(fifty_dropna.index,fifty_dropna['252day_SP500_ret']-1,color='b')
plt.ylabel("Return")
plt.xlabel("Year")
plt.legend(handles=[red_patch,blue_patch])

#######################################################################################
###TRANSLATE FUNCTION##################################################################
#######################################################################################
def translate(value, leftMin, leftMax, rightMin, rightMax):
    # Figure out how 'wide' each range is
    leftSpan = leftMax - leftMin
    rightSpan = rightMax - rightMin

    # Convert the left range into a 0-1 range (float)
    valueScaled = float(value - leftMin) / float(leftSpan)

    # Convert the 0-1 range into a value in the right range.
    return rightMin + (valueScaled * rightSpan)

leftMin = np.min(vix['21day_Delta_VIX'])
leftMax = np.max(vix['21day_Delta_VIX'])

mapped =  translate(-.38,leftMin,leftMax,rightMin = 1,rightMax = 50)