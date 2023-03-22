# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 12:46:27 2020

@author: Ejdrup
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv      
from scipy import stats
import seaborn as sns
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

def moving_average(a, n) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

    

lira_col = 'olive'
nic_col = 'teal'
both_col = 'indigo'
#%% Load

Liraglutide = np.genfromtxt("Liraglutide.csv",delimiter=",")
Both = np.genfromtxt("Both.csv",delimiter=",")
Nicotine = np.genfromtxt("Nicotine.csv",delimiter=",")

#%% Traces
n = 2*1200

smth_liraglutide = np.zeros((len(Liraglutide[46000:-4620,:])-n+1,8))
smth_nicotine = np.zeros((len(Liraglutide[46000:-4620,:])-n+1,8))
smth_both = np.zeros((len(Liraglutide[46000:-4620,:])-n+1,7))
for i in range(1,9):
    smth_liraglutide[:,i-1] = moving_average(Liraglutide[46000:-4620,i],n)
    smth_nicotine[:,i-1] = moving_average(Nicotine[46000:-4620,i],n)
    if i == 8:
        continue
    smth_both[:,i-1] = moving_average(Both[46000:-4620,i],n)
    
#%% Plot
    
    
Liraglutide_mean = np.mean(smth_liraglutide,axis=1)
Liraglutide_SEM = stats.sem(smth_liraglutide,axis=1)
Nicotine_mean = np.mean(smth_nicotine,axis=1)
Nicotine_SEM = stats.sem(smth_nicotine,axis=1)
Both_mean = np.mean(smth_both,axis=1)
Both_SEM = stats.sem(smth_both,axis=1)

time_series = np.linspace(0,len(Liraglutide_mean)/20/60,len(Liraglutide_mean))-45.0+(n/20/60)

fig, ax1 = plt.subplots(figsize=(4, 3), dpi=200) # subplots rather than plt will be used to fine-tune the output
ax1.plot(1,1, lw = 1.5, color = lira_col , label = 'Vehicle') 
ax1.plot(1,1, lw = 1.5, color = nic_col, label = 'Vehicle') 
ax1.plot(1,1, lw = 1.5, color = both_col, label = 'Vehicle') 
ax1.plot([-30,-30],[-4,7.4], color = "dimgrey", LineStyle = "--")
ax1.plot([0,0],[-4,7.4], color = "dimgrey", LineStyle = "--")
plt.annotate("Saline", (-30,7.4), textcoords="offset points", xytext=(0,2), ha='center', color = 'dimgrey',fontweight='bold',fontsize=8)
plt.annotate("Drug", (0,7.4), textcoords="offset points", xytext=(0,2), ha='center', color = 'dimgrey',fontweight='bold',fontsize=8)


ax1.plot(time_series, Liraglutide_mean, LineStyle = '-', lw = 1.5, color = lira_col , label = '_nolegend_') 
ax1.fill_between(time_series, Liraglutide_mean-Liraglutide_SEM, 
                 Liraglutide_mean+Liraglutide_SEM, color = lira_col , alpha=0.3, lw = 0.1, label = '_nolegend_')


ax1.plot(time_series, Nicotine_mean, LineStyle = '-', lw = 1.5, color = nic_col, label = '_nolegend_') 
ax1.fill_between(time_series, Nicotine_mean-Nicotine_SEM, 
                 Nicotine_mean+Nicotine_SEM, color = nic_col, alpha=0.3, lw = 0.1, label = '_nolegend_')


ax1.plot(time_series, Both_mean, LineStyle = '-', lw = 1.5, color = both_col, label = '_nolegend_') 
ax1.fill_between(time_series, Both_mean-Both_SEM, 
                 Both_mean+Both_SEM, color = both_col, alpha=0.3, lw = 0.1, label = '_nolegend_')
ax1.set_ylim(-4,8) 
ax1.spines['top'].set_visible (False)
ax1.spines['right'].set_visible (False)
plt.xlabel("Time from drug (min.)")
plt.ylabel("$\Delta$F/F (%)")
plt.title("Basal DA levels")
plt.legend(("Liraglutide","Nicotine","Both"), loc = "upper right",frameon = False, prop={'size': 8})

#%% Transform timeseries to data with one point per minute

def blocks(data, block_size):
    
    no_blocks = int(len(data)/20/60)
    resized_data = np.zeros((no_blocks,))
    
    lower = 0
    step = int(1200*block_size)
    upper = step
    for i in range(no_blocks):
        
        resized_data[i] = np.mean(data[lower:upper])
        lower += step
        upper += step
        
    return resized_data


lira_export = np.zeros((100,9))
nico_export = np.zeros((100,9))
both_export = np.zeros((100,8))
for i in range(1,9):
    lira_export[:,i] = blocks(Liraglutide[46000:-4620,i], 1)
    nico_export[:,i] = blocks(Nicotine[46000:-4620,i], 1)
    if i == 8:
        continue
    both_export[:,i] = blocks(Both[46000:-4620,i], 1)
    
lira_export[:,0] = np.linspace(1,100,100)
nico_export[:,0] = np.linspace(1,100,100)
both_export[:,0] = np.linspace(1,100,100)

# np.savetxt("lira_1_min_bins.csv",lira_export,delimiter = ",")
# np.savetxt("nico_1_min_bins.csv",nico_export,delimiter = ",")
# np.savetxt("both_1_min_bins.csv",both_export,delimiter = ",")

#%% Boxplot

lira_pre = np.mean(Liraglutide[70000:100000,1:],0)
nic_pre = np.mean(Nicotine[70000:100000,1:],0)
both_pre = np.mean(Both[70000:100000,1:],0)

lira_0_20 = np.mean(Liraglutide[100000:124000,1:],0)
nic_0_20 = np.mean(Nicotine[100000:124000,1:],0)
both_0_20 = np.mean(Both[100000:124000,1:],0)

lira_20_40 = np.mean(Liraglutide[124000:148000,1:],0)
nic_20_40 = np.mean(Nicotine[124000:148000,1:],0)
both_20_40 = np.mean(Both[124000:148000,1:],0)

lira_40_60 = np.mean(Liraglutide[148000:,1:],0)
nic_40_60 = np.mean(Nicotine[148000:,1:],0)
both_40_60 = np.mean(Both[148000:,1:],0)

lira_mean = np.array([np.mean(lira_pre),np.mean(lira_0_20),np.mean(lira_20_40),np.mean(lira_40_60)])
lira_sem = np.array([stats.sem(lira_pre),stats.sem(lira_0_20),stats.sem(lira_20_40),stats.sem(lira_40_60)])
nic_mean = np.array([np.mean(nic_pre),np.mean(nic_0_20),np.mean(nic_20_40),np.mean(nic_40_60)])
nic_sem = np.array([stats.sem(nic_pre),stats.sem(nic_0_20),stats.sem(nic_20_40),stats.sem(nic_40_60)])
both_mean = np.array([np.mean(both_pre),np.mean(both_0_20),np.mean(both_20_40),np.mean(both_40_60)])
both_sem = np.array([stats.sem(both_pre),stats.sem(both_0_20),stats.sem(both_20_40),stats.sem(both_40_60)])

nic_x_pre = np.repeat(0.775,8)
nic_x_0_20 = np.repeat(1.775,8)
nic_x_20_40 = np.repeat(2.775,8)
nic_x_40_60 = np.repeat(3.775,8)
lira_x_pre = np.repeat(1,8)
lira_x_0_20 = np.repeat(2,8)
lira_x_20_40 = np.repeat(3,8)
lira_x_40_60 = np.repeat(4,8)
both_x_pre = np.repeat(1+0.225,7)
both_x_0_20 = np.repeat(2+0.225,7)
both_x_20_40 = np.repeat(3+0.225,7)
both_x_40_60 = np.repeat(4+0.225,7)


fig, ax1 = plt.subplots(figsize=(3, 3), dpi=300)
ax1.plot([0.5,4.5],[0,0], color = "k", LineStyle = "-",lw = 0.8, label = '_nolegend_')
ax1.bar(np.linspace(1,4,4)-0.225, nic_mean, width = 0.2, yerr = nic_sem, error_kw=dict(lw=1, capsize=1.5, capthick=1, ecolor = nic_col), edgecolor = nic_col, lw = 1.5, fill = False)
ax1.scatter(nic_x_pre,nic_pre, color = nic_col, s = 3,label = '_nolegend_')
ax1.scatter(nic_x_0_20,nic_0_20, color = nic_col, s = 3,label = '_nolegend_')
ax1.scatter(nic_x_20_40,nic_20_40, color = nic_col, s = 3,label = '_nolegend_')
ax1.scatter(nic_x_40_60,nic_40_60, color = nic_col, s = 3,label = '_nolegend_')
ax1.bar(np.linspace(1,4,4),lira_mean , width = 0.2, yerr = lira_sem, error_kw=dict(lw=1, capsize=1.5, capthick=1, ecolor = lira_col), edgecolor = lira_col, lw = 1.5,  fill = False )
ax1.scatter(lira_x_pre,lira_pre, color = lira_col, s = 3,label = '_nolegend_')
ax1.scatter(lira_x_0_20,lira_0_20, color = lira_col, s = 3,label = '_nolegend_')
ax1.scatter(lira_x_20_40,lira_20_40, color = lira_col, s = 3,label = '_nolegend_')
ax1.scatter(lira_x_40_60,lira_40_60, color = lira_col, s = 3,label = '_nolegend_')
ax1.bar(np.linspace(1,4,4)+0.225, both_mean, width = 0.2, yerr = both_sem, error_kw=dict(lw=1, capsize=1.5, capthick=1, ecolor = both_col), edgecolor = both_col, lw = 1.5,  fill = False)
ax1.scatter(both_x_pre,both_pre, color = both_col, s = 3,label = '_nolegend_')
ax1.scatter(both_x_0_20,both_0_20, color = both_col, s = 3,label = '_nolegend_')
ax1.scatter(both_x_20_40,both_20_40, color = both_col, s = 3,label = '_nolegend_')
ax1.scatter(both_x_40_60,both_40_60, color = both_col, s = 3,label = '_nolegend_')
ax1.spines['top'].set_visible (False)
ax1.spines['bottom'].set_visible (False)
ax1.spines['right'].set_visible (False)
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False) # labels along the bottom edge are off
ax1.set_xticks(np.linspace(1,4,4))
plt.xlim(0.5,4.5)
plt.ylim(-6,8)
xlabels = ["Pre",'0-20','20-40','40-60']
ax1.set_xticklabels(xlabels,rotation=45)
plt.xlabel("Time after drug adm. (min.)")
plt.ylabel("$\Delta$F/F$_0$ (%)")
#plt.title("Mean DA by time")
plt.legend(("Nicotine","Liraglutide","Both"),frameon = False, loc = "lower left", prop={'size': 8})

#%% Transform boxplot data to export format


lira_binned = np.vstack((lira_pre,lira_0_20,lira_20_40,lira_40_60))
lira_binned = np.column_stack((np.asarray([-20,0,20,40]),lira_binned))

nic_binned = np.vstack((nic_pre,nic_0_20,nic_20_40,nic_40_60))
nic_binned = np.column_stack((np.asarray([-20,0,20,40]),nic_binned))

both_binned = np.vstack((both_pre,both_0_20,both_20_40,both_40_60))
both_binned = np.column_stack((np.asarray([-20,0,20,40]),both_binned))

np.savetxt("lira_binned.csv",lira_binned,delimiter = ",")
np.savetxt("nico_binned.csv",nic_binned,delimiter = ",")
np.savetxt("both_binned.csv",both_binned,delimiter = ",")

