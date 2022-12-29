# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 08:02:44 2022

@author: Amit Ranjan Kar
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.dates as mdates
#conda install -c conda-forge cartopy #install from conda prompt 
import cartopy
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import cartopy.crs as ccrs
import seaborn as sns
import netCDF4
import xarray as xr
#--------------------------------------------------------------------
def mm2inch(*tupl):
    inch = 25.4
    if isinstance(tupl[0], tuple):
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)
def mm2point(mm):
    return mm/(25.4/72)
font = {'family' : 'Arial',
         'weight' : 'normal',
         'size'   : 15}
mpl.rc('font', **font)
mpl.rcParams['axes.linewidth'] = mm2point(0.4)
mpl.rcParams['ytick.major.width'] = mm2point(0.4)
mpl.rcParams['xtick.major.width'] = mm2point(0.4)
#--------------------------------------------------------------------

path='D:/BTU FOLDER/Atmospheric water/Final/Data_finalEx/' #Working directory 

dfZ=df= pd.read_csv(path + 'mrr_Z.csv', sep=";", header=0, 
                    parse_dates=[0], index_col=0, na_values = -999)  #Reading reflectivity data 

dflwc=pd.read_csv(path + 'mrr_LWC.csv', sep=";", header=0,  
                    parse_dates=[0], index_col=0, na_values = -999)  #Reading liquid water content (LWC) data 

dfrr=pd.read_csv(path + 'mrr_RR.csv', sep=";", header=0,  
                    parse_dates=[0], index_col=0, na_values = -999)  #Reading rain rate data  

dfw=pd.read_csv(path + 'mrr_W.csv', sep=";", header=0,  
                    parse_dates=[0], index_col=0, na_values = -999)  #Reading terminal fall velocity data 

#Dates are sliced into 2 separate dates.
#Slicing from a dataset of reflectivity
dfZ_17=dfZ['2017-02-22 00:00:00':'2017-02-22 23:59:00'] 
dfZ_18=dfZ['2018-01-12 00:00:00':'2018-01-12 23:59:00'] 

#Slicing from a liquid of water Dataset LWC
dflwc_17=dflwc['2017-02-22 00:00:00':'2017-02-22 23:59:00'] 
dflwc_18=dflwc['2018-01-12 00:00:00':'2018-01-12 23:59:00'] 

#Slicing from a dataset of rainfall rates
dfrr_17=dfrr['2017-02-22 00:00:00':'2017-02-22 23:59:00'] 
dfrr_18=dfrr['2018-01-12 00:00:00':'2018-01-12 23:59:00'] 

#Slicing from a dataset of terminal fall velocity
dfw_17=dfw['2017-02-22 00:00:00':'2017-02-22 23:59:00'] 
dfw_18=dfw['2018-01-12 00:00:00':'2018-01-12 23:59:00'] 
#########################################################################################

#MRR readings are shown in a panel (2017)

figure, (ax1,ax2,ax3,ax4) = plt.subplots (4,1, figsize=(mm2inch(200,200)))
#MRR Reflectivity Plotting (2017)
figure.add_subplot(ax1)
yy1,xx1= np.meshgrid([int(x) for x in dfZ_17.columns], dfZ_17.index)
z1= dfZ_17.to_numpy()
a1= ax1.pcolormesh(xx1,yy1,z1, cmap= 'jet', shading= 'flat')
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%H"))
figure.colorbar(a1, ax=ax1)
ax1.set_title("MRR: Reflectivity (2017)")

#MRR LWC Plotting (2017)
figure.add_subplot(ax2)
yy2,xx2= np.meshgrid([int(x) for x in dflwc_17.columns], dflwc_17.index)
z2= dflwc_17.to_numpy()
a2= ax2.pcolormesh(xx2,yy2,z2, cmap= 'jet', shading= 'flat')
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%H"))
figure.colorbar(a2, ax=ax2)
ax2.set_title("MRR: liquid Water Content (2017)")

#MRR rain rate Plotting (2017)
figure.add_subplot(ax3)
yy3,xx3= np.meshgrid([int(x) for x in dfrr_17.columns], dfrr_17.index)
z3= dfrr_17.to_numpy()
a3= ax3.pcolormesh(xx3,yy3,z3, cmap= 'jet', shading= 'flat')
ax3.xaxis.set_major_formatter(mdates.DateFormatter("%H"))
figure.colorbar(a3, ax=ax3)
ax3.set_title("MRR: Rain Rate (2017)")

#The final fall velocity is shown. Rainfall rate (2017)
figure.add_subplot(ax4)
yy4,xx4= np.meshgrid([int(x) for x in dfw_17.columns], dfw_17.index)
z4= dfw_17.to_numpy()
a4= ax4.pcolormesh(xx4,yy4,z4, cmap= 'jet', shading= 'flat')
ax4.xaxis.set_major_formatter(mdates.DateFormatter("%H"))
figure.colorbar(a4, ax=ax4)
ax4.set_title("MRR: Terminal Fall Velocity (2017)")
figure.tight_layout()
figure.savefig('D:/BTU FOLDER/Atmospheric water/Final/Final Picture/ MRR_2017.png',format='png', dpi=300, bbox_inches='tight')
figure.show()

##Panel plots of MRR values (2018)

figure1, (ax1,ax2,ax3,ax4) = plt.subplots (4,1, figsize=(mm2inch(200,200)))
#Plotting for MRR Reflectivity (2018)
figure1.add_subplot(ax1)
yy1,xx1= np.meshgrid([int(x) for x in dfZ_18.columns], dfZ_18.index)
z1= dfZ_18.to_numpy()
a1= ax1.pcolormesh(xx1,yy1,z1, cmap= 'jet', shading= 'flat')
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%H"))
figure1.colorbar(a1, ax=ax1)
ax1.set_title("MRR: Reflectivity (2018)")

#Plotting for MRR LWC (2018)
figure1.add_subplot(ax2)
yy2,xx2= np.meshgrid([int(x) for x in dflwc_18.columns], dflwc_18.index)
z2= dflwc_18.to_numpy()
a2= ax2.pcolormesh(xx2,yy2,z2, cmap= 'jet', shading= 'flat')
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%H"))
figure1.colorbar(a2, ax=ax2)
ax2.set_title("MRR: liquid Water Content (2018)")

#Plotting for MRR rain rate (2018)
figure1.add_subplot(ax3)
yy3,xx3= np.meshgrid([int(x) for x in dfrr_18.columns], dfrr_18.index)
z3= dfrr_18.to_numpy()
a3= ax3.pcolormesh(xx3,yy3,z3, cmap= 'jet', shading= 'flat')
ax3.xaxis.set_major_formatter(mdates.DateFormatter("%H"))
figure1.colorbar(a3, ax=ax3)
ax3.set_title("MRR: Rain Rate (2018)")

#The final fall velocity is shown. Rainfall rate (2017)
figure1.add_subplot(ax4)
yy4,xx4= np.meshgrid([int(x) for x in dfw_17.columns], dfw_17.index)
z4= dfw_17.to_numpy()
a4= ax4.pcolormesh(xx4,yy4,z4, cmap= 'jet', shading= 'flat')
ax4.xaxis.set_major_formatter(mdates.DateFormatter("%H"))
figure1.colorbar(a4, ax=ax4)
ax4.set_title("MRR: Terminal Fall Velocity (2018)")
figure1.tight_layout()
figure1.savefig('D:/BTU FOLDER/Atmospheric water/Final/Final Picture/ MRR_2018.png',format='png', dpi=300, bbox_inches='tight')
figure1.show()

#------------------------------------------------------------------------------------------------

#Rainfall classification (convective/stratified)


def plot_CC_for_case(
        inputDF, # Hold's the Data of the Case
        sfig=None # Optional in case you already have a plot
        ):
    
    if sfig is None:
         sfig = plt.gca()             
    sfig.plot(1.17, 4.83, 'ro', label="Convective CC", color="blue", markersize=20)
    sfig.plot(0.92, 3.74, 'ro', label="Stratiform CC",color="green", markersize=20)
    sfig.plot(1.49, 3.62, 'ro', label="X CC", color="orange", markersize=20)
    sfig.scatter(inputDF.Dm, inputDF.Nw, c='black', s=150, zorder=4, label='Case')
    sfig.grid()
    sfig.legend()
    sfig.set_xlabel('$D_m$')
    sfig.set_ylabel('$log_{10} N_w$')
    sfig.plot([1, 2], [4.2,2.5 ], color='k', linestyle='--', linewidth=2)
    sfig.set_ylim(0.5,6.5)
    sfig.set_xlim(0.4,2.5)
    sfig.set_title(inputDF.index[0].strftime('From %b %d %Y %H:%M till ')+inputDF.index[-1].strftime('%b %d %Y %H:%M'))
        
    return sfig
#--- plot ended_CC_for_case---------------------------------------------------#

df_dist=df= pd.read_csv(path + 'dropsize_dist.csv', sep=";", header=0, 
                    parse_dates=[0], index_col=0, na_values = -999)  

# Slicing into two separate dates
df_dist17=df_dist['2017-02-22 00:00:00':'2017-02-22 23:55:00'] 
df_dist18=df_dist['2018-01-12 00:00:00':'2018-01-12 23:55:00'] 

#Creating a graph of rain rates across the time period during which the rain event happens (2017)
figure1, (ax1,ax2) = plt.subplots(2,1, figsize=(mm2inch(240,200)))
ax1.plot(df_dist17.index, df_dist17.RR,"b--",linewidth=2, label="Rain rate (2017-02-22)" )
ax1.set_ylabel('Rain rate (mm)')
ax1.set_xlabel('Time (Hour)')
ax1.set_title('Rain rates of the rain event')
ax1.set_yticks(np.arange(0,10,2))
ax1.legend(loc="upper left", ncol=1 )
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H'))

ax2.plot(df_dist18.index, df_dist18.RR,"b--",linewidth=2, label="Rain rate (2018-01-12)" )
ax2.set_ylabel('Rain rate (mm)')
ax2.set_xlabel('Time (Hour)')
ax2.set_yticks(np.arange(0,25,4))
ax2.legend(loc="upper left", ncol=1 )
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H'))
figure1.tight_layout()
figure1.savefig('D:/BTU FOLDER/Atmospheric water/Final/Final Picture/ rain_rate.png',format='png', dpi=300, bbox_inches='tight')
figure1.show()

#plot of Dn and Nw for 2017-02-22
figure2 = plt.figure(figsize=(12,12))
ax = figure2.add_subplot()
ax = plot_CC_for_case(df_dist17,ax)
figure2.savefig('D:/BTU FOLDER/Atmospheric water/Final/Final Picture/ Dn_vs_Nw(2017).png',format='png', dpi=300, bbox_inches='tight')

#plot of Dn and Nw for 2018-01-12
figure3 = plt.figure(figsize=(12,12))
ax = figure3.add_subplot()
ax = plot_CC_for_case(df_dist18,ax)
figure3.savefig('D:/BTU FOLDER/Atmospheric water/Final/Final Picture/ Dn_vs_Nw(2018).png',format='png', dpi=300, bbox_inches='tight')
#--------------------------------------------------------------------------------------------

##Read_plot_trajectories (2017) 
#---------------------------------
font = {'family' : 'Arial',
        'weight' : 'normal',
        'size'   : 8}
mpl.rc('font', **font)
def mm2inch(*tupl):
    inch = 25.4
    if isinstance(tupl[0], tuple):
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)
#---End of mm2inch------------------------------------------------------------#
def mm2point(mm):
    return mm/(25.4/72)
#---End of mm2point-----------------------------------------------------------#
mwidth = mm2point(0.2)
#--------------------------------------------------------------------------------------- 
def add_trajectory(fig, inputDF, lnstyle, lnwidth):
    for trajectories, trajectoy in inputDF.groupby(inputDF.index):
        fig.plot(trajectoy.lon.values,trajectoy.lat.values,linestyle=lnstyle,linewidth=lnwidth,transform=ccrs.PlateCarree(), label=trajectories.strftime('%b %d %Y %H:%M'))
#--------------------------------------------------------------#
def add_map(fig, inputDF, lnstyle, lnwidth):
    add_trajectory(fig = fig, inputDF = inputDF, lnstyle = lnstyle, lnwidth = lnwidth)   
    fig.set_xticks([-80, -70, -60, -50, -40],crs=ccrs.PlateCarree())
    fig.set_yticks([-30, -20, -10, 0, 10],crs=ccrs.PlateCarree())
    fig.set_extent([-88, -32, -25, 15],crs=ccrs.PlateCarree())
    fig.add_feature(cartopy.feature.LAND.with_scale('110m'), zorder=0, color='lightgrey')
    fig.add_feature(cartopy.feature.BORDERS.with_scale('110m'), zorder=0, edgecolor='black', linewidth=mwidth)
    fig.add_feature(cartopy.feature.COASTLINE.with_scale('110m'), zorder=0, edgecolor='black', linewidth=mwidth)
    fig.spines['geo'].set_linewidth(mwidth)
    fig.xaxis.set_major_formatter(LongitudeFormatter())
    fig.yaxis.set_major_formatter(LatitudeFormatter())
    fig.tick_params(reset=True,axis='both',which='major',labelsize=8,direction='in',bottom = True, top = True, left = True, right = True, width = mwidth)  
    fig.label_outer()  
    fig.set_title(inputDF.index[0].strftime('From %b %d %Y %H:%M till ')+inputDF.index[-1].strftime('%b %d %Y %H:%M'))
    lg=fig.legend(loc='center left',bbox_to_anchor=(0.0, -0.1), ncol=4, fontsize=6,fancybox = True, borderpad=0.4)
    lg.get_frame().set_edgecolor('black')
    lg.get_frame().set_linewidth(mwidth)
#---End of add_map------------------------------------------------------------#

#Plotting trajectories of 2017-02-22 
srcfile='D:/BTU FOLDER/Atmospheric water/Final/Data_finalEx/'
# Case Convection -------------------------------------------------------------
start = '2017-02-22 00:00:00' # insert starting date
end = '2017-02-22 23:00:00'   # insert end date
caseDF = pd.read_csv(srcfile+ "trajectories_2017-02-22.csv",sep= ",", parse_dates=['date','date2'],index_col=['date']).loc[start:end]
caseDF_mean = caseDF.groupby('hour.inc').mean()

#Plottting
figure, case = plt.subplots(1, 1, subplot_kw={'projection': ccrs.PlateCarree()},figsize=(mm2inch(150.0,150.0)),gridspec_kw = {'wspace':0.0, 'hspace':0.0})
add_map(case,caseDF,'solid', mm2point(0.4))
figure.savefig('D:/BTU FOLDER/Atmospheric water/Final/Final Picture/Trajectory(2017).png',format='png', dpi=300, bbox_inches='tight') 

# Plotting linegraphs 
figure,ax = plt.subplots()
caseDF_mean['height'].plot(ax=ax, color="red")
# set x-axis label
ax.set_xlabel('Hours',fontsize=8)
# set y-axis label
ax.set_ylabel('Height in m (asl)',fontsize=8,color="red")
ax2=ax.twinx()
caseDF_mean['qvapor'].plot(ax=ax2, color="blue")
ax2.set_ylabel("Q-Vapor g/kg",color="blue",fontsize=8)
ax.invert_xaxis()
ax.set_title('Convectiv(2017-02-22)', fontweight='bold')
figure.savefig('D:/BTU FOLDER/Atmospheric water/Final/Final Picture/Convectiv_Case_lines (2017).png', bbox_inches='tight', dpi=300)

#Plotting trajectories of 2018-01-12 
srcfile='D:/BTU FOLDER/Atmospheric water/Final/Data_finalEx/'
# Case Convection -------------------------------------------------------------
start = '2018-01-12 00:00:00' # insert your starting date
end = '2018-01-12 23:00:00'   # insert your end date
caseDF = pd.read_csv(srcfile+ "trajectories_2018-01-12.csv",sep= ",", parse_dates=['date','date2'],index_col=['date']).loc[start:end]
caseDF_mean = caseDF.groupby('hour.inc').mean()

#Plottting
figure, case = plt.subplots(1, 1, subplot_kw={'projection': ccrs.PlateCarree()},figsize=(mm2inch(150.0,150.0)),gridspec_kw = {'wspace':0.0, 'hspace':0.0})
add_map(case,caseDF,'solid', mm2point(0.4))
figure.savefig('D:/BTU FOLDER/Atmospheric water/Final/Final Picture/Trajectory(2018).png',format='png', dpi=300, bbox_inches='tight') 

# Plotting linegraphs --------------------------------------------
figure,ax = plt.subplots()
caseDF_mean['height'].plot(ax=ax, color="red")
# set x-axis label
ax.set_xlabel('Hours',fontsize=8)
# set y-axis label
ax.set_ylabel('Height in m (asl)',fontsize=8,color="red")
ax2=ax.twinx()
caseDF_mean['qvapor'].plot(ax=ax2, color="blue")
ax2.set_ylabel("Q-Vapor g/kg",color="blue",fontsize=8)
ax.invert_xaxis()
ax.set_title('Convectiv (2018-01-12)', fontweight='bold')
figure.savefig('D:/BTU FOLDER/Atmospheric water/Final/Final Picture/Convectiv_Case_lines (2018).png', bbox_inches='tight', dpi=300)

#Analysis of brightness temperature 
temp17= xr.open_dataset(path+'2017-02-22.nc')
temp18= xr.open_dataset(path+'2018-01-12.nc')

#Brightness temperature on 17-02-22
mean_t_17 = temp17.Tb.groupby("time.hour").mean(["lat", "lon"])
figure1, ax1 = plt.subplots(figsize=(mm2inch(200,100)))
mean_t_17.plot.line(marker="o",
                              ax=ax1,
                              color="grey",
                              markerfacecolor="red",
                              markeredgecolor="red")
ax1.set(title="Brightness Temperature (2017) ")
ax1.set_xlabel('Time',fontsize=12)
ax1.set_xlabel('Time',fontsize=12)
ax.set_ylabel('Brightness Temperature (K)',fontsize=12)
figure1.savefig('D:/BTU FOLDER/Atmospheric water/Final/Final Picture/Brightness_temp (2017).png', bbox_inches='tight', dpi=300)
figure1.show

#Brightness temperature on 18-01-12
mean_t_18 = temp18.Tb.groupby("time.hour").mean(["lat", "lon"])
figure2, ax2 = plt.subplots(figsize=(mm2inch(200,100)))
mean_t_18.plot.line(marker="o",
                              ax=ax2,
                              color="grey",
                              markerfacecolor="red",
                              markeredgecolor="red")
ax2.set(title="Brightness Temperature (2018) ")
ax2.set_xlabel('Time',fontsize=12)
ax2.set_xlabel('Time',fontsize=12)
ax.set_ylabel('Brightness Temperature (K)',fontsize=12)
figure2.savefig('D:/BTU FOLDER/Atmospheric water/Final/Final Picture/Brightness_temp (2018).png', bbox_inches='tight', dpi=300)
figure2.show





