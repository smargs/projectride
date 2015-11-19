import matplotlib.pyplot as plt 
import numpy as np
import matplotlib 
import matplotlib.cbook as cbook
from scipy.misc import imread

###################################################################
###################################################################
#### load data initially ##########################################
###################################################################
################################################################### 
 
# extract data from 12 months of bike share rides in nyc
fna = '/data/b'
filelist = [1410,1411,1412,1501,1502,1503,1504,1505,1506,1507,1508,1509]
 
###############################################################
###############################################################
###############################################################
### effect of gender as well as month on bike share demands ###
###############################################################
###############################################################
###############################################################

  
f = plt.figure(1)    
matplotlib.pylab.rc('font', family='serif', size=20)        
mlist = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
plt.xticks(range(len(np.linspace(0,11,12))), mlist, size='large')
menMeans   = gendermonth[0]/100000; womenMeans = gendermonth[1]/100000 
ind = np.arange(12)-0.3; width = 0.6
p1 = plt.bar(ind, menMeans,   width, color='#79BAEC')
p2 = plt.bar(ind, womenMeans, width, color='#F75D59', bottom=menMeans)
plt.title('Bikers by Gender and Month') 
ax = plt.gca(); ax.set_ylabel('Bikers (in millions)',labelpad = 10)
plt.legend( (p1[0], p2[0]), ('Men', 'Women') )
plt.show()

#################################################################
#################################################################
#################################################################
### average speed of bike ride ##################################
#################################################################
#################################################################
#################################################################
 
matplotlib.rc('xtick', labelsize=36)
matplotlib.rc('ytick', labelsize=36)

datafile = cbook.get_sample_data('/Users/Pushkarini/Desktop/dataincubator_work/images/map/allmap.png')
img = imread(datafile)
f = plt.figure(1,figsize = (12,17))
plt.imshow(img, zorder=0, extent=[min(ys), max(ys), min(xs), max(xs)+0.01])
plt.scatter(np.asarray(ys)+0.005,np.asarray(xs)+0.01,c=db.labels_*100,cmap=matplotlib.cm.cool,s=80,zorder = 1) 
plt.show()
plt.ylim([min(xs),max(xs)+0.01])
plt.xlim([min(ys),max(ys)])  
plt.gca().xaxis.set_major_locator(plt.NullLocator())    
plt.gca().yaxis.set_major_locator(plt.NullLocator())    
             
samples = 10000;        
f, axarr = plt.subplots(2, sharex=True,figsize=(13,15))
x = np.transpose(np.linspace(0,samples,samples+1));
axarr[0].plot(dm[0:samples+1],ridetimem[0:samples+1],'.b',markersize=10,markerfacecolor='w',markeredgewidth=1)
axarr[0].set_ylim(0, 2) 
axarr[0].set_xlim(0,7.5)
axarr[1].plot(df[0:samples+1],ridetimef[0:samples+1],'.r',markersize=10,markerfacecolor='w',markeredgewidth=1);
axarr[1].set_ylim(0, 2)
axarr[0].set_title('Men',fontsize=30)
axarr[1].set_title('Women',fontsize=30)
axarr[0].text(3,1.5,'8.25 miles/hr',fontsize=30)
axarr[1].text(3,1.5,'7.38 miles/hr',fontsize=30)
for ax in axarr:
    ax.set_ylabel('Time (hrs)',fontsize=30)
axarr[1].set_xlabel('Distance (miles)',fontsize=30)
m1,b1 = np.polyfit(dm[0:samples+1],ridetimem[0:samples+1],1) 
axarr[0].plot(dm[0:samples+1],m1*np.asarray(dm[0:samples+1]), '-k',linewidth=4) 
m2,b2 = np.polyfit(df[0:samples+1],ridetimef[0:samples+1],1) 
axarr[1].plot(df[0:samples+1],m2*np.asarray(df[0:samples+1]), '-k',linewidth=4)  




