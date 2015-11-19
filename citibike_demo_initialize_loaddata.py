import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib
import collections 
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
for i in range(len(filelist)):
    filename = str(fna+str(filelist[i])+'.csv')
    df = pd.read_csv(filename)
    if i == 0: 
        data = df
    else:
        data = pd.concat([data,df])
        
# take out white spaces from df columns for convenience
colnames = data.columns.values; 
colnames_new = []
for i in range(len(colnames)):
    colnames_new.append(colnames[i].replace(" ", ""))
data.columns =colnames_new

# extract year month day and time (=startime)
a = (data.starttime).tolist()
year = []; month = []; day = []; timehr = []; timemin = []; simday = [];
monthc = [0,31,28,31,30,31,30,31,31,30,31,30,31];
for i in range(len(a)):
    b = a[i]; b = b.split('/')
    year.append(b[2][0:4]); month.append(b[0]); day.append(b[1]); 
    c = b[2][4::]; c = c.split(':');  
    timehr.append(int(c[0])); timemin.append(int(c[1]));
    simday.append(int(b[2][0:4])*365+sum(monthc[0:int(b[0])])+int(b[1]))
    
year = map(int,year); month = map(int,month); day = map(int,day); 
simday = map(int,simday); simday = np.asarray(simday); simday = simday - min(simday);
data.year = pd.Series(year, index=data.index)
data.month = pd.Series(month, index=data.index)
data.day = pd.Series(day, index=data.index)
data.simday = pd.Series(simday, index=data.index)
data.timehr = pd.Series(timehr, index=data.index)
data.timemin = pd.Series(timemin, index=data.index)

gendermonth = np.zeros((2,12));
for i in range(12):
    mask1 = ((data.gender == 1) & (data.month == i+1))
    gendermonth[0][i] = np.sum(mask1)
    mask2 = ((data.gender == 2) & (data.month == i+1))
    gendermonth[1][i] = np.sum(mask2)

###############################################################
###############################################################
#### load data initially: speed ###############################
###############################################################
###############################################################


x_s = data.loc[:,'startstationlatitude']; y_s = data.loc[:,'startstationlongitude']
x_e = data.loc[:,'endstationlatitude']; y_e = data.loc[:,'endstationlongitude']
x_e1 = x_e.tolist(); y_s = y_s.tolist(); x_s1 = x_s.tolist(); 

xs = list(set(x_s));  
if len(list(set(y_s))) == len(xs): # we have unique x,y pairs. thanks heavens.
    ys = np.zeros(len(xs)); 
    for i,j in enumerate(xs):
        ys[i] = y_s[x_s1.index(j)] 

x_s = x_s.tolist(); x_e = x_e.tolist(); y_e = y_e.tolist();

X = np.vstack((np.transpose(xs),np.transpose(ys)))
X = np.transpose(X); 
from sklearn.cluster import DBSCAN
db = DBSCAN(eps=0.005, min_samples=1).fit(X);

x = collections.Counter(db.labels_).most_common(1)
manhatl = x[0][0];
xm = []; ym = [];
for i in range(len(X)):
    if db.labels_[i] == x[0][0]:
        xm.append(xs[i]); ym.append(ys[i])
import math  
def dist(x1,y1, x2,y2, x3,y3): # x3,y3 is the point
    px = x2-x1; py = y2-y1; something = px*px + py*py
    u =  ((x3 - x1) * px + (y3 - y1) * py) / float(something)
    if u > 1:
        u = 1
    elif u < 0:
        u = 0
    x = x1 + u * px; y = y1 + u * py; dx = x - x3; dy = y - y3
    dist = math.sqrt(dx*dx + dy*dy);  
    return dist
 

datagen = list(data.gender)
xman_s = []; xman_e = []; yman_s = []; yman_e = []; c = 0;   
d = []; ridetime = []; dur = (data.tripduration); dur = dur.tolist();
xg1 = 40.768103; yg1 = -74.004165
xg2 = 40.777008; yg2 = -73.997470   
dm = []; ridetimem = []; df = []; ridetimef = [];                
for i in range(len(x_e)):
    if x_s[i] in xm and y_s[i] in ym and x_e[i] in xm and y_e[i] in ym:
        xman_s.append(x_s[i]); yman_s.append(y_s[i]);
        xman_e.append(x_e[i]); yman_e.append(y_e[i]);
        d1 = dist(xg1,yg1, xg2,yg2, xman_s[c],yman_s[c])
        d2 = dist(xg1,yg1, xg2,yg2, xman_e[c],yman_e[c])
        d3 = math.sqrt((xman_s[c]-xman_e[c])**2+(yman_s[c]-yman_e[c])**2);
        d4 = abs(d2-d1);
        d5 = math.sqrt(d3**2-d4**2);
        dfin = (d4+d5);
        if dfin != 0.0:  
            d.append(dfin); ridetime.append(dur[i])
            if datagen[i] == 1:
                dm.append((d4+d5)); ridetimem.append(dur[i])
            if datagen[i] == 2:
                df.append((d4+d5)); ridetimef.append(dur[i])
        c = c + 1;

d = [x * 110.0/1.6 for x in d];
dm = [x * 110.0/1.6 for x in dm]; 
df = [x * 110.0/1.6 for x in df];

ridetime  = [x * 1./3600 for x in ridetime];  
ridetimem  = [x * 1./3600 for x in ridetimem]; 
ridetimef  = [x * 1./3600 for x in ridetimef];              
             
sm =  (np.divide(dm,ridetimem)); sf =  (np.divide(df,ridetimef));

