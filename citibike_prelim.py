import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib
 
 
# extract data from 12 months of bike share rides in nyc
fna = '/Users/Pushkarini/Desktop/project_ride/data/b'
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


###############################################################
### effect of gender as well as month on bike share demands ###
###############################################################

gendermonth = np.zeros((2,12));
for i in range(12):
    mask1 = ((data.gender == 1) & (data.month == i+1))
    gendermonth[0][i] = np.sum(mask1)
    mask2 = ((data.gender == 2) & (data.month == i+1))
    gendermonth[1][i] = np.sum(mask2)
  
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

###############################################################
### know more: how long is duration of each trip ##############
###############################################################

dur = (data.tripduration); dur = dur.tolist();  
import collections
counter=collections.Counter(dur)
v1 = counter.keys(); v1 = map(float,v1); 
v1 = np.asarray(v1); v1 = v1/60; v2=counter.values()

z = 60; w1 = v1[0::z]; w2 = v2[0::z]
f = plt.figure(2)
plt.bar(w1[0:3000/z], w2[0:3000/z], 0.6, color='r')
ax = plt.gca(); ax.set_xlabel('Duration (minutes)',labelpad = 10)
ax.set_ylabel('Number of customers',labelpad = 10)


hdum = np.zeros((24,365));
for i in range(len(data)): 
    a1 = simday[i]; a2 = timehr[i];
    hdum[a2][a1] = hdum[a2][a1]+1

             
hourly = np.zeros(24); hourly_std = np.zeros(24);
for i in range(24):
    dum = hdum[i]; dum = filter(lambda a: a != 0, dum);
    hourly[i] = np.mean(dum); hourly_std[i] = np.std(dum); 
xhours = np.linspace(0,23,24);

f = plt.figure(11)    
matplotlib.pylab.rc('font', family='serif', size=20)          
ind = np.arange(24); width = 0.6
p1 = plt.bar(ind, hourly,   width, color='#79BAEC')
 
f = plt.figure(4)
plt.fill_between(xhours, hourly-hourly_std/20, hourly+hourly_std/20, color='grey', alpha='0.3')
plt.plot(xhours,hourly);  


#############################################################
### birthyear historgram ####################################  
## sudden spike after 24 also indicates 'working people' ####
############################################################# 

dumyear = np.asarray(data.birthyear);
a = 1916; b = int(max(dumyear));
dumyear[np.isnan(dumyear)] = 0; dumyear = dumyear.astype(int)
age = np.zeros(100);
for i in range(a,b+1): 
    age[2015-i] = np.sum(dumyear == i)  # neglected increase in age in a year!
 
f = plt.figure(5)    
matplotlib.pylab.rc('font', family='serif', size=20)          
ind = np.arange(100); width = 0.6
p1 = plt.bar(ind, age/1000000,   width, color='#79BAEC')
plt.title('Bikers by Age') 
ax = plt.gca(); ax.set_ylabel('Bikers (in millions)',labelpad = 10)
plt.show()

#############################################
### see influence of weather on demand ######
#############################################


filename = '/Users/Pushkarini/Desktop/project_ride/data/weather_sample.txt'
f = open(filename, 'r'); a = "";
for line in f:
    b = repr(line)
    a+=str(b)
     
lines = [] 
for line in a.split('\\n'): 
    columns = line.split(', '); lines.append(columns)
 
tavg = np.zeros(122); precip = np.zeros(122); precipr = np.zeros(122);
for i in range(7,129):
    x = lines[i][2]; 
    tavg[i-7] = float(x[1:3]);  
    y = lines[i][lines[i].index(',M')+1];
    z = lines[i][lines[i].index(',M')+2];
    if len(y) == 0:
        y = lines[i][lines[i].index(',M')+2];
        z = lines[i][lines[i].index(',M')+3];
    if len(z) == 0:        
        z = lines[i][lines[i].index(',M')+3]; 
        if len(z) == 0:
            z = lines[i][lines[i].index(',M')+4]; 
            
    if y[1::] == 'T': # traces found
        precip[i-7] = 0.001;
    else:
        precip[i-7] = float(y[1::]);
    if z[1::] == 'T': # traces found
        precipr[i-7] = 0.001;
    else:
        precipr[i-7] = float(z[1::]);
 
f.close() 

wtest_data = np.zeros((122)); c = 0;
mlist = [31,30,31,30];
for j in range(4):
    for i in range(mlist[j]):
        mask1 = ((data.month == j+3) & (data.year == 2015) & (data.day==i+1))
        wtest_data[c] = np.sum(mask1)
        c = c + 1; 
    

f, axarr = plt.subplots(4, sharex=True)
x = np.linspace(0,121,122);
axarr[0].plot(x,wtest_data/10000,'--.g',markersize=20,markerfacecolor='w',markeredgewidth=2)
axarr[0].set_title('Day')
axarr[1].plot(x,tavg,'--.r',markersize=20,markerfacecolor='w',markeredgewidth=2);
axarr[2].plot(x,precip,'--.k',markersize=20,markerfacecolor='w',markeredgewidth=2);
axarr[3].plot(x,precipr,'--.b',markersize=20,markerfacecolor='w',markeredgewidth=2);
plt.xlim(0, 30)
    
    
x_s = data.loc[:,'startstationlatitude']; y_s = data.loc[:,'startstationlongitude']
x_e = data.loc[:,'endstationlatitude']; y_e = data.loc[:,'endstationlongitude']
x_e1 = x_e.tolist(); y_s = y_s.tolist();
x_s1 = x_s.tolist(); 

 
xs = list(set(x_s));  
if len(list(set(y_s))) == len(xs): # we have unique x,y pairs. thanks heavens.
    ys = np.zeros(len(xs)); 
    for i,j in enumerate(xs):
        ys[i] = y_s[x_s1.index(j)] 
    
    
x_s = x_s.tolist(); x_e = x_e.tolist();
y_e = y_e.tolist();

X = np.vstack((np.transpose(xs),np.transpose(ys)))
X = np.transpose(X); 
from sklearn.cluster import DBSCAN
db = DBSCAN(eps=0.005, min_samples=1).fit(X);

plt.scatter(ys,xs,c=db.labels_*100,s=100)   

x = collections.Counter(db.labels_).most_common(1)
manhatl = x[0][0];
xm = []; ym = [];
for i in range(len(X)):
    if db.labels_[i] == x[0][0]:
        xm.append(xs[i]); ym.append(ys[i])
        
plt.scatter(ym,xm)

 
import math  
def dist(x1,y1, x2,y2, x3,y3): # x3,y3 is the point
    px = x2-x1; py = y2-y1; something = px*px + py*py
    u =  ((x3 - x1) * px + (y3 - y1) * py) / float(something)
    if u > 1:
        u = 1
    elif u < 0:
        u = 0
    x = x1 + u * px; y = y1 + u * py; dx = x - x3; dy = y - y3
    dist = math.sqrt(dx*dx + dy*dy); proj = [x,y]
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
dm = [x * 110.0/1.6 for x in dm]; df = [x * 110.0/1.6 for x in df];

ridetime  = [x * 1./3600 for x in ridetime];  
ridetimem  = [x * 1./3600 for x in ridetimem]; ridetimef  = [x * 1./3600 for x in ridetimef];              
             
sm =  (np.divide(dm,ridetimem));
sf =  (np.divide(df,ridetimef));            
             
sams = 100000;        
f, axarr = plt.subplots(2, sharex=True)
x = np.transpose(np.linspace(0,sams,sams+1));
axarr[0].plot(dm[0:sams+1],ridetimem[0:sams+1],'.g',markersize=10,markerfacecolor='w',markeredgewidth=1)
plt.ylim(0, 2) 
axarr[1].plot(df[0:sams+1],ridetimef[0:sams+1],'.r',markersize=10,markerfacecolor='w',markeredgewidth=1);
plt.ylim(0, 2)
plt.xlim(0,7.5)
 

