import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

filename = '/Users/Pushkarini/Desktop/dataincubator_work/data/b1509.csv'

data = pd.read_csv(filename) 

# take out white spaces from df columns for convenience
colnames = data.columns.values; colnames_new = []
for i in range(len(colnames)):
    colnames_new.append(colnames[i].replace(" ", ""))
data.columns =colnames_new

data['sdatehour'] = data.starttime.apply(lambda x: x.split(':')[0])
data['month'] = data.sdatehour.apply(lambda x: int(x.split('/')[0]))
data['sday'] = data.sdatehour.apply(lambda x: int(x.split('/')[1]))
data['year1'] = data.sdatehour.apply(lambda x: x.split('/')[2]) 
data['year'] = data.year1.apply(lambda x: int(x[0:4])) 
data['shour'] = data.sdatehour.apply(lambda x: int(x[-2::]))
data['smin'] = data.starttime.apply(lambda x: int(x.split(':')[1]))
data['ssec'] = data.starttime.apply(lambda x: int(x.split(':')[2]))
data['edatehour'] = data.stoptime.apply(lambda x: x.split(':')[0]) 
data['eday'] = data.edatehour.apply(lambda x: int(x.split('/')[1])) 
data['ehour'] = data.edatehour.apply(lambda x: int(x[-2::]))
data['emin'] = data.stoptime.apply(lambda x: int(x.split(':')[1]))
data['esec'] = data.stoptime.apply(lambda x: int(x.split(':')[2]))
 
myday = 1;
data = data[data.sday==myday]

data['startsec'] = data.apply(lambda row: ((row['sday']-myday)*86400+row['shour']*3600+row['smin']*60+row['ssec']),axis=1)
data['stopsec'] = data.apply(lambda row: ((row['eday']-myday)*86400+row['ehour']*3600+row['emin']*60+row['esec']),axis=1)
 
data['startmin'] = data.apply(lambda row: (row['startsec']/60),axis=1)
data['stopmin'] = data.apply(lambda row: (row['stopsec']/60),axis=1)

allbid = list(set(data.bikeid));  

c1 = 0; c2 = 0;

inflow = []; outflow = [];
bmat = np.zeros((max(list(set(data.startstationid)))+1,1440))
for b in range(len(allbid)):
    bid = allbid[b];
    bdata = data[data.bikeid == bid];
    for i in range(len(bdata)): 
        x = bdata.iloc[i]['startmin']; 
        if i == 0:
            bmat[bdata.iloc[i]['startstationid']][0:x] = bmat[bdata.iloc[i]['startstationid']][0:x] + 1
        else:
            y = bdata.iloc[i-1]['stopmin'];
            if bdata.iloc[i-1]['endstationid'] == bdata.iloc[i]['startstationid']:
                bmat[bdata.iloc[i]['startstationid']][y+1:x+1] = bmat[bdata.iloc[i]['startstationid']][y+1:x+1] + 1
                c1 = c1 + 1;
            else:
                inflow.append([bdata.iloc[i]['startstationid'],x])
                outflow.append([bdata.iloc[i-1]['endstationid'],y])
                c2 = c2 + 1;
            
            #print float(y)/60,float(x)/60, bdata.iloc[i-1]['endstationid']
    bmat[bdata.iloc[len(bdata)-1]['endstationid']][bdata.iloc[len(bdata)-1]['stopmin']:3600] = bmat[bdata.iloc[len(bdata)-1]['endstationid']][bdata.iloc[len(bdata)-1]['stopmin']:3600] + 1

import matplotlib

bmeans = np.sum(bmat,1); actives = np.where(bmeans>0);
bmat = bmat[actives]

inflow = np.asarray(inflow); inflow.sort(axis=0)
inflow = np.transpose(inflow); in1 = inflow[0]; in2 = inflow[1];
infin = np.zeros((max(list(set(data.startstationid)))+1)); 
inset = sorted(list(set(in1)));
for i in range(len(inset)):
    a = np.where(in1==inset[i]); a = in2[a]; a = min(a) - 10;
    infin[inset[i]]=a 
infin = infin[actives]
outflow = np.asarray(outflow); outflow.sort(axis=0)
outflow = np.transpose(outflow); out1 = outflow[0]; out2 = outflow[1];
outfin = np.zeros((max(list(set(data.startstationid)))+1)); 
outset = sorted(list(set(out1)));
for i in range(len(outset)):
    a = np.where(out1==outset[i]); a = out2[a]; a = min(a) + 10;
    outfin[outset[i]]=a    
outfin = outfin[actives]

#tbmat = np.transpose(bmat); tbmat = sum(tbmat,0)
#order = [i[0] for i in sorted(enumerate(tbmat), key=lambda x:x[1])]
#bmat = bmat[order]

#plt.plot([y+10,y+10],[bdata.iloc[i-1]['endstationid'],bdata.iloc[i-1]['endstationid']+1],linewidth=4,c='g')  
#plt.plot([x-10,x-10],[bdata.iloc[i]['endstationid'],bdata.iloc[i]['endstationid']+1],linewidth=4,c='r')

import matplotlib
import matplotlib.pyplot as plt
matplotlib.rc('xtick', labelsize=20)
matplotlib.rc('ytick', labelsize=20)

 
randset = [111,114,115,181, 319, 123,27, 99,113,180,182,183,184,38,224,251,221,262,373,65,255,112,309,110,341]
np.random.shuffle(randset) 
f = plt.figure(figsize=(20,15))
plt.title('Number of bikes at stations - 1st September, 2015',fontsize=30)
heatmap = plt.pcolor(bmat[randset], cmap=matplotlib.cm.Blues)
out1 = outfin[randset]; in1 = infin[randset]; 
for i in range(len(out1)):
    if out1[i] > 0:
        plt.plot([out1[i],out1[i]],[i+0.2,i+1-0.2],linewidth=8,c='r')  
 
for i in range(len(in1)):
    if in1[i] > 0:
        plt.plot([in1[i],in1[i]],[i+0.2,i+1-0.2],linewidth=8,c='g')  

for i in range(len(in1)):
    plt.plot([0,1440],[i,i],linewidth=1,c='w')

cbar = plt.colorbar(heatmap)
cbar.ax.tick_params(labelsize=30) 
plt.ylabel('Station Id',fontsize=30)
plt.xticks(np.arange(180,1440-180+360,360),('3:00am','9:00am','3:00pm','9:00pm','12:00pm'),fontsize=30)
plt.xlim(0,1440); plt.ylim(0,25)
plt.yticks()
        
plt.show()
    

 
 

 
 
 
 
 
 