import pandas as pd
import numpy as np 
from sklearn import ensemble
import math
import matplotlib
import matplotlib.pyplot as plt 

filename = '/Users/Pushkarini/bike_requests_and_weather_summary.csv'
data = pd.read_csv(filename) 

a1 = data[data['month'] == 6].index.tolist(); p1 = a1[0:len(a1)/3]
p2 = data[data['month'] != 6].index.tolist(); p2.extend(a1[len(a1)/3::])
train = data.ix[p2]; 
test = data.ix[p1];
 
features = ['hour','month','dayofweek','workingday','hum','temp','precip'];
    
rX = train[features] 
ry = train["requests"]
tX = test[features] 
train_requests = train["requests"]
test_requests = test["requests"]

params = {'n_estimators': 2000, 'max_depth': 8, 'min_samples_split': 2,
      'learning_rate': 0.01, 'loss': 'ls', 'random_state' : 12345}

tree = ensemble.GradientBoostingRegressor(**params)
tree.fit(rX, ry)
 
ptest_requests =  tree.predict(tX); ptest_requests[ptest_requests<0] = 0
test_requests = np.asarray(test_requests) + 1
ptest_requests = np.asarray(ptest_requests) + 1

ptrain_requests =  tree.predict(rX); ptrain_requests[ptrain_requests<0] = 0
train_requests = np.asarray(train_requests) + 1
ptrain_requests = np.asarray(ptrain_requests) + 1

err_train = math.sqrt(np.mean(np.divide((train_requests-ptrain_requests),train_requests)**2))
err_test = math.sqrt(np.mean(np.divide((test_requests-ptest_requests),test_requests)**2))
 
 
matplotlib.rc('xtick', labelsize=30)
matplotlib.rc('ytick', labelsize=30)
f, axarr = plt.subplots(2, sharex=True,figsize=(27,18))
axarr[0].plot(test_requests/1000,'--or',markersize=12,mec = 'r',mfc='w',mew=5)
axarr[0].plot(ptest_requests/1000,'--ob',markersize=12,mec = 'b',mfc='w',mew=5)
axarr[0].set_ylabel('Requests (in thousands)',fontsize=30)
axarr[0].get_yaxis().set_label_coords(-0.05,0.5)
axarr[0].set_yticks((0.0,1.0,2.0,3.0,4.0))
axarr[0].set_title('Prediction in June',fontsize=30)
axarr[1].plot(test['precip'],'--ok',markersize=12,mec = 'k',mfc='w',mew=5)
axarr[1].set_ylabel('Precipitation',fontsize=30)
axarr[1].get_yaxis().set_label_coords(-0.05,0.5)
plt.yticks([0.0,0.2,0.4])
plt.xticks(np.arange(12,252,24),('day 1','day 2','day 3','day 4','day 5','day 6','day 7','day 8','day 9','day 10'),fontsize=30)
plt.xlim([0,240])
axarr[0].text(10,3.5,'data',fontsize=30,color='r')
axarr[0].text(10,4,'prediction',fontsize=30,color='b')
plt.show()
 
    
    


