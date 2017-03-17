# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 13:03:07 2017

@author: Thomas
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Mar 12 21:41:18 2017
@author: Thomas
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Mar 09 23:27:57 2017
@author: Thomas
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Mar 09 10:29:59 2017
@author: Thomas
"""
#

import numpy as np
# Preprocessor to remove the test (only needed once)
#preprocessEColi('F:\\ecoli\\ecoli.data','ecoli_proc.data')

with open("F:\\Housing\\housing_proc.data") as f:
    ncols = len(f.readline().split())
housing = np.loadtxt('housing_proc.data',dtype='S').astype(np.float)
#Attempting to normalize data
housing[:,:14] = housing[:,:14]-housing[:,:14].mean(axis=0)
imax = np.concatenate((housing.max(axis=0)*np.ones((1,14)),np.abs(housing.min(axis=0)*np.ones((1,14)))),axis=0).max(axis=0)
housing[:,:14] = housing[:,:14]/imax[:14]
#dataset[:,:n-1] = dataset[:,:n-1]-dataset[:,:n-1].mean(axis=0)
#imax= np.concatenate((dataset.max(axis=0)*np.ones((1,n-1))), np.abs(dataset.min(axis=0)*np.ones((1,n-1))),axis=0).max(axis=0)
#dataset[:,:n-1] = dataset[:,:n-1]/imax[:n-1]
#print ("Here comes some values\n")
#Random Matrix.
#x = np.linspace(0,14,506).reshape((506,14))
x = np.random.random((506,14))
#print(x)
#x = (x - x.mean(axis=0))/x.var(axis=0)  #normalize
#print(x)
#housing = (housing - housing.mean(axis=0))/housing.var(axis=0)
print(housing)
# Split into training, validation, and test sets


# Randomly order the data
order = range(np.shape(housing)[0])
np.random.shuffle(order)
print("The old housing\n")
print(housing)
newHousing = housing[order,:]
np.random.shuffle(housing)
print("The new housing\n")
print(housing)

train = housing[0:250,0:13]       #49.4 % train set
traint = housing[0:250,0:14] # 
valid = housing[250:377,0:13]
validt = housing[250:377,0:14]
test = housing[377:505, 0:13]
testt = housing[377:505,0:14]



# Train the network
import mlp
net = mlp.mlp(train,traint,20,outtype='linear')
#net.earlystopping(train,traint,valid,validt,0.001)
net.mlptrain(train,traint,0.001,250)
net.earlystopping(train,traint,valid,validt,0.001)
net.confmat(test,testt)
#print train.max(axis=0), train.min(axis=0)
