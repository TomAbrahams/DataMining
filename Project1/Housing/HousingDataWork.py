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
import pylab as pl
import numpy as np
# Preprocessor to remove the test (only needed once)
#preprocessEColi('F:\\ecoli\\ecoli.data','ecoli_proc.data')

with open("F:\\Housing\\housing_proc.data") as f:
    ncols = len(f.readline().split())
housing = np.loadtxt('housing_proc.data',dtype='S').astype(np.float)
#Attempting to normalize data
print("Do you want to fly?")
print(housing)
housing[:,:13] = housing[:,:13]-housing[:,:13].mean(axis=0)
#housingStd = housing.std(axis=0)
#housing[:,:13] = housing[:,:13]/housingStd[:13]

#True way
imax = np.concatenate((housing.max(axis=0)*np.ones((1,14)),np.abs(housing.min(axis=0)*np.ones((1,14)))),axis=0).max(axis=0)
housing[:,:13] = housing[:,:13]/imax[:13]

#dataset[:,:n-1] = dataset[:,:n-1]-dataset[:,:n-1].mean(axis=0)
#imax= np.concatenate((dataset.max(axis=0)*np.ones((1,n-1))), np.abs(dataset.min(axis=0)*np.ones((1,n-1))),axis=0).max(axis=0)
#dataset[:,:n-1] = dataset[:,:n-1]/imax[:n-1]
#print ("Here comes some values\n")
#Random Matrix.
#x = np.linspace(0,1,506).reshape((506,1))
#x = (x-0.5)*2
x = np.random.uniform(low = 0.0, high = 1.0, size =(506,14))
print("This is x")
print(x)
#xstd = x.std(x)
x = (x - x.mean(axis=0))/x.var(axis=0)  #normalize
#print(x)
#housing = (housing - housing.mean(axis=0))/housing.var(axis=0)
print("This is housing normalized")
print(housing)
# Split into training, validation, and test sets


# Randomly order the data
order = range(np.shape(housing)[0])
np.random.shuffle(order)
#order = range(np.shape(iris)[0])
#np.random.shuffle(order)
#iris = iris[order,:]
#target = target[order,:]
print("The old housing non randomized\n")
print(housing)
t = housing
np.random.shuffle(t)
print("The new housing randomized\n")
print(housing)

train = x[0:250,0:13]       #49.4 % train set
traint = t[0:250,0:14] # 
valid = x[250:377,0:13]
validt = t[250:377,0:14]
test = x[377:505, 0:13]
testt = t[377:505,0:14]

#plot the data
v = x[0:1]
g = t[0:1]
pl.plot(v,g,'o')
pl.xlabel('x')
pl.ylabel('t')

# Train the network
import mlp
net = mlp.mlp(train,traint,20,outtype='linear')
#net.earlystopping(train,traint,valid,validt,0.001)
net.mlptrain(train,traint,0.001,250)
net.earlystopping(train,traint,valid,validt,0.001)
net.confmat(test,testt)
print train.max(axis=0), train.min(axis=0)

# Test out different sizes of network
#count = 0
#out = np.zeros((10,7))
#for nnodes in [1,2,3,5,10,25,50]:
 #   for i in range(10):
  #      net = mlp.mlp(train,traint,nnodes,outtype='linear')
   #     out[i,count] = net.earlystopping(train,traint,valid,validt,0.25)
    #count += 1
    
test = np.concatenate((test,-np.ones((np.shape(test)[0],1))),axis=1)


pl.show()
