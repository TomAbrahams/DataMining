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
def preprocessEColi(infile,outfile):
    #The number has a sequence name. This should be removed.
    #Need to change outputs.
    stext1 = 'cp' #will
    stext2 = 'im' #last
    stext3 = 'imS' 
    stext4 = 'imL'
    stext5 = 'imU' 
    stext6 = 'om'#2nd last
    stext7 = 'omL'
    stext8 = 'pp'
    
    
    rtext1 = '0'
    rtext2 = '1'
    rtext3 = '2'
    rtext4 = '3'
    rtext5 = '4'
    rtext6 = '5'
    rtext7 = '6'
    rtext8 = '7'

    fid = open(infile,"r")
    oid = open(outfile,"w")

    for s in fid:
        if s.find(stext1)>-1:
            oid.write(s.replace(stext1, rtext1))
        elif s.find(stext3)>-1:
            oid.write(s.replace(stext3, rtext3))
        elif s.find(stext4)>-1:
            oid.write(s.replace(stext4, rtext4))
        elif s.find(stext5)>-1:
            oid.write(s.replace(stext5, rtext5))
        elif s.find(stext7)>-1:
            oid.write(s.replace(stext7, rtext7))
        elif s.find(stext8)>-1:
            oid.write(s.replace(stext8, rtext8))
        elif s.find(stext2)>-1:
            oid.write(s.replace(stext2, rtext2))
        elif s.find(stext6)>-1:
            oid.write(s.replace(stext6, rtext6))
        
    fid.close()
    oid.close()

import numpy as np
import urllib

#This is added to get the data from UCI directly.
#Get the data from uci.
urllib.urlretrieve("https://archive.ics.uci.edu/ml/machine-learning-databases/ecoli/ecoli.data", "ecoliUCI.data")

#Preprocessor to remove the test (only needed once)
preprocessEColi('ecoliUCI.data','ecoli_proc.data')     

with open("ecoli_proc.data") as f:
    ncols = len(f.readline().split())

# Get the text file


ecoli = np.loadtxt('ecoli_proc.data',dtype='S', usecols = range(1,ncols)).astype(np.float)
# Normalization of DATA
ecoli[:,:7] = ecoli[:,:7]-ecoli[:,:7].mean(axis=0)
imax = np.concatenate((ecoli.max(axis=0)*np.ones((1,8)),np.abs(ecoli.min(axis=0)*np.ones((1,8)))),axis=0).max(axis=0)
#Normalize the last item.
ecoli[:,:7] = ecoli[:,:7]/imax[:7]



# Split into training, validation, and test sets

target = np.zeros((np.shape(ecoli)[0],8));
            
print(np.shape(ecoli)[0],8)
indices = np.where(ecoli[:,7]==0) 
target[indices,0] = 1
indices = np.where(ecoli[:,7]==1)
target[indices,1] = 1
indices = np.where(ecoli[:,7]==2)
target[indices,2] = 1
indices = np.where(ecoli[:,7]==3)
target[indices,3] = 1
indices = np.where(ecoli[:,7]==4)
target[indices,4] = 1
indices = np.where(ecoli[:,7]==5)
target[indices,5] = 1
indices = np.where(ecoli[:,7]==6)
target[indices,6] = 1
indices = np.where(ecoli[:,7]==7)
target[indices,7] = 1

#This is to shuffle the items.
order = range(np.shape(ecoli)[0])
np.random.shuffle(order)
ecoli = ecoli[order,:]
target = target[order,:]

train = ecoli[0:168,0:7]    # Gets all possible inputs from row 0 to 167.
traint = target[0:168]      # Gets the target outputs from row 0 to 167.
valid = ecoli[168:252,0:7]  # Gets the valid inputs from row 168 to 251 
validt = target[168:252]    # Gets the target outpus from row 168 to 251
test = ecoli[252:336, 0:7]  # Obtain the test set
testt = target[252:336]     # Obtain the target test set for confusion matrix.

# Train the network
import mlp
net = mlp.mlp(train,traint,21,outtype='softmax')
net.mlptrain(train,traint,0.40, 350)
net.earlystopping(train,traint,valid,validt,0.20)
net.confmat(test,testt)
#print train.max(axis=0), train.min(axis=0)

