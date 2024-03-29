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
    
    #stext1 = 'cp'
    #stext2 = 'imL'
    #stext3 = 'imS'
    #stext4 = 'imU'
    #stext5 = 'omL'
    #stext6 = 'om'
    #stext7 = 'im'
    #stext8 = 'pp'
    
    
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
# Preprocessor to remove the test (only needed once)
#preprocessEColi('F:\\ecoli\\ecoli.data','ecoli_proc.data')

with open("F:\\ecoli\\ecoli_proc.data") as f:
    ncols = len(f.readline().split())
#print(ncols)
#print('\n')
ecoli = np.loadtxt('ecoli_proc.data',dtype='S', usecols = range(1,ncols)).astype(np.float)
#print(ecoli.shape)
#print(ecoli)
ecoli[:,:7] = ecoli[:,:7]-ecoli[:,:7].mean(axis=0)
imax = np.concatenate((ecoli.max(axis=0)*np.ones((1,8)),np.abs(ecoli.min(axis=0)*np.ones((1,8)))),axis=0).max(axis=0)
#print("imax size")
#print(imax.shape)
#print('\n')
#print(ecoli[:,:7].shape)
#print('\n')
ecoli[:,:7] = ecoli[:,:7]/imax[:7]
#dataset[:,:n-1] = dataset[:,:n-1]-dataset[:,:n-1].mean(axis=0)
#imax= np.concatenate((dataset.max(axis=0)*np.ones((1,n-1))), np.abs(dataset.min(axis=0)*np.ones((1,n-1))),axis=0).max(axis=0)
#dataset[:,:n-1] = dataset[:,:n-1]/imax[:n-1]
#print ("Here comes some values\n")
print ecoli[0:8,:]


# Split into training, validation, and test sets

#print("The np.shape(iris)[0]")
#print(np.shape(ecoli)[0])
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
#print("Indices")
#print(indices)
#print("target")
#print(target)
#print(target.size)
# Randomly order the data
order = range(np.shape(ecoli)[0])
np.random.shuffle(order)

ecoli = ecoli[order,:]
target = target[order,:]
#print("Ecoli \n")
#print(ecoli)

train = ecoli[0:168,0:7] # Want every second element, from colums 0,1,2,3
#print("Shape of train is ")
#print(train.shape)
#print('\n')
traint = target[0:168]    # Want to target every second element period.
#print("Shape of traint is ")
#print(traint.shape)
#print('\n')
valid = ecoli[168:252,0:7]  # Starting from row 1 get me the forth element, from columns 0,1,2,3 so the 5th item
#print("output of valid is: \n")
#print(valid.shape)
#print("\nShape of valid is ")
#print(valid.shape)
#print('\n')
validt = target[168:252]   # From row 1, get me every 4th element. this has 84 elements so the 5th item
#print("Shape of validt is ")
#print(validt.shape)
#print('\n')
test = ecoli[252:336, 0:7]   # From the third row on, get me every 4th element in columns 0,1,2,3 so the 7th item
#print("Shape of test is ")
#print(test.shape)
#print('\n')
testt = target[252:336]    # From the third row on, get me every 4th element 7th.
#print("Shape of testt is ")
#print(testt.shape)
#print('\n')
#print train.max(axis=0), train.min(axis=0)

# Train the network
import mlp
net = mlp.mlp(train,traint,8,outtype='softmax')
net.earlystopping(train,traint,valid,validt,0.1)
net.confmat(test,testt)
#print train.max(axis=0), train.min(axis=0)

