# -*- coding: utf-8 -*-
"""
Created on Thu Mar 09 10:29:59 2017
@author: Thomas
"""

# Code from Chapter 4 of Machine Learning: An Algorithmic Perspective (2nd Edition)
# by Stephen Marsland (http://stephenmonika.net)

# You are free to use, change, or redistribute the code in any way you wish for
# non-commercial purposes, but please maintain the name of the original author.
# This code comes with no warranty of any kind.

# Stephen Marsland, 2008, 2014

# The iris classification example

def preprocessIris(infile,outfile):
    #The number has a sequence name. This should be removed.
    #Need to change outputs.
    #stext1 = 'cp'
    #stext2 = 'im'
    #stext3 = 'pp'
    #stext4 = 'imU'
    #stext5 = 'om'
    #stext6 = 'omL'
    #stext7 = 'imL'
    #stext8 = 'imS'
    
    stext1 = 'cp'
    stext2 = 'imL'
    stext3 = 'imS'
    stext4 = 'imU'
    stext5 = 'omL'
    stext6 = 'om'
    stext7 = 'im'
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
        elif s.find(stext2)>-1:
            oid.write(s.replace(stext2, rtext2))
        elif s.find(stext3)>-1:
            oid.write(s.replace(stext3, rtext3))
        elif s.find(stext4)>-1:
            oid.write(s.replace(stext4, rtext4))
        elif s.find(stext5)>-1:
            oid.write(s.replace(stext5, rtext5))
        elif s.find(stext6)>-1:
            oid.write(s.replace(stext6, rtext6))
        elif s.find(stext7)>-1:
            oid.write(s.replace(stext7, rtext7))
        elif s.find(stext8)>-1:
            oid.write(s.replace(stext8, rtext8))
    fid.close()
    oid.close()

import numpy as np
# Preprocessor to remove the test (only needed once)
preprocessIris('D:\\ecoli\\ecoli.data','ecoli_proc.data')

with open("D:\\ecoli\\ecoli.data") as f:
    ncols = len(f.readline().split())
print(ncols)
print('\n')
ecoli = np.loadtxt('ecoli_proc.data',dtype='S', usecols = range(1,ncols)).astype(np.float)

print(ecoli)
ecoli[:,:8] = ecoli[:,:8]-ecoli[:,:8].mean(axis=0)
imax = np.concatenate((ecoli.max(axis=0)*np.ones((1,8)),np.abs(ecoli.min(axis=0)*np.ones((1,8)))),axis=0).max(axis=0)
ecoli[:,:8] = ecoli[:,:8]/imax[:8]
print ("Here comes some values\n")
print ecoli[0:5,:]

# Split into training, validation, and test sets
target = np.zeros((np.shape(ecoli)[0],3));
indices = np.where(ecoli[:,7]==0) 
target[indices,0] = 1
indices = np.where(ecoli[:,7]==1)
target[indices,1] = 1
indices = np.where(ecoli[:,7]==2)
target[indices,2] = 1
indices = np.where(ecoli[:,7]==3)
target[indices,2] = 1
indices = np.where(ecoli[:,7]==4)
target[indices,2] = 1
indices = np.where(ecoli[:,7]==5)
target[indices,2] = 1
indices = np.where(ecoli[:,7]==6)
target[indices,2] = 1
indices = np.where(ecoli[:,7]==7)
target[indices,2] = 1
      
# Randomly order the data
order = range(np.shape(ecoli)[0])
np.random.shuffle(order)
ecoli = ecoli[order,:]
target = target[order,:]

train = ecoli[::2,0:7] #Want every second element, from colums 0,1,2,3
print("Shape of train is ")
print(train.shape)
print('\n')
traint = target[::2]    #Want to target every second element period.
print("Shape of traint is ")
print(traint.shape)
print('\n')
valid = ecoli[1::4,0:7]  #Starting from row 1 get me the forth element, from columns 0,1,2,3
print("Shape of valid is ")
print(valid.shape)
print('\n')
validt = target[1::7]   #From row 1, get me every 4th element. this has 84 elements
print("Shape of target is ")
print(target.shape)
print('\n')
test = ecoli[3::4,0:7]   #From the third row on, get me every 4th element in columns 0,1,2,3
print("Shape of test is ")
print(test.shape)
print('\n')
testt = target[3::7]    #From the third row on, get me every 3rd element in columns 3,4
print("Shape of testt is ")
print(testt.shape)
print('\n')
#print train.max(axis=0), train.min(axis=0)

# Train the network
import mlp
net = mlp.mlp(train,traint,5,outtype='logistic')
net.earlystopping(train,traint,valid,validt,0.1)
net.confmat(test,testt)
