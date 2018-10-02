"""
dynamic pooling to reduce all matrices to the same size matrix (28x28)
with min pool
"""

import numpy as np
import pandas as pd
import tensorflow as tf

print('loading data ...')

train_mat = np.load('train_matrix.npy')
np.random.shuffle(train_mat)
train_in, labels = np.transpose(train_mat)

print('deleting empty questions...')
pos = []
count = 0
delete_mask = np.ones((train_mat.shape[0]),bool)
for i in range(len(train_mat)):
    l1,l2 = train_mat[i][0].shape
    if l1 ==0 or l2==0 or l1==1 or l2==1 or l1==2 or l2==2:
        count+=1
        pos.append(i)
        delete_mask[i] = False
print(count)
train_in = train_in[delete_mask]
labels = labels[delete_mask]


#checking if deleting worked
print('checking if deleting worked ...')
pos = []
count = 0
for i in range(len(train_in)):
    l1,l2 = train_in[i].shape
    if l1 ==0 or l2==0:
        count+=1
        pos.append(i)
    if l1 ==1 or l2==1:
        count+=1
        pos.append(i)
print(count)

print('upsampling to match row or column to 28')
npool = 28
#upsampling to increase row count above npool
for j in range(len(train_in)):
    if j%5000==0:
        print(j/len(train_in))
    l1,l2 = train_in[j].shape
    if l1<npool:
        temp = train_in[j]
        m= 0
        while temp.shape[0]<npool:
            i = np.random.randint(temp.shape[0])
            temp = np.concatenate((temp,[temp[i]]))
        train_in[j] = temp.astype(np.float32)
    if l2<npool:
        temp = train_in[j]
        temp = np.transpose(train_in[j])
        while temp.shape[0]<npool:
            i = np.random.randint(temp.shape[0])
            temp = np.concatenate((temp,[temp[i]]))
        train_in[j] = np.transpose(temp).astype(np.float32)
print(j==len(train_in)-1)

print('min pool operation on indices...')
#pooling operation
train_pool = np.ones((len(train_in),npool,npool))
for j in range(len(train_in)):
    #print(j)
    R,C = train_in[j].shape
    wr1 = int(np.floor(R/npool))
    wc1 = int(np.floor(C/npool))
    if j%5000==0:
        print(100*j/len(train_in))
    i = 0
    pool_mat = np.ones((npool*npool),float)
    k=0
    l=0
    while k <=(npool-1)*wr1:
        while l<=(npool-1)*wc1:
            if k< (npool-1)*wr1 and l<(npool-1)*wc1:
                pool_mat[i] = np.min(train_in[j][k:k+wr1,l:l+wc1])
            if k<(npool-1)*wr1 and l>= (npool-1)*wc1:
                pool_mat[i] = np.min(train_in[j][k:k+wr1,l:C])
            if k>= (npool-1)*wr1 and l<(npool-1)*wc1:
                pool_mat[i] = np.min(train_in[j][k:R,l:l+wc1])
            if k>= (npool-1)*wr1 and l>= (npool-1)*wc1:
                pool_mat[i] = np.min(train_in[j][k:R,l:C])
            l+=wc1
            i+=1
        l=0
        k+=wr1

    train_pool[j] = pool_mat.reshape((npool,npool))
print('saving train_pool matrix ...')
np.save('train_pool.npy',train_pool)
