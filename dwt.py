# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 11:40:07 2018

@author: LocalAdmin
"""

import pywt
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

def dwt(a):
    [ca,cd] = pywt.dwt(a,'haar')
    return ca,cd


def idwt(ca,cd):
    ori = pywt.idwt(ca,cd,'haar')
    return ori

if __name__ == '__main__':
    datatest_5 = sio.loadmat('E:\\北航\\研究生\\2012\\SRSVD\\Data\\20040301_144_2.mat')['data']   #121*2016
    datatest_10 = datatest_5[:,::2]   #121*1008
    datatest_20 = datatest_5[:,::4]   #121*504
    datatest_40 = datatest_5[:,::8]   #121*252
    datatest_60 = datatest_5[:,::12]  #121*168
    dataset = datatest_5[0,:] / 1000;
    cA, cD = dwt(dataset)
    print(cA)
    print(cD)
    recover = idwt(cA, cD)
    print(recover)
    plt.figure(figsize=(10,8),dpi=80)
    plt.subplot(3,1,1)
    plt.plot([2*(i+1)  for i in range(len(cA))],cA)
    plt.title('cA component')
    plt.subplot(3,1,2)
    plt.plot([2*(i+1)  for i in range(len(cD))],cD)
    plt.title('cD component')
    plt.subplot(3,1,3)
    plt.plot([(i+1)  for i in range(len(dataset))],dataset)
    plt.title('Original data')
    plt.show()

