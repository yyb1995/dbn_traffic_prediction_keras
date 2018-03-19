# -*- coding: utf-8 -*-
"""
@author: LocalAdmin
"""

import numpy as np
import matplotlib.pyplot as plt


def generateData(sample, outputnum):
    a = np.array(sample)
    mu = np.mean(a)
    sigma_2 = np.var(a) / 2
    result = np.random.normal(loc = mu, scale = np.sqrt(sigma_2), size = outputnum)
    print('mu = %f\tsigma^2 = %f'%(mu,sigma_2))
    return mu,sigma_2,result


def drawResult(mu,sigma_2,result):
    plt.figure(figsize=(10,8),dpi=80)
    count, bins, ignored = plt.hist(result, 30, normed=True)
    plt.plot(bins, 1/(np.sqrt(2 * np.pi * sigma_2)) *np.exp( - (bins - mu)**2 / (2 * sigma_2) ),linewidth=2, color='r')
    