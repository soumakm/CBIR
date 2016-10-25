# -*- coding: utf-8 -*-
"""
Created on Sun Sep 18 08:53:22 2016

@author: Soumak
"""

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import math

hist_r=[0]*256 
hist_g=[0]*256
hist_b=[0]*256 

def compute_hist(img, hist_r, hist_g, hist_b):
    row = img.shape[0]
    col = img.shape[1]
    
    red = img[:,:,0]
    green = img[:,:,1]
    blue = img[:,:,2]    
    
    
    
    for i in range(row):
        for j in range(col):
            p=red[i,j]
            hist_r[p] += 1.0/(row*col)
            p=green[i,j]
            hist_g[p] += 1.0/(row*col)
            p=blue[i,j]
            hist_b[p] += 1.0/(row*col)


hist_tuples = []


for i in range(10):
    c1_file = 'c1/' + str(i) + '.ppm'
    img = plt.imread(c1_file)
    hist_r=[0]*256 
    hist_g=[0]*256
    hist_b=[0]*256 
    compute_hist(img, hist_r, hist_g, hist_b)
    hist_tuples.append(('c1', hist_r, hist_g, hist_b))
    
    c2_file = 'c2/' + str(i) + '.ppm'
    img = plt.imread(c2_file)
    hist_r=[0]*256 
    hist_g=[0]*256
    hist_b=[0]*256  
    compute_hist(img, hist_r, hist_g, hist_b)
    hist_tuples.append(('c2', hist_r, hist_g, hist_b))
    
    c3_file = 'c3/' + str(i) + '.ppm'
    img = plt.imread(c3_file)
    hist_r=[0]*256 
    hist_g=[0]*256
    hist_b=[0]*256 
    compute_hist(img, hist_r, hist_g, hist_b)
    hist_tuples.append(('c3', hist_r, hist_g, hist_b))
    
#query image
hist_r=[0]*256 
hist_g=[0]*256
hist_b=[0]*256 
img = plt.imread('c3/1.ppm')
compute_hist(img, hist_r, hist_g, hist_b)
#remove query image from dataset
hist_tuples.remove(('c3', hist_r, hist_g, hist_b))
#compute histogram intersection 

match = []
for i in range(len(hist_tuples)):
    HI_r =0
    HI_g =0
    HI_b =0
    for j in range(256):
        HI_r += min(hist_r[j],hist_tuples[i][1][j]) 
        HI_g += min(hist_g[j],hist_tuples[i][2][j]) 
        HI_b += min(hist_b[j],hist_tuples[i][3][j])
    HI = HI_r + HI_g + HI_b
    match.append((hist_tuples[i][0],HI))   
    
#sort match
sorted_match = sorted(match, key=lambda x: x[1], reverse=True)

#calculate precision & recall
p = []
r = []

for i in range(1,30,2):
    ml=0
    for j in range(0,i):
        if sorted_match[j][0] == 'c3':
            ml += 1.0
    p.append(ml/i)
    r.append(ml/9)
    
plt.plot(p,r)   
plt.xlim([0,1])    
plt.ylim([0,1])

    