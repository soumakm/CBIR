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
from scipy.ndimage import filters

TOTAL_NO = 1000
NO_OF_CLASS = 10
NO_EACH_CLASS = 100
SIGMA=0.5

def compute_hist(img):
    """
    input = image with 3 color channels
    returns : 3 histogram as 3 lists
    """
    row = img.shape[0]
    col = img.shape[1]
    
    red = img[:,:,0]
    green = img[:,:,1]
    blue = img[:,:,2]    
    
    hist_r=[0]*256 
    hist_g=[0]*256
    hist_b=[0]*256 
    
    for i in range(row):
        for j in range(col):
            p=red[i,j]
            hist_r[p] += 1.0/(row*col)
            p=green[i,j]
            hist_g[p] += 1.0/(row*col)
            p=blue[i,j]
            hist_b[p] += 1.0/(row*col)
    
    return  hist_r, hist_g, hist_b          

#hist_list is the list of tuples 
# It will hold class information and histtograms for 3 channels            
hist_list = []

for i in range(TOTAL_NO):
    fname = 'image.orig/'+ str(i) + '.jpg'
    img_orig = plt.imread(fname)
    #use filtered image
    img = filters.gaussian_filter(img_orig,SIGMA)
    
    hist_r, hist_g, hist_b = compute_hist(img)
    hist_list.append((i//NO_EACH_CLASS, hist_r, hist_g, hist_b))

#declare a list which will hold a tuple of average precision and rank
#for each query image    
pr_list = []   
        
##query image
fname = 'image.orig/887.jpg'
img_orig = plt.imread(fname)
#use filtered image
img = filters.gaussian_filter(img_orig,SIGMA)
hist_r, hist_g, hist_b = compute_hist(img)

##compute histogram intersection 
match = []
for i in range(TOTAL_NO):
    HI_r =0
    HI_g =0
    HI_b =0
    for j in range(256):
        HI_r += min(hist_r[j],hist_list[i][1][j]) 
        HI_g += min(hist_g[j],hist_list[i][2][j]) 
        HI_b += min(hist_b[j],hist_list[i][3][j])
    HI = HI_r + HI_g + HI_b
#store the category and Histogram Intersection value    
    match.append((hist_list[i][0],HI))   
#sort match
sorted_match = sorted(match, key=lambda x: x[1], reverse=True)

            
#calculate precision & recall
p = []
r = []
ml=0.0
for i in range(1,1000):
    if sorted_match[i-1][0] == 8:
        ml += 1.0
    p.append(ml/i)
    r.append(ml/100)
    
plt.plot(p,r)   
plt.xlim([0,1])    
plt.ylim([0,1])
    
        