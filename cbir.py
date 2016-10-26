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

TOTAL_NO = 1000
NO_OF_CLASS = 10
NO_EACH_CLASS = 100

#hist_r=[0]*256 
#hist_g=[0]*256
#hist_b=[0]*256 

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
    img = plt.imread(fname)
    
    hist_r, hist_g, hist_b = compute_hist(img)
    hist_list.append((i//NO_EACH_CLASS, hist_r, hist_g, hist_b))

#declare a list which will hold a tuple of average precision and rank
#for each query image    
pr_list = []   
new_hist_list = [] 
for k in range(TOTAL_NO):         
    ##query image
    fname = 'image.orig/'+ str(k) + '.jpg'
    img = plt.imread(fname)
    hist_r, hist_g, hist_b = compute_hist(img)
    
    #copy hist_list
    new_hist_list = hist_list[:]
    #remove query image from dataset
   # new_hist_list.remove((k//NO_EACH_CLASS, hist_r, hist_g, hist_b))
    ##compute histogram intersection 
    #
    match = []
    for i in range(TOTAL_NO):
        HI_r =0
        HI_g =0
        HI_b =0
        for j in range(256):
            HI_r += min(hist_r[j],new_hist_list[i][1][j]) 
            HI_g += min(hist_g[j],new_hist_list[i][2][j]) 
            HI_b += min(hist_b[j],new_hist_list[i][3][j])
        HI = HI_r + HI_g + HI_b
    #store the category and Histogram Intersection value    
        match.append((new_hist_list[i][0],HI))   
        
    #sort match
    sorted_match = sorted(match, key=lambda x: x[1], reverse=True)
    
    #calculate precision & rank
    p_av = 0.0
    r_av = 0.0
    

    for i in range(1,101):
        ml=0
        r = 0
        for j in range(0,i):
            #if the image is relevant, increment ml
            if sorted_match[j][0] == k//NO_EACH_CLASS:
                ml += 1.0
                r += float(j+1)
        p_av += ml/i
        r_av += r 
#divide by 100 to get average precision and rank
    p_av = p_av/100.0    
    r_av = r_av/100.0
    
    pr_list.append((p_av, r_av))

#calculate average of average precision and average rank for each class
class_pr = []
for i in range(NO_OF_CLASS):
    class_pr.append(np.sum(pr_list[i*NO_EACH_CLASS:(i+1)*NO_EACH_CLASS], axis=0)/NO_EACH_CLASS)
    print class_pr[i]
            
#plt.plot(p_av,r_av)   
#plt.xlim([0,1])    
#plt.ylim([0,1])
    
        