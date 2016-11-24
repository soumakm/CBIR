# -*- coding: utf-8 -*-
"""
Created on Sun Sep 18 08:53:22 2016

@author: Soumak
"""
from PIL import Image
import os
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

TOTAL_NO = 1000
NO_OF_CLASS = 10
NO_EACH_CLASS = 100

   

#feature_list is the list of tuples 
# It will hold class information and feature descriptor for each image            
feature_list = []

for i in range(TOTAL_NO):
    fname = 'image.orig/'+ str(i) + '.jpg'
    img = cv2.imread(fname)
    sift = cv2.SIFT()
    kp1, des1 = sift.detectAndCompute(img,None)
    feature_list.append((i//NO_EACH_CLASS, des1))

#declare a list which will hold a tuple of average precision and rank
#for each query image    
pr_list = []   
for k in range(800,900):         
    ##query image
    des1 = feature_list[k][1]
    ##compute feature match 
    #
    match_list = []
    for i in range(TOTAL_NO):
        des2 = feature_list[i][1]
        bf = cv2.BFMatcher()
         # Match descriptors.
        matches = bf.match(des1,des2)
# Sort them in the order of their distance.
        matches = sorted(matches, key = lambda x:x.distance)
        M_SUM = matches[0].distance
    #store the category and Histogram Intersection value    
        match_list.append((feature_list[i][0],M_SUM))   
    #sort match
    #smaller match is best    
    sorted_match = sorted(match_list, key=lambda x: x[1])
    
    #calculate precision & rank
    p_av = 0.0
    ml = 0
    for i in range(0,100):
        #if the image is relevant, increment ml
        if sorted_match[i][0] == k//NO_EACH_CLASS:
            ml += 1.0
        p_av += ml/(i+1)
        
#divide by 100 to get average precision and rank
    p_av = p_av/100.0 
    pr_list.append(p_av)