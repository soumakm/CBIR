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
import math

TOTAL_NO = 1000
NO_OF_CLASS = 10
NO_EACH_CLASS = 100

def process_image(imagename,resultname,params="--edge-thresh 10 --peak-thresh 5"):
	""" process an image and save the results in a file"""

	if imagename[-3:] != 'pgm':
		#create a pgm file
         try:
             im = Image.open('sift_image/'+imagename[10:-4]+'tmp.pgm')
         except:    
             im = Image.open(imagename).convert('L')
             im.save('sift_image/'+imagename[10:-4]+'tmp.pgm')
         imagename = 'sift_image/'+imagename[10:-4]+'tmp.pgm'

	cmmd = str("sift "+imagename+" --output="+resultname+
				" "+params)
	os.system(cmmd)
#	print 'processed', imagename, 'to', resultname
def read_features_from_file(filename):
	""" read feature properties and return in matrix form"""
	f = np.loadtxt(filename)
	return f[:,:4],f[:,4:] # feature locations, descriptors

def match(desc1,desc2):
	""" for each descriptor in the first image, 
		select its match in the second image.
		input: desc1 (descriptors for the first image), 
		desc2 (same for second image). """
	
	desc1 = np.array([d/np.linalg.norm(d) for d in desc1])
	desc2 = np.array([d/np.linalg.norm(d) for d in desc2])
	
	dist_ratio = 0.6
	desc1_size = desc1.shape
	
	matchscores = np.zeros((desc1_size[0],1))
	desc2t = desc2.T #precompute matrix transpose
	for i in range(desc1_size[0]):
		dotprods = np.dot(desc1[i,:],desc2t) #vector of dot products
		dotprods = 0.9999*dotprods
		#inverse cosine and sort, return index for features in second image
		indx = np.argsort(np.arccos(dotprods))
		#check if nearest neighbor has angle less than dist_ratio times 2nd
		if np.arccos(dotprods)[indx[0]] < dist_ratio * np.arccos(dotprods)[indx[1]]:
                 matchscores[i] = int(indx[0])
	return matchscores
 
def match_twosided(desc1,desc2):
	""" two-sided symmetric version of match(). """
	
	matches_12 = match(desc1,desc2)
	matches_21 = match(desc2,desc1)
	
	ndx_12 = matches_12.nonzero()[0]
	
	#remove matches that are not symmetric
	for n in ndx_12:
		if matches_21[int(matches_12[n])] != n:
			matches_12[n] = 0
	
	return matches_12       

#feature_list is the list of tuples 
# It will hold class information and feature descriptor for each image            
feature_list = []

for i in range(TOTAL_NO):
    fname = 'image.orig/'+ str(i) + '.jpg'
    sname = 'sift_image/'+ str(i) + '.sift'
    process_image(fname,sname)
    l,d = read_features_from_file(sname)
    feature_list.append((i//NO_EACH_CLASS, d))

#declare a list which will hold a tuple of average precision and rank
#for each query image    
pr_list = []   
for k in range(TOTAL_NO):         
    ##query image
    d = feature_list[k][1]
    ##compute feature match 
    #
    match_list = []
    for i in range(TOTAL_NO):
        m = match(d,feature_list[i][1])
        M_SUM = sum(m)
    #store the category and Histogram Intersection value    
        match_list.append((feature_list[i][0],M_SUM))   
    #sort match
    sorted_match = sorted(match_list, key=lambda x: x[1], reverse=True)
    
    #calculate precision & rank
    p_av = 0.0
    ml = 0
    r = 0
    count = 0
    for i in range(0,TOTAL_NO):
        #if the image is relevant, increment ml
        if sorted_match[i][0] == k//NO_EACH_CLASS:
            if i < 100:
                ml += 1.0
            r += float(i)
            count += 1
            if count == 100:
                break
        if i < 100:   
            p_av += ml/(i+1)
        
#divide by 100 to get average precision and rank
    p_av = p_av/100.0 
    r_av = r/100.0
    pr_list.append((p_av, r_av))

#calculate average of average precision and average rank for each class
class_pr = []
for i in range(NO_OF_CLASS):
    class_pr.append(np.sum(pr_list[i*NO_EACH_CLASS:(i+1)*NO_EACH_CLASS], axis=0)/NO_EACH_CLASS)
    print( class_pr[i])
                 