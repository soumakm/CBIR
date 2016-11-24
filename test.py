from PIL import Image
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

def process_image(imagename,resultname,params="--edge-thresh 10 --peak-thresh 5"):
	""" process an image and save the results in a file"""

	if imagename[-3:] != 'pgm':
		#create a pgm file
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
	
	desc1 = array([d/linalg.norm(d) for d in desc1])
	desc2 = array([d/linalg.norm(d) for d in desc2])
	
	dist_ratio = 0.6
	desc1_size = desc1.shape
	
	matchscores = zeros((desc1_size[0],1))
	desc2t = desc2.T #precompute matrix transpose
	for i in range(desc1_size[0]):
		dotprods = dot(desc1[i,:],desc2t) #vector of dot products
		dotprods = 0.9999*dotprods
		#inverse cosine and sort, return index for features in second image
		indx = argsort(arccos(dotprods))
		
		#check if nearest neighbor has angle less than dist_ratio times 2nd
#		if arccos(dotprods)[indx[0]] < dist_ratio * arccos(dotprods)[indx[1]]:
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
 
#process_image('image.orig/0.jpg','sift_image/0.sift')
#l,d = read_features_from_file('sift_image/0.sift')
#
#process_image('image.orig/1.jpg','sift_image/1.sift')
#l2,d2 = read_features_from_file('sift_image/1.sift')
img1 = cv2.imread('image.orig/0.jpg') 
img2 = cv2.imread('image.orig/1.jpg') 
sift = cv2.SIFT()
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)
#m = match_twosided(d,d2)
bf = cv2.BFMatcher()
#matches = bf.knnMatch(des1,des2, k=2)
matches = bf.match(des1,des2)
matches = sorted(matches, key = lambda x:x.distance)
 # Apply ratio test
#good = []
#for m,n in matches:
#    if m.distance < 0.75*n.distance:
#        good.append([m])
# cv2.drawMatchesKnn expects list of lists as matches.
#img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,flags=2)
#plt.imshow(img3),plt.show()

