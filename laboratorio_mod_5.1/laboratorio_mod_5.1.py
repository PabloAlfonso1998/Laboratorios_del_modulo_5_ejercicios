#Laboratorio 5_1
import cv2
import urllib.request
import os
from matplotlib import pyplot as plt
#%matplotlib inline
import numpy as np
from pylab import rcParams

miniature-schnauzer-puppies_url = ""
miniature-schnauzer-puppies_filename = "miniature-schnauzer-puppies.jpeg"
urllib.request.urlretrieve(miniature-schnauzer-puppies_url, miniature-schnauzer-puppies_filename)

miniature-schnauzer-puppies = cv2.imread(miniature-schnauzer-puppies-_filename)
#plt.axis("off")
#img_corrected = cv2.cvtColor(miniature-schnauzer-puppies, cv2.COLOR_BGR2RGB)
#plt.imshow(img_corrected)
#plt.show()


gray_miniature-schnauzer-puppies = cv2.cvtColor(miniature-schnauzer-puppies, cv2.COLOR_BGR2GRAY)
#plt.imshow(gray_miniature-schnauzer-puppies, cmap = "gray")
#plt.axis("off") #remove axes ticks
#plt.title('Grayscale Minions')
#plt.show()


#rcParams['figure.figsize'] = 8,4

#plt.hist(gray_miniature-schnauzer-puppies.ravel(),256,[0,256])
#plt.title('Histogram of Grayscale miniature-schnauzer-puppies.jpg')
#plt.show()


rcParams['figure.figsize'] = 8, 4

color = ('b', 'g', 'r')
for i, col in enumerate(color):
    histr = cv2.calcHist([miniature-schnauzer-puppies],[i], None, [256], [0, 256])
    plt.plot(histr, color=col)
    plt.xlim([0, 256])
plt.show()
