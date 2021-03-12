import urllib.request
import cv2
import os
import math
import matplotlib.pyplot as plt
#%matplotlib inline
from sklearn.cluster import KMeans
import numpy as np

## Pez ##
fish_image_url = "http://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/CV0101/Dataset/fish.png"
# downloads file as "fish.png"
urllib.request.urlretrieve(fish_image_url, "fish.png")
im2 = cv2.imread("fish.png")
fish_im_corrected = cv2.cvtColor(im2, cv2.COLOR_BGR2RGB)
plt.axis('off')
plt.imshow(fish_im_corrected)
print("Original size of fish image is: {} Kilo Bytes".format(str(math.ceil((os.stat('fish.png').st_size)/1000))))

num_rows_fish = im2.shape[0]
num_cols_fish = im2.shape[1]
transform_fish_image_for_KMeans = im2.reshape(num_rows_fish * num_cols_fish, 3)

kmeans_fish = KMeans(n_clusters=8)
kmeans_fish.fit(transform_fish_image_for_KMeans)
cluster_centroids_fish = np.asarray(kmeans_fish.cluster_centers_,dtype=np.uint8)

labels_fish = np.asarray(kmeans_fish.labels_,dtype=np.uint8 )
labels_fish = labels_fish.reshape(num_rows_fish,num_cols_fish)

compressed_image = np.ones((num_rows, num_cols, 3), dtype=np.uint8)
for r in range(num_rows):
    for c in range(num_cols):
        compressed_image[r, c, :] = cluster_centroids[labels[r, c], :]

cv2.imwrite("compressed_fish.png", compressed_image)
compressed_fish_im = cv2.imread("compressed_fish.png")
compressed_fish_im_corrected = cv2.cvtColor(
    compressed_fish_im, cv2.COLOR_BGR2RGB)
plt.axis('off')
plt.imshow(compressed_fish_im_corrected)
print("Compressed size of fish's image is: {} Kilo Bytes".format(str(math.ceil((os.stat('compressed_fish.png').st_size)/1000))))

## Mariposa ##
butterfly_image_url = "http://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/CV0101/Dataset/butterfly.png"
# downloads file as "butterfly.png"
urllib.request.urlretrieve(butterfly_image_url, "butterfly.png")
im3 = cv2.imread("butterfly.png")
butterfly_im_corrected = cv2.cvtColor(im3, cv2.COLOR_BGR2RGB)
plt.axis('off')
plt.imshow(butterfly_im_corrected)
print("Original size of butterfly image is: {} Kilo Bytes".format(str(math.ceil((os.stat('butterfly.png').st_size)/1000))))

num_rows_butterfly = im2.shape[0]
num_cols_butterfly = im2.shape[1]
transform_butterfly_image_for_KMeans = im2.reshape(num_rows_butterfly * num_cols_fish, 3)

kmeans = KMeans(n_clusters=8)
kmeans.fit(transform_image_for_KMeans)

cluster_centroids = np.asarray(kmeans.cluster_centers_, dtype=np.uint8)

labels = np.asarray(kmeans.labels_, dtype=np.uint8)
labels = labels.reshape(num_rows, num_cols)

compressed_image = np.ones((num_rows, num_cols, 3), dtype=np.uint8)
for r in range(num_rows):
    for c in range(num_cols):
        compressed_image[r, c, :] = cluster_centroids[labels[r, c], :]

cv2.imwrite("compressed_butterfly.png", compressed_image)
compressed_butterfly_im = cv2.imread("compressed_butterfly.png")
compressed_butterfly_im_corrected = cv2.cvtColor(
    compressed_butterfly_im, cv2.COLOR_BGR2RGB)
plt.axis('off')
plt.imshow(compressed_butterfly_im_corrected)
print("Compressed size of butterfly's image is: {} Kilo Bytes".format(str(math.ceil((os.stat('compressed_butterfly.png').st_size)/1000))))
