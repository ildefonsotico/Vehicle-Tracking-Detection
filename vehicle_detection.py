
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
import pickle
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from utilities import extract_features, get_hog_features, find_cars, find_cars_2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage import img_as_float

print('Loading Classifier parameters...')
# load a pe-trained svc model from a serialized (pickle) file
dist_pickle = pickle.load( open("svc_pickle.p", "rb" ) )

# get attributes of our svc object
svc = dist_pickle["svc"]
X_scaler = dist_pickle["scaler"]
orient = dist_pickle["orient"]
pix_per_cell = dist_pickle["pix_per_cell"]
cell_per_block = dist_pickle["cell_per_block"]
spatial_size = dist_pickle["spatial_size"]
hist_bins = dist_pickle["hist_bins"]

img = mpimg.imread('test_images/test_image.jpg')


print('Spatial Size ', spatial_size)
print('Hist Bins ', hist_bins)
print('Pix per cell', pix_per_cell)
print('Cell per Block ', cell_per_block)
ystart = 400
ystop = 656
scale = 1.5

out_img, xpto = find_cars_2(img, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size,
                    hist_bins)

out_img = out_img.astype(np.float32) * 255

plt.imshow(out_img)

#print(out_img)
plt.savefig('car_found.jpg')