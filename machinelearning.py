
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
import pickle
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from utilities import extract_features, get_hog_features

cars = glob.glob('vehicles/**/*.png', recursive=True)
noncars = glob.glob('non-vehicles/**/*.png', recursive=True)
# Stat
print('Cars: {}'.format(len(cars)))
print('Non-Cars: {}'.format(len(noncars)))

color_space    = 'YUV'    # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient         = 9          # HOG orientations
pix_per_cell   = 8          # HOG pixels per cell
cell_per_block = 2          # HOG cells per block
hog_channel    = "ALL"          # Can be 0, 1, 2, or "ALL"
spatial_size   = (32, 32)   # Spatial binning dimensions
hist_bins      = 32         # Number of histogram bins
spatial_feat   = True       # Spatial features on or off
hist_feat      = True       # Histogram features on or off
hog_feat       = True       # HOG features on or off

car = mpimg.imread(cars[98])
noncar = mpimg.imread(noncars[89])

get_hog_features(car[:,:,0], orient, pix_per_cell, cell_per_block, True, True, True, 'car_R')
get_hog_features(car[:,:,1], orient, pix_per_cell, cell_per_block, True, True, True, 'car_G')
get_hog_features(car[:,:,2], orient, pix_per_cell, cell_per_block, True, True, True, 'car_B')
get_hog_features(noncar[:,:,0], orient, pix_per_cell, cell_per_block, True, True, True, 'non-car_R')
get_hog_features(noncar[:,:,1], orient, pix_per_cell, cell_per_block, True, True, True, 'non-car_G')
get_hog_features(noncar[:,:,2], orient, pix_per_cell, cell_per_block, True, True, True, 'non-car_B')

print('Data to be used in the project is done')
print('Extracting features about car image')
car_features = extract_features(cars, color_space=color_space,
                                   spatial_size=spatial_size, hist_bins=hist_bins,
                                   orient=orient, pix_per_cell=pix_per_cell,
                                   cell_per_block=cell_per_block,
                                   hog_channel=hog_channel, spatial_feat=spatial_feat,
                                   hist_feat=hist_feat, hog_feat=hog_feat)
print('Extracting features about the non-car images')
non_car_features = extract_features(noncars, color_space=color_space,
                                   spatial_size=spatial_size, hist_bins=hist_bins,
                                   orient=orient, pix_per_cell=pix_per_cell,
                                   cell_per_block=cell_per_block,
                                   hog_channel=hog_channel, spatial_feat=spatial_feat,
                                   hist_feat=hist_feat, hog_feat=hog_feat)

print('All features extracted')
X = np.vstack((car_features, non_car_features)).astype(np.float64)
# Split up data into randomized training and test sets
# Define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(non_car_features))))

rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=rand_state)

print('Split the dataset in two part. 20% will be used to test dataset')
print('Normalizing dataset')
# Fit a per-column scaler
X_scaler = StandardScaler().fit(X_train)
# Apply the scaler to X
scaled_X = X_scaler.transform(X_train)
scaled_X_test = X_scaler.transform(X_test)



print('Using spatial binning of:',spatial_size,
    'and', hist_bins,'histogram bins')
print('Feature vector length:', len(X_train[0]))
# Use a linear SVC
svc = LinearSVC()
# Check the training time for the SVC
print('Starting the training')
t=time.time()
svc.fit(X_train, y_train)
t2 = time.time()
print('Training done')
print(round(t2-t, 2), 'Seconds to train SVC...')
# Check the score of the SVC
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
# Check the prediction time for a single sample
t=time.time()
n_predict = 50
print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
print('For these',n_predict, 'labels: ', y_test[0:n_predict])
t2 = time.time()
print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVC')

# Save data to pickle file
dist_pickle = {}
dist_pickle["svc"] = svc
dist_pickle["scaler"] = X_scaler
dist_pickle["orient"] = orient
dist_pickle["pix_per_cell"] = pix_per_cell
dist_pickle["cell_per_block"] = cell_per_block
dist_pickle["spatial_size"] = spatial_size
dist_pickle["hist_bins"] = hist_bins
pickle.dump(dist_pickle, open("svc_pickle.p", 'wb') )


# parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
# svr = svm.SVC()
# clf = grid_search.GridSearchCV(svr, parameters)
# clf.fit(iris.data, iris.target)