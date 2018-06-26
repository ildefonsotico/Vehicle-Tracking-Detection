## Writeup Template
### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The project consist in three files - `utilities.py`, `machinelearning.py`, `vehicle_detection.py`
The code where we extract features (including HoG) can be found in lines # 43 through 54 # of the file called `machinelearning.py`.  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![image0001](https://user-images.githubusercontent.com/19958282/41934552-67c75466-795d-11e8-9382-55003b050ed5.png)
![image59](https://user-images.githubusercontent.com/19958282/41934574-74f555b6-795d-11e8-910a-195f5be982eb.png)

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=15`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`, `hog channel=All`:
![car_hog_y](https://user-images.githubusercontent.com/19958282/41936095-6a48c148-7962-11e8-8ddf-f46f87d3f708.png)
![car_hog_cr](https://user-images.githubusercontent.com/19958282/41936214-e69ec09e-7962-11e8-8f74-8087aa3de9b5.png)
![car_hog_cb](https://user-images.githubusercontent.com/19958282/41936310-2a5a13e2-7963-11e8-9601-b90dd6b53127.png)
![non_car_hog_y](https://user-images.githubusercontent.com/19958282/41936478-ae76056e-7963-11e8-8c96-ea5834d04847.png)
![non_car_hog_cr](https://user-images.githubusercontent.com/19958282/41936487-b3a5a512-7963-11e8-8b7d-aeadf8111e5b.png)
![non_car_hog_cb](https://user-images.githubusercontent.com/19958282/41936570-e82ce3b8-7963-11e8-9a26-dd1aaab41e4f.png)

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters. I started `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`, `hog channel=0` and `color_space=HSV` but i got a lot of noise into the image. I changed the parameters a lot of time to see whether if fits good all frames.
I also changed the parameters to get more features `orientations=15`, `pixels_per_cell=(4, 4)` and `cells_per_block=(2, 2)`, `hog channel=All` and `color_space=YCrCb`. The approach got better but still had some noises in a specifics frames. I solved the issues by the windows and scale.


#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

First, I extract features from the vehicles and non-vehicles images. I used HoG features, color features (YCrCb), spatial features and also a histogram of the features. I trained a linear SVM using.

```Python
color_space    = 'YCrCb'    # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient         = 15          # HOG orientations
pix_per_cell   = 8          # HOG pixels per cell
cell_per_block = 2          # HOG cells per block
hog_channel    = "ALL"          # Can be 0, 1, 2, or "ALL"
spatial_size   = (32, 32)   # Spatial binning dimensions
hist_bins      = 32         # Number of histogram bins
spatial_feat   = True       # Spatial features on or off
hist_feat      = True       # Histogram features on or off
hog_feat       = True       # HOG features on or off
```

```Python
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
```

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search from `pixel=350 to pixel=700`. Along this range I changed the scale according the pixel was reaching the bottom. I started using a scale 0.9 and I increased by each window. The scale used was:
```Python
ystart_ystop_scale = [(350, 400, 0.9),(400, 500, 1.), (400, 600, 1.5),(500, 650, 2.0), (550, 700, 2.5)]
```
Actually I have tried a several attempts by different number of the windows and also the sizes of them. 


#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:
![test1_rectangles](https://user-images.githubusercontent.com/19958282/41938449-8441438e-7969-11e8-9cca-aa00c18c5ae8.png)
![test4_rectangles](https://user-images.githubusercontent.com/19958282/41938450-8485a858-7969-11e8-8698-087827ef9a91.png)
![test5_rectangles](https://user-images.githubusercontent.com/19958282/41938452-84cc34c6-7969-11e8-9b55-627402d8ba2b.png)
![test_rectangles](https://user-images.githubusercontent.com/19958282/41938453-850ff8d2-7969-11e8-804c-28b184e267fe.png)


---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](https://youtu.be/UaJWaO9KpZI)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions (Threshold used was 8). I constructed bounding boxes to cover the area of each blob detected.  Here are the examples of the image with their respective heatmap before and after the threshold.

![evolution_test4_heatmap](https://user-images.githubusercontent.com/19958282/41939797-341315e6-796d-11e8-9d61-5d0e1c63376b.png)
![evolution_test5_heatmap](https://user-images.githubusercontent.com/19958282/41939798-344d856e-796d-11e8-946c-21bef2a894e5.png)
![evolution_test1_heatmap](https://user-images.githubusercontent.com/19958282/41939800-348b487c-796d-11e8-9419-547c447e711b.png)
![evolution_test3_heatmap](https://user-images.githubusercontent.com/19958282/41939802-34cc8ba2-796d-11e8-8d9d-02b28900a965.png)

#### 3. The final test images processed.
Here will be shown the test images processed.
![test5_processed](https://user-images.githubusercontent.com/19958282/41940314-b655be22-796e-11e8-927a-514c5a824172.png)
![test1_processed](https://user-images.githubusercontent.com/19958282/41940315-b6968894-796e-11e8-8ece-34c82ed2cddc.png)
![test3_processed](https://user-images.githubusercontent.com/19958282/41940316-b6e2083c-796e-11e8-8b42-1b4eaa1947e8.png)
![test4_processed](https://user-images.githubusercontent.com/19958282/41940318-b7283050-796e-11e8-9467-32a8dc13ab90.png)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

It is pretty important to search the best method in order to get more accuracy in terms of detection and also decrease the noise.
There are a lot of options to explore, since parameters until algorithm like (SVN, Decision Three, etc). It's pretty important to explore more possibilities and more features to trainning the network. 

It also will be posted different networks weights done.

