
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
import pickle
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from utilities import extract_features, get_hog_features, find_cars, find_cars_2, add_heat, apply_threshold, draw_labeled_bboxes, draw_res
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage import img_as_float
from scipy.ndimage.measurements import label
from moviepy.editor import VideoFileClip

class LabelsQueue():
    def __init__(self):
        # Number labels to store
        self.queue_len = 10
        self.queue = []

    # Put new frame
    def put_labels(self, labels):
        if (len(self.queue) > self.queue_len):
            tmp = self.queue.pop(0)
        self.queue.append(labels)

    # Get last N frames hot boxes
    def get_labels(self):
        b = []
        for label in self.queue:
            b.extend(label)
        return b

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

last_hot_labels = LabelsQueue()
img = mpimg.imread('test_images/test5.jpg')


def cars_detection(img):
    out_img, win_pos = find_cars_2(img, svc, X_scaler, orient,'YCrCb', pix_per_cell,
                                 cell_per_block, spatial_size, hist_bins)
    # Read in image similar to one shown above

    plt.imshow(out_img)
    plt.savefig('output_car_found_test5.jpg')

    heat = np.zeros_like(img[:, :, 0]).astype(np.float)

    # AVG boxes
    last_hot_labels.put_labels(win_pos)
    win_pos = last_hot_labels.get_labels()

    # Add heat to each box in box list
    heat = add_heat(heat, win_pos)


    plt.imshow(heat, cmap='hot')
    plt.savefig('heatmap_before_test5.jpg')
    # Apply threshold to help remove false positives
    heat = apply_threshold(heat, 2)
    plt.imshow(heat, cmap='hot')
    plt.savefig('heatmap_after_test5.jpg')

    # Visualize the heatmap when displaying
    heatmap = np.clip(heat, 0, 255)
    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    #draw_img = draw_labeled_bboxes(np.copy(img), labels)

    # plot_row2(img, out_img, 'Source', 'All Detections')
    # plot_row2(draw_img, heatmap, 'Car Positions', 'Heat Map')
    return labels, heatmap

def image_pipeline(image):
    # Vehicle detection pipeline
    labels, heatmap = cars_detection(image)

    # Create an output image to draw on and  visualize the result
    out_img = image #np.dstack((image, image, image))*255
    # Draw output
    processed_img = draw_res(image, labels)
    return processed_img

# out_img, xpto = find_cars_2(img, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size,
#                    hist_bins)
# out_img, xpto = find_cars(img,ystart,ystop,scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size,
#                     hist_bins)



# last_hot_labels = LabelsQueue()
# output_video = 'video_result.mp4'
# clip1 = VideoFileClip("project_video.mp4")#.subclip(0,3)
# video_clip = clip1.fl_image(image_pipeline)
# video_clip.write_videofile(output_video, audio=False)

out_img = image_pipeline(img)

plt.imshow(out_img)

#print(out_img)
plt.savefig('car_found_test5_processed.jpg')