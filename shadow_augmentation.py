# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 13:17:54 2017

@author: lenovo
"""
import numpy as np
import cv2
from matplotlib import pyplot as plt

def shadow_augmentation(image):
    
    # Let us define the random quadrilateral where we want to apply the shadow 
    pt1 = np.array([np.random.choice([0, image.shape[1]]), 0])
    pt2 = np.array([pt1[0], image.shape[0]])
    pt3 = np.array([np.random.randint(0, image.shape[0]//2), image.shape[0]])
    pt4 = np.array([np.random.randint(0, image.shape[0]//2), 0])
    pts = np.array([pt1, pt2, pt3, pt4])
    print(pts)
    # Convert the image to Hue, Lightness, Saturation color model.
    image_HLS = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    
    # Initialize the mask
    shadow_mask = 0*image_HLS[:, :, 1]
    # Now fill the quadrilateral as defined before
    shadow_mask = cv2.fillConvexPoly(shadow_mask, pts, 1)
    shadow_prob = np.random.random()
    # Apply shadow augmentation randomly to the images
    if shadow_prob > 0.5:
        random_shadow = 0.5
        image_HLS[:, :, 1][shadow_mask==1] = image_HLS[:, :, 1][shadow_mask==1]*random_shadow
    
    # Convert back to RGB Color model.
    image = cv2.cvtColor(image_HLS, cv2.COLOR_HLS2RGB)
    return image
def brightness_augment(image):
	# Convert to the HSV colorspace first.
    image = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    image = np.array(image, dtype = np.float64)
	# Randomly assign the brightness value 
    brightness_random = .5 + np.random.uniform()
    image[:,:,2] = image[:,:,2]*brightness_random
    image[:,:,2][image[:,:,2]>255]  = 255
    image = np.array(image, dtype = np.uint8)
    image = cv2.cvtColor(image,cv2.COLOR_HSV2RGB)
    return image
    
if __name__ in "__main__":
    #from keras.layers import Dense, Flatten, Activation, Convolution2D, Cropping2D, Lambda, Dropout
    #from keras.models import Sequential
    import csv
    import numpy as np
    import cv2
    from sklearn.utils import shuffle
    #from shadow_augmentation import shadow_augmentation
    from matplotlib import pyplot as plt 
    lines = []
    with open('C:/Users/lenovo/Documents/SDCND/Project-3/data/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)

    images = []
    measurements = []
    i = 0
    print(i)
    for line in lines:
        if line[3] == "steering":
            continue
        source_path = line[0]
        measurement = float(line[3])		
        measurements.append(measurement)
        filename = source_path.split('/')[-1]
        current_path = 'C:/Users/lenovo/Documents/SDCND/Project-3/data/IMG/' + filename
        image = cv2.imread(current_path)
        
        image =cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #plt.imshow(image, cmap = 'gray'), plt.show()
        image = brightness_augment(image)
        #print(image.shape)
        plt.imshow(image), plt.show()	
        images.append(image)
        i +=1
        if i == 10:
            break