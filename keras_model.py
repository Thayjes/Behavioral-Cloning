# python file
from keras.layers import Dense, Flatten, Activation, Convolution2D, Cropping2D, Lambda, Dropout
from keras.models import Sequential
import csv
import numpy as np
import cv2
from sklearn.utils import shuffle
from shadow_augmentation import shadow_augmentation
from matplotlib import pyplot as plt 
lines = []
with open('./data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)

images = []
measurements = []
i = 0
for line in lines:
	if line[3] == "steering":
		continue
	source_path = line[0]
	measurement = float(line[3])	
	measurements.append(measurement)
	filename = source_path.split('/')[-1]
	current_path = './data/IMG/' + filename
	image = cv2.imread(current_path)
	image =cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	image = shadow_augmentation(image)
	#plt.imshow(image, cmap = 'gray'), plt.show()
	images.append(image)
		
	i +=1
	if i == 3:
		break

X_train = np.array(images)
y_train = np.array(measurements)
print(X_train.shape), print(y_train.shape)
sample_X_train = X_train#[0:1000,:,:,:]
sample_y_train = y_train#[0:1000]
sample_X_train, sample_y_train = shuffle(X_train, y_train)
print(sample_X_train.shape), print(sample_y_train.shape)
def shadow_augmentation(image):
    
    # Let us randomly define the quadrilateral where we want to apply the shadow 
    pt1 = np.array([np.random.choice([0, image.shape[1]]), 0])
    pt2 = np.array([pt1[0], image.shape[0]])
    pt3 = np.array([np.random.randint(0, image.shape[0]//2), image.shape[0]])
    pt4 = np.array([np.random.randint(0, image.shape[0]//2), 0])
    pts = np.array([pt1, pt2, pt3, pt4])
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
def read_image(line, index):
		source_path = line[index]
		filename = source_path.split('/')[-1]
		current_path = './data/IMG/' + filename
		image = cv2.cvtColor(cv2.imread(current_path), cv2.COLOR_BGR2RGB)
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

def augmentation(line):
	image_index = 0
	cam_image = np.random.choice(['left', 'right', 'center'])
	steering = float(line[3])
	if abs(steering) < 0.05:
		steering_prob = np.random.random()
		if steering_prob < 0.3:
			return 0, 0
	if cam_image == "left":
		steering += 0.25
		image_index = 1
	elif cam_image == "right":
		steering -= 0.25
		image_index = 2
	image = read_image(line, image_index)
	flip_prob = np.random.random()
	if flip_prob > 0.5:
		image = cv2.flip(image, 1)
		steering = -1*steering
	# Brightness Augmentation
	image = brightness_augment(image)
	# Shadow Augmentation
	image = shadow_augmentation(image)
	# Preprocessing
	image = preprocess_image(image)
	return image, steering
def crop_image(image):
	cropped_image = image[55:135, :, :]
	return image 
def normalize_image(image):
	image.astype(np.float32)
	image = image/255. - 0.5 
	return image 
def resize_image(image, target_shape):
	return cv2.resize(image, target_shape)
		
def preprocess_image(image,target_shape):
	image = crop_image(image)
	image = normalize_image(image)
	image = resize_image(image, target_shape)
	return image 

def resize_function(image):
	import tensorflow as ktf
	resized = ktf.image.resize_images(image, (64, 64))
	return resized
def data_generator(lines, batch_size = 32):
	N = len(lines)
	batches_per_epoch = N // batch_size
	i = 0
	
	while(True):
		start = i*batch_size
		end = start + batch_size - 1
		# Initialize the batch data 
		X_train = np.zeros((batch_size, 64, 64, 3), dtype = np.float32)
		y_train = np.zeros((batch_size,), dtype = np.float32)
		for k in range(batch_size):
			index = np.random.randint(N)
			line = lines[index]
			keep_steering_angle = 0
			while keep_steering_angle == 0:
				x, y = augmentation(line)
				# If absolute value of steering angle smaller than 0.1, discard it with some probability.
				if abs(y) < 0.1:
					prob = np.random.uniform()
					# Set steer_prob_threshold depending on how many steering angles close to zero, we want to discard.
					if prob > steer_prob_threshold:
						keep_steering_angle = 1
				else:
					# If absolute value of steering angle is greater than 0.1, then we keep it.
					keep_steering_angle = 1
			X_train[k], y_train[k] = x, y 	
		i += 1 
		if i == batches_per_epoch - 1:
			i = 0 
		yield X_train, y_train 
			
			
def grayscale(image):
	return 0 
steer_prob_threshold = 0.3
def get_model():
	model = Sequential()
	# Data Preprocessing
	# Cropping Layer
	model.add(Cropping2D(cropping = ((70, 25), (1, 1)), input_shape = (160, 320, 3)))
	# Resizing Layer
	model.add(Lambda(resize_function, input_shape=(65,318,3), output_shape=(64, 64, 3)))
	# Normalization Layer
	model.add(Lambda(lambda x: (x/255.0 -0.5), input_shape = (64, 64, 3)))
	# Convolution 1st layer
	model.add(Convolution2D(filters = 24, kernel_size = (5, 5), strides = (2, 2), padding = "same", input_shape = (64, 64, 3), activation = 'elu'))
	# 2nd layer
	model.add(Convolution2D(filters = 36, kernel_size = (5, 5), strides = (2, 2), padding = "same", input_shape = (32, 32, 24)), activation = 'elu')
	# 3rd layer
	model.add(Convolution2D(filters = 48, kernel_size = (5, 5), strides = (2, 2), padding = "same", input_shape = (16, 16, 36), activation = 'elu'))
	# 4th layer 
	model.add(Convolution2D(filters = 64, kernel_size = (3, 3), strides = (1, 1), padding = "valid", input_shape = (8, 8, 48), activation = 'elu'))
	# 5th layer
	model.add(Convolution2D(filters = 64, kernel_size = (3, 3), strides = (1, 1), padding = "valid", input_shape = (6, 6, 64), activation = 'elu'))

	model.add(Flatten())
	model.add(Dense(1164))
	model.add(Dropout(0.5))
	model.add(Dense(100))
	model.add(Dropout(0.5))
	model.add(Dense(50))
	model.add(Dense(1))
	model.summary()
	model.compile(loss = "mse", optimizer = "adam")
	return model 

if __name__ in "__main__":
	model.fit_generator(training_generator, steps_per_epoch = 10, epochs = 1, verbose = 1)
	#model.fit(sample_X_train, sample_y_train, validation_split = 0.2, epochs = 10, batch_size = 32)
	model.save('sample_model_NVIDIA_generator.h5')



