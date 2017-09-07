# **Behavioral Cloning** 


---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/Track1_Network.PNG "Model Visualization"
[image2]: ./examples/HistNoAug.jpeg "Histogram of Original Steering Angle Data"
[image3]: ./examples/Augmented_Images.jpeg "Augmented Images"
[image4]: ./examples/HistAugNoDrop.jpeg "Histogram of Augmented Steering Angle Data"
[image5]: ./examples/HistAugDrop(40%).jpeg "Histogram of Augmented Steering Angle Data (40%drop)"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 filter sizes and depths between 32 and 128 (model.py lines 18-24) followed by 3 fully connected layers of sizes 1024, 512 and 100. 

The model includes ELU layers to introduce nonlinearity (code line 20), and the data is normalized in the model using an external function (code line ??). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 21). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

#### 4. Appropriate training data

The training data used was entirely the dataset provided by Udacity.

For details about how I augmented the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to observe the training and validation loss and experiment with the number of layers, filters and dropout. And finally test it on the track to see how well it performs.

Initially I did not drop steering angles from the dataset which were close to zero. I later learned that the set was highly biased for steering angles at zero, by observing the histogram of the angles.

My first attempt at the model was to use a convolution neural network model similar to the NVIDIA network. I thought this model might be appropriate because it has been used for similar applications with success. 

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. 

Unfortunately this network resulted in poor performance on the track and the first few attempts at running the simulator were not good.

I then began implementing data augmentation, in the hope that this would reduce the bias of the data(such as zero steering angle, and angles for left turns). Further details on augmentation techniques is covered below.

I then observed that the model architecture may be too involved, and with the augmented dataset, a smaller network might perform better for this task. After training for 3 epochs with a smaller network on the augmented data set, this improved performance and my car was able to go past the bridge but shortly fell off the track afterwards.

Finally I dropped around 40% of the steering angle data with absolute values <= 0.01. And trained the same model for another 3 epochs.

This final step took the car over the finish line, and it succesfully completed one lap around the entire track autonomously without leaving the track even once!


#### 2. Final Model Architecture

The final model architecture  consisted of a convolution neural network with the following layers and layer sizes 

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 64x64x3 RGB Image   							| 
| Convolution 1x1     	| 1x1 stride, same padding, outputs 64x64x3	|
| ELU					|												|
| Convolution 3x3  | 1x1 stride, same padding, outputs 64x64x32 |
| ELU					|												|
| Max Pooling		      | 2x2 stride, outputs 32x32x32 |
| Dropout 0.5		|												|
| Convolution 3x3  | 1x1 stride, same padding, outputs 32x32x64 |
| ELU					|												|
| Max Pooling		      | 2x2 stride, outputs 16x16x64 |
| Dropout 0.5		|												|
| Convolution 3x3  | 1x1 stride, same padding, outputs 16x16x128 |
| ELU					|												|
| Max Pooling		      | 2x2 stride, outputs 8x8x128  |
| Dropout 0.5		|	
| Flatten 8x8x128 | outputs 8192x1			|
|	Dense   8192x1			|		outputs 512x1									|
| ELU					|												|
|	Dense   512x1 			|		outputs 200x1									|
| ELU					|												|
|	Dense   200x1 			|		outputs 100x1									|
| ELU					|												|
|	Dense   100x1 			|		outputs 1x1  									|

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![Network Architecture Diagram][image1]

#### 3. Creation of the Training Set & Training Process

I was able to complete the first track successfully using only the Udacity data along with augmentation techniqiues.

The first step was to visualize the histogram of the steering angles of the given training data.

![Histogram without Augmentation][image2]

Clearly there is a bias towards angles close to zero as well an unequal distribution of sample for different angle values.

I used the following augmentation techniques to improve the quality of the dataset:
* Brightness Augmentation: Adjusted the brightness of the image randomly.
* Flipping: Randomly flipped the image and reversed the steering angle accordingly.
* Shadow Augmentation: Placed a shadow randomly on portions of the image.
* Using the Left and Right Camera Images: Randomly selected left, right camera images(along with center) and added a corrective steering
angle value of +- 0.25.
* Translation: Randomly shifted the image by a small value in x and y directions, to imitate slopes and getting close to the sides
of the track.

I used the following preprocessing methods:
* Cropping: Cropped the image by 70 pixels at top and 25 pixels at the bottom, so as to just retain the important part of the image.
* Resizing: Resized the image from (320, 160) to (64, 64) to improve speed of training.
* Normalized: Normalized the image pixel values to lie in the range -0.5 to 0.5.

Here are some examples of augmented and preprocessed images:

![alt text][image3]

Here is the histogram of steering angles after augmentation:

![alt text][image4]

And finally, the histogram of steering angles after dropping 40% of angles with absolute values <=0.01:

![alt text][image5]

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
