# Behavioral Cloning Project

## Introduction
This is the third project of term 1 of Udacity's Self Driving Car Nanodegree Program. The aim of the project is to train a model to autonomously steer the car around a track. This is achieved by building a convolutional neural network in Keras which predicts steering angles from images.

### NOTE
Track1Bridge.h5 is an initial model which was obtained by training the same network for 2 epochs, using augmented data without dropping low valued steering angles. Track1Bridge.h5 fails to stay on the track after the bridge.
The above model was then loaded and trained for 3 more epochs (with augmented and dropped data) using model.ipynb. This generated the final model.h5 file, which when used with drive.py completes track1 entirely.
