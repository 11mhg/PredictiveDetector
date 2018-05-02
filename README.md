# Predictive Object Detector

Modifications and Predictive Detection contributions by : Mohammed Gasmallah


The following code is a modification and an expension on the YAD2K (for more info on YAD2K, look below).
The code has been modified to allow for a recurrency and a time distribution convolutional 2D LSTM layer to
be added. This network takes multiple input frames and predicts bounding boxes on the next frame.

Current implementation works well at predicting COCO classes and MOT Classes.



model:
images->Conv2DLSTM->TimeDistributed(MaxPooling)->TimeDistributed(LeakyReLU)->Detector->Output 


## YAD2K? What is it?

You only look once, but you reimplement neural nets over and over again.

YAD2K is a 90% Keras/10% Tensorflow implementation of YOLO_v2.

Original paper: [YOLO9000: Better, Faster, Stronger](https://arxiv.org/abs/1612.08242) by Joseph Redmond and Ali Farhadi.


YAD2K Source: https://github.com/allanzelener/YAD2K


