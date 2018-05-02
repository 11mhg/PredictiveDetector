import tensorflow as tf
import numpy as np
import os
from tracker import *

img_dir = './Dataset/MOT/images/train/MOT17-13/img1/'

frame_1 = 0
frame_2 = 1
frame_3 = 2
frame_4 = 3
frame_5 = 4

t = Tracker()


for i in range(1):
    frame_1 +=1
    frame_2 +=1
    frame_3 +=1
    frame_4 +=1
    frame_5 +=1
    frame_6 = frame_5 + 1
    image_1 = img_dir + str(frame_1).zfill(6)+'.jpg'
    image_2 = img_dir + str(frame_2).zfill(6)+'.jpg'
    image_3 = img_dir + str(frame_3).zfill(6)+'.jpg'
    image_4 = img_dir + str(frame_4).zfill(6)+'.jpg'
    image_5 = img_dir + str(frame_5).zfill(6)+'.jpg'
    out_boxes, out_scores, out_classes, image_6 = t.predict_on_images(image_1,image_2,image_3, image_4, image_5)
    image_6_test = img_dir + str(frame_6).zfill(6)+'.jpg'
    assert image_6 == image_6_test
    t.draw_image(out_boxes, out_scores, out_classes, image_6)
    print("done")


