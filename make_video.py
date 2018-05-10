import tensorflow as tf
import numpy as np
import os
from pod import *
from preprocess import *

datas = process_MOT_dataset()


datas = datas[0:1]


t = Pod()

images = []
for data in datas:
    
    general_img_dir = data['img_dir']
    print(general_img_dir)
    frame_1 = 0
    frame_2 = 1
    frame_3 = 2
    frame_4 = 3
    frame_5 = 4
    image_1 = None

    counter = 0

    while(True):
        counter += 1
        if counter%100 ==0:
            t=Pod() 
        frame_1 +=1
        frame_2 +=1
        frame_3 +=1
        frame_4 +=1
        frame_5 +=1
        frame_6 = frame_5 + 1
        if (image_1 is not None):
            image_1 = image_2 
            image_2 = image_3
            image_3 = image_4
            image_4 = image_5
        else:
            image_1 = general_img_dir + str(frame_1).zfill(6)+'.jpg'
            image_2 = general_img_dir + str(frame_2).zfill(6)+'.jpg'
            image_3 = general_img_dir + str(frame_3).zfill(6)+'.jpg'
            image_4 = general_img_dir + str(frame_4).zfill(6)+'.jpg'
    
        image_5 = general_img_dir + str(frame_5).zfill(6)+'.jpg' 
    
        if not os.path.isfile(image_5):
            break

        out_boxes, out_scores, out_classes, image_6 = t.predict_on_images(image_1,image_2, image_3, image_4, image_5)
        image_6_test = general_img_dir + str(frame_6).zfill(6)+'.jpg'
        assert image_6 == image_6_test
        if not os.path.isfile(image_6):
            break
        images.append(t.draw_image(out_boxes, out_scores, out_classes, image_6))
    print("done video {}".format(general_img_dir))
image2video(images,"output_video.avi")
