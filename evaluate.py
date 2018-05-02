import tensorflow as tf
import os
import imghdr
from preprocess import *
from keras import backend as K
from keras.models import load_model
from PIL import Image, ImageDraw, ImageFont
from tracker import *
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from mean_average_precision.detection_map import DetectionMAP


def evaluate_map(frames,nb_class):
    mAP = DetectionMAP(nb_class)
    for frame in frames:
        mAP.evaluate(*frame)
    mAP.plot()
    plt.savefig('./eval.jpg')
    mean_average_precision = []
    for i in range(nb_class):
        precision, recalls = mAP.compute_precision_recall_(i, True)
        average_precision = mAP.compute_ap(precision, recalls)
        mean_average_precision.append(average_precision)
    print(mean_average_precision)
    print("Mean average precision : {:0.2f}".format(sum(mean_average_precision)/len(mean_average_precision)))

def evaluate():
    model_path = "tracker.h5"
    classes_path = "model_data/mot_classes.txt"
    anchors_path = "model_data/mot_anchors.txt"

    with open(classes_path) as f:
        labels = f.readlines()
    labels = [c.strip() for c in labels]
    
    with open(anchors_path) as f:
       anchors = f.readline()
       anchors = [float(x) for x in anchors.split(',')]
       anchors = np.array(anchors).reshape(-1,2)


    data = process_MOT_dataset()
    data = data[0:1] 
    nb_classes = len(labels)
    nb_anchors = len(anchors)
    
    frames = []

    for video in data:
        img_dir = video['img_dir']
        counter = 0
        K.clear_session()
        t=Tracker()
        for frame_id in sorted(video['frame'].keys()):
            if (frame_id%50==0):
                break
                K.clear_session()
                t=Tracker()
            if (frame_id==len(video['frame'])-1):
                break
            counter+=1
            if counter == 1:
                img_1 = img_dir+str(frame_id).zfill(6)+'.jpg'
            elif counter == 2:
                img_2 = img_dir+str(frame_id).zfill(6)+'.jpg'
            elif counter == 3:
                img_3 = img_dir + str(frame_id).zfill(6)+'.jpg'
            elif counter == 4:
                img_4 = img_dir + str(frame_id).zfill(6)+'.jpg'
            elif counter == 5:
                print(frame_id)
                img_5 = img_dir + str(frame_id).zfill(6)+'.jpg'
                counter = 4
                out_boxes, out_scores, out_classes, _ = t.predict_on_images(img_1, img_2,img_3,img_4, img_5)
                img_1 = img_2
                img_2 = img_3
                img_3 = img_4
                img_4 = img_5
                print("Found {} number of boxes".format(len(out_boxes)))
                pred_boxes = []
                pred_classes = []
                pred_confidence = []
                for i, c in reversed(list(enumerate(out_classes))):
                    predicted_class = labels[c]
                    box = out_boxes[i]
                    score = out_scores[i]
                    box = np.array([box[1],box[2],box[3],box[0]])
                    pred_boxes.append(box)
                    pred_classes.append(c)
                    pred_confidence.append(score)
                pred_boxes = np.array(pred_boxes)
                pred_classes = np.array(pred_classes)
                pred_confidence = np.array(pred_confidence)
            
                true_boxes = []
                true_classes = []
                for box in video['frame'][frame_id+1]:
                    x_center = box[0]
                    y_center = box[1]
                    box_width = box[2]
                    box_height = box[3]
                    box_label = box[4]

                    gt_box = [x_center - (box_width/2.), y_center - (box_height/2.), x_center + (box_width/2.), y_center + (box_height/2.)]
                    true_classes.append(box_label)
                    true_boxes.append(np.array(gt_box))
                true_boxes = np.array(true_boxes)
                true_classes = np.array(true_classes)
                frames.append((pred_boxes,pred_classes,pred_confidence,true_boxes,true_classes))
    evaluate_map(frames,len(labels))



print("Beginning evaluation")
evaluate()
print("done evaluation")









