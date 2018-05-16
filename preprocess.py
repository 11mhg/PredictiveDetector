import tensorflow as tf
import PIL
import numpy as np
import os
from yad2k.models.keras_yolo import preprocess_true_boxes, yolo_eval, yolo_head
from keras import backend as K
from PIL import Image
import cv2
from keras.models import load_model
import h5py
import pickle

dict_MOT = {1:'Pedestrian',2:'Person on vehicle',3:'Car',4:'Bicycle',5:'Motorbike',6:'Non motorized vehicle',7:'static person',8:'distractor',9:'occluder',10:'occluder on the ground',11:'occluder full',12:'Reflection'}


def process_MOT_dataset(valid=False):
    if not valid:
        with open(b'MOT-train.obj', 'rb') as f:
           data = pickle.load(f)
    else:
        with open(b'MOT-train.obj', 'rb') as f:
           data = pickle.load(f)
    return data

def get_all_detector_masks(data, anchors):
    #data is full dict from process_MOT_dataset
    for video in data:
        video['detector_mask'] = {}
        for frame_id in video['frame']: 
            detectors_mask, matching_true_box = preprocess_true_boxes(video['frame'][frame_id], anchors,[608,608])
            video['detector_mask'][frame_id] = [detectors_mask, matching_true_box]
    return data

def get_video(video_source):
    vidcap = cv2.VideoCapture(video_source)
    video = []
    success, image = vidcap.read()
    count = 0
    video.append(image)
    success = True
    while success:
        success, image = vidcap.read()
        if success:
            video.append(image)
        count+=1
    print("Num frames : ", count)
    return video


def get_MOT_as_COCO(valid=False):
    imageDir = 'Dataset/MOT/images/train/'
    data = []
    for folder in os.listdir(imageDir):
        if '.txt' in folder:
            continue
        height = 0
        width = 0
        with open(imageDir+folder+'/seqinfo.ini','r') as info:
            for lines in info:
                if 'imWidth' in lines:
                    lines = lines.split('=')
                    width = float(lines[1])
                elif 'imHeight' in lines:
                    lines = lines.split('=')
                    height = float(lines[1])
                else:
                    continue

        assert height!=0
        assert width !=0

        with open(imageDir+folder+'/gt/gt.txt','r') as gt:
            dict_annot = {}
            dict_annot['img_height'] = height
            dict_annot['img_width'] = width
            dict_annot['frame'] = {}
            print(height,width)
            for index, lines in enumerate(gt):
                splitline = [float(x.strip()) for x in lines.split(',')]
                label = int(splitline[7])-1
                x_val = splitline[2]
                y_val = splitline[3]
                box_width = splitline[4]
                box_height = splitline[5]

                x_center = x_val + (box_width/2.)
                y_center = y_val + (box_height/2.)
                
                x_center = max(0,min(0.9999999999,x_center/width))
                y_center = max(0,min(0.9999999999,y_center/height))
                box_width = max(0,min(0.999999999,box_width/width))
                box_height = max(0,min(0.99999999,box_height/height))

                box = [x_center, y_center, box_width, box_height, label]
                box = np.array(box)

                frame_id =int(splitline[0])
                if frame_id in dict_annot['frame']:
                    dict_annot['frame'][frame_id].append(box)
                else:
                    dict_annot['frame'][frame_id] = []
                    dict_annot['frame'][frame_id].append(box)
            dict_annot['img_dir'] = imageDir+folder+'/img1/'
            data.append(dict_annot)
    for video in data:
        threshold = int(0.8*len(video['frame']))
        new_video = {}
        for index, frame_id in enumerate(video['frame']):
            if valid and index < threshold:
                continue
            if not valid and index > threshold:
                continue
            boxes = video['frame'][frame_id]
            boxes = np.array(boxes)
            new_video[frame_id] = boxes
        video['frame'] = new_video
        print("Video size is : "+str(len(new_video)))
    if not valid:
        with open(b'MOT-train.obj', 'wb') as f:
            pickle.dump(data, f)

    else:
        with open(b'MOT-test.obj', 'wb') as f:
            pickle.dump(data, f)

def image2video(images,filename):
    width, height = images[0].size
    fourcc = cv2.VideoWriter_fourcc(*'MPEG')
    out = cv2.VideoWriter(filename, fourcc, 30.0, (width, height))

    for image in images:
        temp = np.array(image)
        temp = cv2.cvtColor(temp, cv2.COLOR_RGB2BGR)
        out.write(temp)
    out.release()

    print("Output video save at {}".format(filename))
