import os, io
import numpy as np
from keras.utils import Sequence
import PIL
import math
import imgaug as ia
from imgaug import augmenters as iaa
#needs to be in [image_data_1, image_data_2, boxes, detectors_masks, matching_true_boxes]

class VideoSequence(Sequence):
    def __init__(self, data, batch_size,validation=False):
        self.data = data
        self.batch_size = batch_size
        self.pdata = []
        self.validation=validation
        self.jitter = True 

        sometimes = lambda aug: iaa.Sometimes(0.5, aug)

        self.aug_pipe = iaa.Sequential(
            [
                iaa.SomeOf((0,5),
                    [
                        iaa.OneOf([
                            iaa.GaussianBlur((0,3.0)),
                            iaa.AverageBlur(k=(2,7)),
                            iaa.MedianBlur(k=(3,11)),
                        ]),
                        iaa.Sharpen(alpha=(0,1.0), lightness=(0.75,1.5)),
                        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0,0.05*255),per_channel=0.5),
                        iaa.Dropout((0.01,0.1), per_channel=0.5),
                        iaa.Add((-10,10), per_channel=0.5),
                        iaa.Multiply((0.5,1.5), per_channel=0.5),
                        iaa.ContrastNormalization((0.5,2.0), per_channel=0.5),
                    ],
                    random_order=True
                )
            ],
            random_order=True
        )

        for video in self.data:
            temp_frame_boxes = []
            temp_frame_image_1 = []
            temp_frame_image_2 = []
            temp_frame_detector_masks = []
            temp_frame_matching_true_boxes = []
            real_counter=0
            counter = 0
            for frame_id in sorted(video['frame'].keys()):
                print(frame_id)
                if counter == 0:
                    temp_frame_image_1 = video['img_dir']+str(frame_id).zfill(6)+'.jpg'
                if counter == 1:
                    temp_frame_image_2 = video['img_dir'] + str(frame_id).zfill(6)+'.jpg'
                if counter == 2:
                    temp_frame_image_3 = video['img_dir'] + str(frame_id).zfill(6)+'.jpg'
                if counter == 3:
                    temp_frame_image_4 = video['img_dir']+str(frame_id).zfill(6)+'.jpg'
                if counter == 4:
                    temp_frame_image_5 = video['img_dir'] + str(frame_id).zfill(6)+'.jpg'
                if counter == 5:
                    temp_frame_boxes = np.array(video['frame'][frame_id])
                    temp_frame_detector_mask = video['detector_mask'][frame_id][0]
                    temp_frame_matching_true_boxes = video['detector_mask'][frame_id][1]
                    self.pdata.append([temp_frame_image_1,temp_frame_image_2,temp_frame_image_3,temp_frame_image_4,temp_frame_image_5,temp_frame_boxes,temp_frame_detector_mask, temp_frame_matching_true_boxes])
                    temp_frame_image_1 = temp_frame_image_2
                    temp_frame_image_2 = temp_frame_image_3
                    temp_frame_image_3 = temp_frame_image_4
                    temp_frame_image_4 = temp_frame_image_5
                    temp_frame_image_5 = video['img_dir']+str(frame_id).zfill(6)+'.jpg'
                    counter = 4
                counter += 1
        print("Done Readying the Data for the sequence")

    def __len__(self):
        return math.floor(len(self.pdata)/self.batch_size)

    def aug_image(self, image):
        return np.array(self.aug_pipe.augment_image(np.array(image,dtype=np.uint8)),dtype=np.float)


    def __getitem__(self, idx):
        batch =  self.pdata[idx * self.batch_size:(idx + 1) * self.batch_size]
        #batch_x = []
        input_image_1 = []
        input_image_2 = []
        input_image_3 = []
        input_image_4 = []
        input_image_5 = []
        input_boxes = []
        input_mask = []
        input_true_boxes = []
        batch_y = []
        for elem in batch:
            
            image_1 = PIL.Image.open(elem[0])
            image_2 = PIL.Image.open(elem[1])
            image_3 = PIL.Image.open(elem[2])
            image_4 = PIL.Image.open(elem[3])
            image_5 = PIL.Image.open(elem[4])
            image_1 = image_1.resize((608,608),PIL.Image.BICUBIC)
            image_2 = image_2.resize((608,608), PIL.Image.BICUBIC)
            image_3 = image_3.resize((608,608),PIL.Image.BICUBIC)
            image_4 = image_4.resize((608,608), PIL.Image.BICUBIC)
            image_5 = image_5.resize((608,608),PIL.Image.BICUBIC)
            image_1 = np.array(image_1, dtype=np.float)
            image_2 = np.array(image_2, dtype=np.float)
            image_3 = np.array(image_3, dtype=np.float)
            image_4 = np.array(image_4, dtype=np.float)
            image_5 = np.array(image_5, dtype=np.float)
            if not self.validation:
                image_1 = self.aug_image(image_1)
                image_2 = self.aug_image(image_2)
                image_3 = self.aug_image(image_3)
                image_4 = self.aug_image(image_4)
                image_5 = self.aug_image(image_5)
            image_1 /= 255
            image_2 /= 255
            image_3 /= 255
            image_4 /= 255
            image_5 /= 255
            boxes = elem[5]
            temp = []
            for box in boxes:
                centerx = box[0]
                centery = box[1]
                box_width = box[2]
                box_height = box[3]
                label = box[4]
                temp.append([centerx,centery,box_width,box_height,label])
            boxes = np.array(temp)
            mask = elem[6]
            true_boxes = elem[7]
            #temp = [image_1, image_2, boxes,mask,true_boxes]
            #temp = np.array(temp)
            #batch_x.append(temp)
            input_image_1.append(image_1)
            input_image_2.append(image_2)
            input_image_3.append(image_3)
            input_image_4.append(image_4)
            input_image_5.append(image_5)
            input_boxes.append(boxes)
            input_mask.append(mask)
            input_true_boxes.append(true_boxes)
            batch_y.append(np.zeros(len(image_1)))
        max_boxes = 0
        for boxz in input_boxes:
            if boxz.shape[0] > max_boxes:
                max_boxes = boxz.shape[0]
        #zero pad 
        for i, boxz in enumerate(input_boxes):
            if boxz.shape[0] < max_boxes:
                zero_padding = np.zeros( (max_boxes - boxz.shape[0], 5), dtype=np.float32)
                input_boxes[i] = np.vstack((boxz, zero_padding))
        temp = np.array([np.array(input_image_1),np.array(input_image_2),np.array(input_image_3),np.array(input_image_4),np.array(input_image_5)]).transpose(1,0,2,3,4)
        return [temp,np.array(input_boxes), np.array(input_mask),np.array(input_true_boxes)] , np.array(batch_y)

