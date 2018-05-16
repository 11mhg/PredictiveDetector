import os, io
import numpy as np
from keras.utils import Sequence
import PIL
import math
import imgaug as ia
from imgaug import augmenters as iaa
#needs to be in [image_data_1, image_data_2, boxes, detectors_masks, matching_true_boxes]

class VideoSequence(Sequence):
    def __init__(self, data, sequence_size, batch_size,validation=False):
        self.data = data
        self.batch_size = batch_size
        self.seq = sequence_size
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
            temp_frame_images = []
            temp_frame_detector_masks = []
            temp_frame_matching_true_boxes = []
            counter = 0
            for frame_id in sorted(video['frame'].keys()):
                if counter >= self.seq:
                    temp_frame_boxes = np.array(video['frame'][frame_id])
                    temp_frame_detector_mask = video['detector_mask'][frame_id][0]
                    temp_frame_matching_true_boxes = video['detector_mask'][frame_id][1]
                    self.pdata.append(temp_frame_images+[temp_frame_boxes,temp_frame_detector_mask, temp_frame_matching_true_boxes])
                    temp_frame_images = [temp_frame_images[i] for i in range(1,len(temp_frame_images))]
                    temp_frame_images.append(video['img_dir']+str(frame_id).zfill(6)+'.jpg')
                    counter = self.seq - 1
                else:
                    temp_frame_images.append(video['img_dir']+str(frame_id).zfill(6)+'.jpg')
                counter += 1 
        print("Done Readying the Data for the sequence")

    def __len__(self):
        return math.floor(len(self.pdata)/self.batch_size)

    def aug_image(self, image):
        return np.array(self.aug_pipe.augment_image(np.array(image,dtype=np.uint8)),dtype=np.float)


    def __getitem__(self, idx):
        batch =  self.pdata[idx * self.batch_size:(idx + 1) * self.batch_size]
        #batch_x = []
        #set size
        input_images = []
        input_boxes = []
        input_mask = []
        input_true_boxes = []
        batch_y = []
        for index, elem in enumerate(batch):
            images = [PIL.Image.open(elem[i]) for i in range(self.seq-1)]
            images = [i.resize((608,608),PIL.Image.BICUBIC) for i in images]
            images = [np.array(i, dtype=np.float) for i in images]
            if not self.validation:
                images = [self.aug_image(i) for i in images]
            images =[i/255 for i in images]
            boxes = elem[self.seq]
            temp = []
            for box in boxes:
                centerx = box[0]
                centery = box[1]
                box_width = box[2]
                box_height = box[3]
                label = box[4]
                temp.append([centerx,centery,box_width,box_height,label])
            boxes = np.array(temp)
            mask = elem[self.seq+1]
            true_boxes = elem[self.seq+2]
            if input_images == []:
                input_images.append([np.array(images)])
            else:
                input_images = [input_images[index].append(image) for index, image in enumerate(input_images)]
            input_boxes.append(boxes)
            input_mask.append(mask)
            input_true_boxes.append(true_boxes)
            batch_y.append(np.zeros(len(images)))
        max_boxes = 0
        for boxz in input_boxes:
            if boxz.shape[0] > max_boxes:
                max_boxes = boxz.shape[0]
        #zero pad 
        for i, boxz in enumerate(input_boxes):
            if boxz.shape[0] < max_boxes:
                zero_padding = np.zeros( (max_boxes - boxz.shape[0], 5), dtype=np.float32)
                input_boxes[i] = np.vstack((boxz, zero_padding))
        #final input info requires images in sequence and per batch
        return [np.array(image_batch) for image_batch in input_images]+[np.array(input_boxes), np.array(input_mask),np.array(input_true_boxes)] , np.array(batch_y)

