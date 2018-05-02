import tensorflow as tf
from preprocess import *
from generator import *
import numpy as np
from keras.optimizers import SGD, Adam, RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
import random
import time
import colorsys
from keras import backend as K
from keras.utils import plot_model
from keras.layers import Input, Lambda
from keras.models import Model
from PIL import ImageFont, ImageDraw, Image
from yad2k.models.keras_yolo import preprocess_true_boxes, yolo_body, yolo_eval, yolo_head, yolo_loss
from yad2k.utils.draw_boxes import draw_boxes

anchors_path = 'model_data/mot_anchors.txt'
classes_path = 'model_data/mot_classes.txt'

class Tracker():
    def __init__(self):
        with open(classes_path) as f:
            self.class_names = f.readlines()
        self.class_names = [c.strip() for c in self.class_names]

        with open(anchors_path) as f:
            self.anchors = f.readline()
            self.anchors = [float(x) for x in self.anchors.split(',')]
            self.anchors = np.array(self.anchors).reshape(-1,2)

        self.score_threshold = 0.3
        self.iou_threshold = 0.5

        self.evaluated = False

        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                     for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
                  map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                  self.colors))
        random.seed(10101)  # Fixed seed for consistent colors across runs.
        random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        random.seed(None)

        self.loaded=False

        self.image_shape = (608,608)
        self.detectors_mask_shape = (19,19,5,1) 
        self.matching_boxes_shape = (19,19,5,5)
       
        self.image_input = Input(shape=(None,608,608,3))
        self.boxes_input = Input(shape=(None,5))
        self.detectors_mask_input = Input(shape=self.detectors_mask_shape)
        self.matching_boxes_input = Input(shape=self.matching_boxes_shape)

        self.model_body = yolo_body(self.image_input,len(self.anchors),len(self.class_names))

        self.model_body = Model(self.image_input, self.model_body.output)

        self.model_body.summary()
        plot_model(self.model_body, to_file='model.png')

    def train(self):
        data = process_MOT_dataset()
        
        model_loss = Lambda(
                yolo_loss,
                output_shape=(1, ),
                name='tracker_loss',
                arguments={'anchors': self.anchors,
                        'num_classes': len(self.class_names),
                        #'rescore_confidence': True,
                        'print_loss': False})([
                            self.model_body.output, self.boxes_input,
                            self.detectors_mask_input, self.matching_boxes_input
                        ])
        model = Model(
            [self.image_input, self.boxes_input, self.detectors_mask_input,
            self.matching_boxes_input], model_loss)
        model.compile(
                optimizer='RMSprop', loss={
                    'tracker_loss': lambda y_true, y_pred: y_pred
                })
        logging = TensorBoard()
        checkpoint = ModelCheckpoint("tracker_weights_best_so_far.h5", monitor='val_loss',
                                     save_weights_only=True, save_best_only=True)
        early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='auto')
        data = get_all_detector_masks(data, self.anchors)

        #Regular generator performs image augmentation
        generator = VideoSequence(data[0:1], 1,validation=False)
        #valid generator does not perform any image augmentation
        valid_generator = VideoSequence(data[0:1], 1,validation=True)

        #load model
        if not self.loaded:
            self.model_body.load_weights('tracker_weights_best_so_far.h5')
            self.loaded=True
 
        model.fit_generator(generator, epochs = 50, validation_data=valid_generator, shuffle=True, callbacks=[logging,checkpoint,early_stopping])
        model.save_weights('tracker_weights.h5')
        self.model_body.save('model_data/tracker.h5')

    def predict_on_images(self, image_1, image_2, image_3, image_4, image_5):
        sess = K.get_session()
        #load model
        if not self.loaded:
            self.model_body = load_model('model_data/tracker.h5')
            self.model_body.load_weights('tracker_weights_best_so_far.h5')
            self.loaded=True
    
        yolo_outputs = yolo_head(self.model_body.output, self.anchors, len(self.class_names)) 
        
        input_image_shape = K.placeholder(shape=(2, ))
        boxes, scores, classes = yolo_eval(
                yolo_outputs,
                input_image_shape,
                score_threshold = 0.1,
                iou_threshold = 0.5)
    
        image_1_data_init = Image.open(image_1)
        image_2_data_init = Image.open(image_2)
        image_3_data_init = Image.open(image_3)
        image_4_data_init = Image.open(image_4)
        image_5_data_init = Image.open(image_5)


        image_1_data = image_1_data_init.resize((608,608), Image.BICUBIC)
        image_2_data = image_2_data_init.resize((608,608), Image.BICUBIC)
        image_3_data = image_3_data_init.resize((608,608), Image.BICUBIC)
        image_4_data = image_4_data_init.resize((608,608), Image.BICUBIC)
        image_5_data = image_5_data_init.resize((608,608), Image.BICUBIC) 
    
        image_1_data = np.array(image_1_data, dtype='float32')
        image_2_data = np.array(image_2_data, dtype='float32')
        image_3_data = np.array(image_3_data, dtype='float32')
        image_4_data = np.array(image_4_data, dtype='float32')
        image_5_data = np.array(image_5_data, dtype='float32')

    
        image_1_data /=255.
        image_2_data /=255.
        image_3_data /=255.
        image_4_data /=255.
        image_5_data /=255.
     
        images_data = [image_1_data, image_2_data, image_3_data, image_4_data, image_5_data]

        images_data = np.expand_dims(images_data,0)

        start =int(round(time.time()*1000))
        out_boxes, out_scores, out_classes = sess.run(
                              [boxes, scores, classes],
                              feed_dict={
                                  self.model_body.input: images_data,
                                  input_image_shape: [image_1_data_init.size[1], image_1_data_init.size[0]],
                                  K.learning_phase(): 0
                              })
        end = int(round(time.time()*1000))
        print("Time taken is "+str(end-start)+" ms")
        image_6 = os.path.splitext(os.path.basename(image_5))[0]
        image_6 = os.path.dirname(image_5)+'/'+str(int(image_6)+1).zfill(6)+'.jpg'
        #sess.close()
        return out_boxes, out_scores, out_classes, image_6

    def draw_image(self,out_boxes, out_scores, out_classes, image_6):
        print('Found {} boxes for {}'.format(len(out_boxes),image_6))
        image_6_data = Image.open(image_6)
        font = ImageFont.truetype(
                         font='font/FiraMono-Medium.otf',
                         size=np.floor(3e-2 * image_6_data.size[1] + 0.5).astype('int32'))
        thickness = (image_6_data.size[0] + image_6_data.size[1]) // 300
    
        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]
            label = '{} {:.2f}'.format(predicted_class, score)
    
            draw = ImageDraw.Draw(image_6_data)
            label_size = draw.textsize(label, font)
    
            top, left, bottom, right = box
            height = abs(top-bottom)
            top = top
            bottom = bottom      
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image_6_data.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image_6_data.size[0], np.floor(right + 0.5).astype('int32'))
            print(label, (left, top), (right, bottom))
            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])
    
            # My kingdom for a good redistributable image drawing library.
            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[c])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[c])
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            del draw
    
        image_6_data.save('predicted_tracker.jpg', quality=90)
        return image_6_data
