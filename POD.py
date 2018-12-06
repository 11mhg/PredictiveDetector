import tensorflow as tf
from preprocess import *
from generator import *
import numpy as np
from keras.optimizers import SGD, Adam, RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
import random
import time
import colorsys
from keras import backend as K
from keras.utils import plot_model
from keras.layers import Input, Lambda
from keras.models import Model
from PIL import ImageFont, ImageDraw, Image
from yad2k.models.keras_yolo import *
from yad2k.utils.draw_boxes import draw_boxes
from keras.backend.tensorflow_backend import set_session
import keras.backend as K
from ldljutils import *

anchors_path = 'model_data/mot_anchors.txt'
classes_path = 'model_data/mot_classes.txt'

class Pod():
    def __init__(self, model_type='POD',dtype='float32'):
        self.model_type = model_type
        self.dtype=dtype
        if self.dtype=='float16':
            K.set_floatx('float16')
        
        with open(classes_path) as f:
            self.class_names = f.readlines()
        self.class_names = [c.strip() for c in self.class_names]
        with open(anchors_path) as f:
            self.anchors = f.readline()
            self.anchors = [float(x) for x in self.anchors.split(',')]
            self.anchors = np.array(self.anchors).reshape(-1,2)

        self.score_threshold = 0.3
        self.iou_threshold = 0.5
        self.sequence_length = 5 

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

        '''
        Some session configuration
        '''
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        set_session(sess)


        self.image_shape = (608,608)
        self.detectors_mask_shape = (19,19,5,1) 
        self.matching_boxes_shape = (19,19,5,5)
       
        self.image_input = Input(shape=(None,608,608,3))
        self.boxes_input = Input(shape=(None,5))
        self.detectors_mask_input = Input(shape=self.detectors_mask_shape)
        self.matching_boxes_input = Input(shape=self.matching_boxes_shape)
        self.true_grid_input = Input(shape=self.detectors_mask_shape)

        if self.model_type == 'POD':
            self.model_body = yolo_body(self.image_input,len(self.anchors),len(self.class_names))
        elif self.model_type == 'tiny_POD':
            self.model_body = small_POD_body(self.image_input, len(self.anchors), len(self.class_names))
        self.model_body = Model(self.image_input, self.model_body.output)

        self.model_body.summary()
#        plot_model(self.model_body, to_file='model_{}.png'.format(self.model_type))

    def train(self):
        data = process_MOT_dataset(valid=False)
        valid_data = process_MOT_dataset(valid=True)

        model_loss = Lambda(
                yolo_loss,
                output_shape=(1, ),
                name='pod_loss',
                arguments={'anchors': self.anchors,
                        'num_classes': len(self.class_names),
                        #'rescore_confidence': True,
                        'print_loss': False})([
                            self.model_body.output, self.boxes_input,
                            self.detectors_mask_input, self.matching_boxes_input,
                            self.true_grid_input
                        ])
        model = Model(
            [self.image_input, self.boxes_input, self.detectors_mask_input,
            self.matching_boxes_input,self.true_grid_input], model_loss)
        model.compile(
                optimizer='RMSprop', loss={
                    'pod_loss': lambda y_true, y_pred: y_pred
                })
        logging = TensorBoard()
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5,min_lr=1.**-5)
        checkpoint = ModelCheckpoint("pod_{}_weights_best_so_far.h5".format(self.model_type), monitor='val_loss',
                                     save_weights_only=True, save_best_only=True)
        early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=1, mode='auto')
        true_grid = true_to_grid(data,self.image_shape[0]//32,self.image_shape[1]//32,self.class_names,self.anchors)
        val_true_grid = true_to_grid(valid_data,self.image_shape[0]//32,self.image_shape[1]//32,self.class_names,self.anchors)

        data = get_all_detector_masks(data, self.anchors)
        valid_data = get_all_detector_masks(valid_data, self.anchors)
        
        #Regular generator performs image augmentation
        if self.dtype=='float32':
            generator = VideoSequence(data,true_grid,self.sequence_length, 1,validation=False,dtype=np.float32)
            #valid generator does not perform any image augmentation
            valid_generator = VideoSequence(valid_data,val_true_grid,self.sequence_length, 1,validation=True,dtype=np.float32)
        else:
            generator = VideoSequence(data,true_grid,self.sequence_length, 1,validation=False,dtype=np.float16)
            valid_generator = VideoSequence(valid_data,val_true_grid,self.sequence_length, 1,validation=True,dtype=np.float16)
        #load model
        if not self.loaded:
            if os.path.exists('pod_{}_weights_best_so_far.h5'.format(self.model_type)):
                print("Weights found, loading now.")
                self.model_body.load_weights('pod_{}_weights_best_so_far.h5'.format(self.model_type))
            self.loaded=True
 
        model.fit_generator(generator, epochs = 200, validation_data=valid_generator, shuffle=False, callbacks=[logging,checkpoint,early_stopping,reduce_lr])
        model.save_weights('pod_{}_weights.h5'.format(self.model_type))
        self.model_body.save('model_data/pod_{}.h5'.format(self.model_type))

    def predict_on_images(self, images):
        sess = K.get_session()
        #load model
        if not self.loaded:
            self.model_body = load_model('model_data/pod_{}.h5'.format(self.model_type))
            self.model_body.load_weights('pod_{}_weights_best_so_far.h5'.format(self.model_type))
            self.loaded=True
        yolo_outputs = yolo_head(self.model_body.output, self.anchors, len(self.class_names)) 
        
        input_image_shape = K.placeholder(shape=(2, ))
        boxes, scores, classes = yolo_eval(
                yolo_outputs,
                input_image_shape,
                max_boxes = 30,
                score_threshold = 0.1,
                iou_threshold = 0.5)
    
        images_data_init = [Image.open(image) for image in images]

        images_data = [image.resize((608,608),Image.BICUBIC) for image in images_data_init]

        images_data = [np.array(image,dtype='float32') for image in images_data]

        images_data = [image/255. for image in images_data] 

        images_data = np.expand_dims(images_data,0)

        start =int(round(time.time()*1000))
        out_boxes, out_scores, out_classes = sess.run(
                              [boxes, scores, classes],
                              feed_dict={
                                  self.model_body.input: images_data,
                                  input_image_shape: [images_data_init[0].size[1], images_data_init[0].size[0]],
                                  K.learning_phase(): 0
                              })
        end = int(round(time.time()*1000))
        print("Time taken is "+str(end-start)+" ms")
        return out_boxes, out_scores, out_classes

    def draw_image(self,out_boxes, out_scores, out_classes, image_out):
        print('Found {} boxes for {}'.format(len(out_boxes),image_out))
        image_data = Image.open(image_out)
        font = ImageFont.truetype(
                         font='font/FiraMono-Medium.otf',
                         size=np.floor(3e-2 * image_data.size[1] + 0.5).astype('int32'))
        thickness = (image_data.size[0] + image_data.size[1]) // 300
    
        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]
            label = '{} {:.2f}'.format(predicted_class, score)
    
            draw = ImageDraw.Draw(image_data)
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
    
            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[c])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[c])
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            del draw
    
        image_data.save('predicted_pod_{}.jpg'.format(self.model_type), quality=90)
        return image_data
