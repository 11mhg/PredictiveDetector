import numpy as np
import os
import pickle
from tqdm import tqdm
from preprocess import *

def gt_to_yolo(box,h,w):
    #num cells width and height are W H 
    # image size width and height are w h
    W = 13
    H = 13
    #size of a cell in pixels
    cellx = w/W
    celly = h/H
    bbox = box[:4]
    #get the grid which lies at the center of gt
    cx = int(bbox[0] / cellx)
    cy = int(bbox[1] / celly)
    
    #clamp so the numbers stay in range of 0-12
    cx = max(0, min(W-1,cx))
    cy = max(0, min(H-1, cy))
    
    return cy, cx

def cxcy_to_x0y0(box):
    cx = box[0]
    cy = box[1]
    w = box[2]
    h = box[3]
    box[0] = (cx- (w/2.))
    box[1] = (cy- (h/2.))
    box[2] = (cx+ (w/2.))
    box[3] = (cy+ (h/2.))
    return box

def intersect_area(box_a,box_b):
    max_xy = np.minimum(box_a[2:], box_b[2:])
    min_xy = np.maximum(box_a[:2], box_b[:2])
    diff_xy = (max_xy - min_xy)
    return diff_xy[0]*diff_xy[1]

def compute_IoU(box_a, box_b):
    inter = intersect_area(box_a, box_b)
    if inter <= 0.0 :
        return 0.0
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a+area_b - inter
    return inter/union    

def IoU_diff(objects):
    out = []
    for i in range(objects.shape[0]-1):
        j = i+1
        box_a = np.array(objects[i,:4])
        box_b = np.array(objects[j,:4])
        if np.count_nonzero(box_a) < 3 and np.count_nonzero(box_b) >= 3:
            out.append(0)
        elif np.count_nonzero(box_b) < 3 and np.count_nonzero(box_a) >=3:
            out.append(0)
        elif np.count_nonzero(box_a) >=3 and np.count_nonzero(box_b) >=3:
            out.append(compute_IoU(box_a, box_b))
        else:
            continue
    return np.array(out)


def get_mot():
    imageDir = './Dataset/MOT/images/train/'
    data = []
    for folder in os.listdir(imageDir):
        if '.txt.' in folder:
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
        assert width!=0
        with open(imageDir+folder+'/gt/gt.txt','r') as gt:
            dict_annot = {}
            dict_annot['img_height'] = height
            dict_annot['img_width'] = width
            dict_annot['frame'] = {}
            for index, lines in enumerate(gt):
                splitline = [float(x.strip()) for x in lines.split(',')]
                label = int(splitline[7])-1
                x_val = splitline[2] 
                y_val = splitline[3] 
                box_width = splitline[4] 
                box_height = splitline[5] 

                x_center = x_val + (box_width/2.)
                y_center = y_val + (box_height/2.)
                object_id = int(splitline[1])
                x0 = x_center - (box_width/2.)
                y0 = y_center - (box_height/2.)
                x1 = x_center + (box_width/2.)
                y1 = y_center + (box_height/2.)
                box = [x0,y0,x1,y1, label,object_id]
                box = np.array(box)
                frame_id = int(splitline[0])
                if frame_id not in dict_annot['frame']:
                    dict_annot['frame'][frame_id] = []
                dict_annot['frame'][frame_id].append(box)
                dict_annot['img_dir'] = imageDir+folder+'/img1/'
            data.append(dict_annot)
    return data

def true_to_grid(datas,grid_H, grid_W, classes, anchors):
    C = len(classes)
    B = len(anchors)
    H = grid_H
    W = grid_W
    processed_data = []
    pbar = tqdm(datas)
    for data in pbar:
        processed_frames = {}
        w = data['img_width']
        h = data['img_height']
        cellx = w/W
        celly = h/H
        for frame_id in sorted(data['frame'].keys()):
            pbar.set_description('Converting to yolov2 grid format: {}'.format(str(frame_id)))
            boxes = np.array(data['frame'][frame_id])
            labels = boxes[:,4]
            object_ids = boxes[:,5]
            boxes = boxes[:,:4]
            boxes[:,0] = boxes[:,0]/w
            boxes[:,2] = boxes[:,2]/w
            boxes[:,1] = boxes[:,1]/h
            boxes[:,3] = boxes[:,3]/h
            allobj = []
            for ind in range(object_ids.shape[0]):
                obj_id = int(object_ids[ind])
                box = boxes[ind,:]
                label = int(labels[ind])
                cx = box[0]/cellx
                cy = box[1]/celly
                if cx >= W or cy >= H: continue
                #the following is based on darkflow yolov2 batch data script lines 32-44
                b_w = box[2] / w
                b_h = box[3] / h
                #box regression and label
                obj = [label, cx-np.floor(cx), cy-np.floor(cy), np.sqrt(b_w), np.sqrt(b_h)]
                #grid index 
                obj += [int(np.floor(cy)*W + np.floor(cx))]
                obj += [obj_id]
                allobj.append(obj)
            #this algorithm focuses on accumulating all objects for ONE picture and translating to grid format
            #Now we calculate the appropriate values
            allobj = np.array(allobj)
            obj_ids = np.zeros([H*W,B])
            prear = np.zeros([H*W,4])
            for obj in allobj:
                ind = int(obj[5])
                obj_ids[ind,:] = [obj[6]] * B
                prear[ind,0] = obj[1] - obj[3]**2 * .5 * W #xleft
                prear[ind,1] = obj[2] - obj[4]**2 * .5 * H #yup
                prear[ind,2] = obj[1] + obj[3]**2 * .5 * W #xright
                prear[ind,3] = obj[2] + obj[4]**2 * .5 * H #ybottom
            #calculate GT area and the such
            upleft = np.expand_dims(prear[:,0:2],1)
            botright = np.expand_dims(prear[:,2:4],1)
            wh = botright - upleft
            area = wh[:,:,0] * wh[:,:,1]
            upleft = np.concatenate([upleft]*B,1)
            botright = np.concatenate([botright]*B,1)
            area = np.concatenate([area] * B, 1)
            
            
            #calculate anchor box areas and the such
            coords = np.ones([H,W,B,4])
            coords = coords.reshape((H*W,B,4))
            coords_xy = coords[:,:,0:2]
            coords_wh = np.sqrt(coords[:,:,2:4] * np.reshape(anchors,[1,B,2]) / np.reshape([W,H],[1,1,2]))
            coords[:,:,0:2] = coords_xy
            coords[:,:,2:4] = coords_wh
            new_wh = np.power(coords[:,:,2:4],2) * np.reshape([W,H],[1,1,2])
            new_area = wh[:,:,0] * wh[:,:,1]
            newcenter = coords[:,:,0:2]
            floor = newcenter - (new_wh * .5)
            ceil = newcenter + (new_wh * .5)


            #calculate IOU of anchor box vs GT 
            intersect_upleft = np.maximum(floor,upleft)
            intersect_botright = np.minimum(ceil,botright)
            intersect_wh = intersect_botright - intersect_upleft
            intersect_wh = np.maximum(intersect_wh, 0.0) # nothing lower than 0
            intersect = np.multiply(intersect_wh[:,:,0], intersect_wh[:,:,1]) 
            
            #Get mask
            iou = np.true_divide(intersect, area+new_area - intersect)
            best_box = np.equal(iou,np.amax(iou ,axis=1, keepdims = True))
            best_box = best_box.astype(float)
            #apply mask on object ids
            obj_ids = np.multiply(best_box,obj_ids)
            #at this point, obj ids should be an array with object ids labelled on an array of [H*W,B]
            #reshape for comprehension
            obj_ids = obj_ids.reshape([H,W,B]) 
            processed_frames[frame_id] = obj_ids
        processed_data.append(processed_frames)
    return processed_data

