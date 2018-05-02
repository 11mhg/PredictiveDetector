import random
import cv2
import argparse
import json
from preprocess import *
import numpy as np



def IOU(ann, centroids):
    w,h = ann
    similarities = []
    for centroid in centroids:
        c_w, c_h = centroid
        if c_w >=w and c_h >=h:
            similarity = w*h/(c_w*c_h)
        elif c_w >=w and c_h <=h:
            similarity = w*c_h/(w*h+(c_w-w)*c_h)
        elif c_w <= w and c_h>=h:
            similarity = c_w*h/(w*h+c_w*(c_h-h))
        else:
            similarity = (c_w*c_h)/(w*h)
        similarities.append(similarity)
    return np.array(similarities)

def avg_IOU(anns, centroids):
    n, d = anns.shape
    sum =0.

    for i in range(anns.shape[0]):
        sum+= max(IOU(anns[i], centroids))

    return sum/n

def print_anchors(centroids):
    anchors = centroids.copy()
    widths = anchors[:, 0]
    sorted_indices = np.argsort(widths)
    r = "anchors: ["
    for i in sorted_indices[:-1]:
        r += '%0.10f,%0.10f, ' % (anchors[i,0], anchors[i,1])
    #there should not be comma after last anchor, that's why
    r += '%0.10f,%0.10f' % (anchors[sorted_indices[-1:],0], anchors[sorted_indices[-1:],1])
    r += "]"
    print (r)

def run_kmeans(ann_dims, anchor_num):
    ann_num = ann_dims.shape[0]
    iterations = 0
    prev_assignments = np.ones(ann_num)*(-1)
    iteration = 0
    old_distances = np.zeros((ann_num, anchor_num))

    indices = [random.randrange(ann_dims.shape[0]) for i in range(anchor_num)]
    centroids = ann_dims[indices]
    anchor_dim = ann_dims.shape[1]

    while True:
        distances = []
        iteration +=1
        for i in range(ann_num):
            d = 1-IOU(ann_dims[i], centroids)
            distances.append(d)
        distances = np.array(distances)

        print ("iteration {}: dists = {}".format(iteration,np.sum(np.abs(old_distances-distances))))

        assignments = np.argmin(distances,axis=1)

        if (assignments == prev_assignments).all():
            return centroids
        #calc new centroids
        centroid_sums = np.zeros((anchor_num, anchor_dim), np.float)
        for i in range(ann_num):
            centroid_sums[assignments[i]] += ann_dims[i]
        for j in range(anchor_num):
            centroids[j] = centroid_sums[j] / (np.sum(assignments==j)+1e-6)

        prev_assignments = assignments.copy()
        old_distances = distances.copy()


def main():
    num_anchors = 5

    data = process_MOT_dataset()

    grid_w =int(608/19)
    grid_h =int(608/19)

    #run k_mean to find the num_anchors
    annotation_dims = []
    for d in data:
        cell_w = d['img_width']/grid_w
        cell_h = d['img_height']/grid_h
        for frame in d['frame']:
            for obj in d['frame'][frame]:
                xmax = obj[0] + 0.5*obj[2]
                ymax = obj[1] + 0.5*obj[3]
                xmin = obj[0] - 0.5*obj[2]
                ymin = obj[1] - 0.5*obj[3]
                relative_w = (float(xmax)- float(xmin))/cell_w
                relative_h = (float(ymax)- float(ymin))/cell_h
                temp = list(map(float, (relative_w, relative_h)))
                annotation_dims.append(temp)
    annotation_dims = np.array(annotation_dims)
    
    print(annotation_dims[0])

    centroids = run_kmeans(annotation_dims, num_anchors)

    print('\naverage IOU for', num_anchors, 'anchors:', '%0.10f' % avg_IOU(annotation_dims, centroids))
    print_anchors(centroids)



if __name__ == '__main__':
    main()
