from preprocess import *
from new_generator import *

with open('model_data/mot_anchors.txt') as f:
    anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    anchors = np.array(anchors).reshape(-1,2)

data = process_MOT_dataset()
data = get_all_detector_masks(data[0:1],anchors)

gen = VideoSequence(data,5,5)
gen.__getitem__(1)
