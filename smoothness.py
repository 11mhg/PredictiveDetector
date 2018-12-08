import numpy as np
from utils import get_mot
from tqdm import tqdm

#as fast as the numpy and a better approximation
def diff2(speed, fs):
    dt_squared = pow((1./fs),2)
    speed = np.array(speed)
    s1 = speed[2:]
    s2 = speed[1:-1]
    s3 = speed[0:-2]
    out = (s1-(2*s2)+s3) / dt_squared
    return out 

#as fast as the numpy and a better approximation
def diff1(speed, fs):
    dt = 1./fs
    speed = np.array(speed)
    s1 = speed[1:]
    s2 = speed[0:-1]
    out = (s1-s2) / dt
    return out

#normal regression DLJ
def dlj(speed, fs):
    speed = np.array(speed)
    speed_peak = max(abs(speed))
    dt = 1./fs
    movement_dur = len(speed)*dt
    jerk = diff2(speed, fs) 
    scale = pow(movement_dur,5)/pow(speed_peak,2)
    return -scale*sum(pow(jerk,2))*dt

#normal LDLJ
def ldlj(speed,fs):
    ldlj_val = -np.log(abs(dlj(speed,fs)))
    ldlj_val = np.nan if ldlj_val == np.inf or ldlj_val == -np.inf else ldlj_val
    return ldlj_val

#Proposed IOU DLJ
def dlj_iou(ious, fs):
    ious = np.array(ious)
    ious_peak = np.exp(-(max(abs(ious))))
    dt = 1./fs
    movement_dur = len(ious) * dt
    jerk = diff2(ious,dt)
    scale = pow(movement_dur,5)/pow(ious_peak,2)
    return -scale*sum(pow(jerk,2))*dt

def ldlj_iou(ious, fs):
    ldlj_val = -np.log(abs(dlj_iou(ious,fs)))
#    ldlj_val = np.nan if ldlj_val == np.inf or ldlj_val == -np.inf else ldlj_val
    return ldlj_val
