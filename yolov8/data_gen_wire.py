import cv2
import pickle, sys
import numpy as np
import matplotlib.pyplot as plt



data_dir='../../../recorded_data/'
# data_sets=['ER316L/wallbf_70ipm_v7_70ipm_v7/','ER316L/wallbf_150ipm_v15_150ipm_v15/',
#            'ER316L/trianglebf_100ipm_v10_100ipm_v10','ER316L/cylinderspiral_100ipm_v10/',
#            'ER316L/streaming/cylinderspiral_T19000/','ER316L/streaming/cylinderspiral_T25000/',
#            'ER316L/VPD10/cylinderspiral_50ipm_v5','ER316L/VPD10/cylinderspiral_70ipm_v7',
#            'ER316L/VPD10/cylinderspiral_110ipm_v11','ER316L/VPD10/cylinderspiral_130ipm_v13']

# data_sets=['ER316L/VPD10/tubespiral_70ipm_v7','ER316L/VPD10/tubespiral_150ipm_v15']
# data_sets=['ER316L/VPD10/tubespiral_50ipm_v5']
# data_sets=['ER316L/VPD10/tubespiral_180ipm_v18']
# data_sets=['ER316L/streaming/wall2/bf_ol_v10_f100','ER316L/streaming/right_triangle/bf_ol_v10_f100']
data_sets=['ER316L/streaming/right_triangle/bf_ol_v10_f100']

output_dir='../../../recorded_data/yolov8/tip-wire/'

num_images_per_set=10
img_counts=160
#radmonly select 10 images from each set
for data_set in data_sets:
    with open(data_dir+data_set+'/ir_recording.pickle', 'rb') as file:
        ir_recording = pickle.load(file)
    #select random frames
    idxs=np.random.choice(range(1000,len(ir_recording)), num_images_per_set, replace=False)
    for idx in idxs:
        ir_torch_tracking=np.rot90(ir_recording[idx], k=-1)
        pixel_threshold=0.77*np.max(ir_torch_tracking)
        ir_torch_tracking[ir_torch_tracking>pixel_threshold]=pixel_threshold
        ir_torch_tracking_normalized = ((ir_torch_tracking - np.min(ir_torch_tracking)) / (np.max(ir_torch_tracking) - np.min(ir_torch_tracking))) * 255
        #save image
        cv2.imwrite(output_dir+str(img_counts)+'.png', ir_torch_tracking_normalized)
        img_counts+=1


