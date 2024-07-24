import ultralytics
from ultralytics import YOLO
import cv2
import pickle, sys
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon
from shapely.ops import nearest_points


if __name__ == "__main__":
    
    # Load the IR recording data from the pickle file
    # data_dir='../../../recorded_data/ER316L/wallbf_100ipm_v10_100ipm_v10/'
    data_dir='../../../recorded_data/ER316L/streaming/cylinderspiral_T22222/'
    # data_dir='../../../recorded_data/ER4043/wallbf_100ipm_v10_70ipm_v7/'


    with open(data_dir+'/ir_recording.pickle', 'rb') as file:
        ir_recording = pickle.load(file)
    ir_ts=np.loadtxt(data_dir+'/ir_stamps.csv', delimiter=',')

    # Set the colormap (inferno) and normalization range for the color bar
    cmap = cv2.COLORMAP_INFERNO

    frame=5555
    ir_torch_tracking = np.rot90(ir_recording[frame], k=-1)
    # pixel_threshold=1e4    
    pixel_threshold=0.8*np.max(ir_torch_tracking)
    ir_torch_tracking[ir_torch_tracking>pixel_threshold]=pixel_threshold
    ir_torch_tracking_normalized = ((ir_torch_tracking - np.min(ir_torch_tracking)) / (np.max(ir_torch_tracking) - np.min(ir_torch_tracking))) * 255
    ir_torch_tracking_normalized = ir_torch_tracking_normalized.astype(np.uint8)
    ir_torch_tracking = cv2.cvtColor(ir_torch_tracking_normalized, cv2.COLOR_GRAY2BGR)

    #run yolo
    model = YOLO("tip_wire.pt")
    # results = yolo(ir_torch_tracking_normalized)
    # results= model.predict("datasets/torch_yolov8_data/train/images/30_png.rf.0971a2c2d86bf76299a739b019758aa0.jpg",imgsz=320)
    result= model.predict(ir_torch_tracking,conf=0.5)[0]

    boxes = result.boxes  # Boxes object for bounding box outputs
    #convert to numpy
    # print(result.boxes.cls.cpu().numpy())
    print(boxes.cpu().xyxy[0].numpy())
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    result.show()  # display to screen


    cls_list = result.boxes.cls.cpu().numpy().astype(int)
    counts = np.bincount(cls_list)
    # Check if there is exactly one 0 and one 1
    if len(counts) >= 2 and counts[0] == 1 and counts[1] == 1:
        idx0=np.where(cls_list==0)[0][0]
        idx1=np.where(cls_list==1)[0][0]
        #get the lower right corner of the bounding box of cls 0
        x0_left,y0_left,x1_left,y1_left = boxes.cpu().xyxy[idx0].numpy()
        #get the lower left corner of the bounding box of cls 1
        x0_right,y0_right,x1_right,y1_right = boxes.cpu().xyxy[idx1].numpy()
        #get the average of the two points
        centroid_x = (x1_left+x0_right)/2
        centroid_y = (y1_left+y1_right)/2
        #draw a 3x3 bounding box around the centroid
        cv2.rectangle(ir_torch_tracking, (int(centroid_x)-1, int(centroid_y)-1), (int(centroid_x)+1, int(centroid_y)+1), (0, 255, 0), 2)
        cv2.imshow('ir_torch_tracking', ir_torch_tracking)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

