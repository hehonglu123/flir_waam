import numpy as np
import cv2, copy
from shapely.geometry import Point, Polygon, LineString
from shapely.ops import nearest_points



##############################################FLIR COUNTS TO TEMPERATURE CONVERSION##############################################
####FROM https://flir.custhelp.com/app/answers/detail/a_id/3321/~/the-measurement-formula
def counts2temp(data_counts,R,B,F,J1,J0,Emiss):
    # reflected energy
    TRefl = 18

    # atmospheric attenuation
    TAtmC = 20
    TAtm = TAtmC + 273.15
    Tau = 0.99 #transmission

    # external optics
    TExtOptics = 20
    TransmissionExtOptics = 1
  
    K1 = 1 / (Tau * Emiss * TransmissionExtOptics)
        
    # Pseudo radiance of the reflected environment
    r1 = ((1-Emiss)/Emiss) * (R/(np.exp(B/TRefl)-F))
    # Pseudo radiance of the atmosphere
    r2 = ((1 - Tau)/(Emiss * Tau)) * (R/(np.exp(B/TAtm)-F)) 
    # Pseudo radiance of the external optics
    r3 = ((1-TransmissionExtOptics) / (Emiss * Tau * TransmissionExtOptics)) * (R/(np.exp(B/TExtOptics)-F))
            
    K2 = r1 + r2 + r3
    
    data_obj_signal = (data_counts - J0)/J1
    data_temp = (B / np.log(R/((K1 * data_obj_signal) - K2) + F)) -273.15
    
    return data_temp


def counts2temp_4learning(data_counts,R,B,F,J1,J0):
    Emiss=data_counts[-1]
    data_counts=data_counts[:-1]
    # reflected energy
    TRefl = 18

    # atmospheric attenuation
    TAtmC = 20
    TAtm = TAtmC + 273.15
    Tau = 0.99 #transmission

    # external optics
    TExtOptics = 20
    TransmissionExtOptics = 1
  
    K1 = 1 / (Tau * Emiss * TransmissionExtOptics)
        
    # Pseudo radiance of the reflected environment
    r1 = ((1-Emiss)/Emiss) * (R/(np.exp(B/TRefl)-F))
    # Pseudo radiance of the atmosphere
    r2 = ((1 - Tau)/(Emiss * Tau)) * (R/(np.exp(B/TAtm)-F)) 
    # Pseudo radiance of the external optics
    r3 = ((1-TransmissionExtOptics) / (Emiss * Tau * TransmissionExtOptics)) * (R/(np.exp(B/TExtOptics)-F))
            
    K2 = r1 + r2 + r3
    
    data_obj_signal = (data_counts - J0)/J1
    data_temp = (B / np.log(R/((K1 * data_obj_signal) - K2) + F)) -273.15
    
    return data_temp

def flame_detection_aluminum(raw_img,threshold=1.0e4,area_threshold=4,percentage_threshold=0.8):
    ###flame detection by raw counts thresholding and connected components labeling
    ###adaptively increase the threshold to % of the maximum pixel value
    threshold=max(threshold,percentage_threshold*np.max(raw_img))
    thresholded_img=(raw_img>threshold).astype(np.uint8)

    nb_components, labels, stats, centroids = cv2.connectedComponentsWithStats(thresholded_img, connectivity=4)
    
    valid_indices=np.where(stats[:, cv2.CC_STAT_AREA] > area_threshold)[0][1:]  ###threshold connected area
    if len(valid_indices)==0:
        return None, None
    
    average_pixel_values = [np.mean(raw_img[labels == label]) for label in valid_indices]   ###sorting
    valid_index=valid_indices[np.argmax(average_pixel_values)]      ###get the area with largest average brightness value

    # Extract the centroid and bounding box of the largest component
    centroid = centroids[valid_index]
    bbox = stats[valid_index, :-1]

    return centroid, bbox

def weld_detection_aluminum(raw_img,yolo_model,threshold=1.0e4,area_threshold=4,percentage_threshold=0.8):
    #centroids: x,y
    #bbox: x,y,w,h
    
    flame_centroid, flame_bbox=flame_detection_aluminum(raw_img,threshold,area_threshold,percentage_threshold)

    ## Torch detection
    torch_centroid, torch_bbox=torch_detect_yolo(raw_img,yolo_model)
    if torch_centroid is None:   #if no torch detected, return None
        return None, None, None, None
    
    return flame_centroid, flame_bbox, torch_centroid, torch_bbox


# def weld_detection_steel(raw_img,yolo_model,threshold=1.5e4,area_threshold=50,percentage_threshold=0.6):
#     ###welding point detection without flame
#     #centroids: [x,y], top pixel coordinate of the weldpool (intersection between wire and piece)
#     #bbox: x,y,w,h

#     ###adaptively increase the threshold to 60% of the maximum pixel value
#     threshold=max(threshold,percentage_threshold*np.max(raw_img))
#     thresholded_img=(raw_img>threshold).astype(np.uint8)


#     nb_components, labels, stats, centroids = cv2.connectedComponentsWithStats(thresholded_img, connectivity=4)
#     #find the largest connected area
#     areas = stats[:, 4]
#     areas[0] = 0    # Exclude the background component (label 0) from the search

#     if np.max(areas)<area_threshold:    #if no hot spot larger than area_threshold, return None
#         return None, None, None, None
    
#     # Find the index of the component with the largest area
#     largest_component_index = np.argmax(areas)
#     pixel_coordinates = np.flip(np.array(np.where(labels == largest_component_index)).T,axis=1)

#     ## Torch detection
#     torch_centroid, torch_bbox=torch_detect_yolo(raw_img,yolo_model)
#     if torch_centroid is None:   #if no torch detected, return None
#         return None, None, None, None
#     template_bottom_center=torch_bbox[:2]+np.array([torch_bbox[2]/2,torch_bbox[3]])
#     hull = cv2.convexHull(pixel_coordinates)

#     poly = Polygon([tuple(point[0]) for point in hull])
#     point = Point(template_bottom_center[0],template_bottom_center[1])
    
#     ###find the intersection between the line and the hull
#     downward_line = LineString([point, (template_bottom_center[0], 320)])
#     # Find the intersection between the line and the polygon's hull
#     intersection = poly.exterior.intersection(downward_line)

#     if not intersection.is_empty and not isinstance(intersection, LineString):
#         if intersection.geom_type == 'MultiPoint':
#             # get the average of the intersections
#             points_list = [point for point in intersection.geoms]
#             # Now you can iterate over points_list or perform operations that require iteration
#             weld_pool = Point(np.mean([p.x for p in points_list], dtype=int), np.mean([p.y for p in points_list], dtype=int))
#         else:
#             weld_pool = intersection
#     else:
#         ### no intersection, find the closest point to the line
#         weld_pool, _ = nearest_points(poly.exterior, downward_line)



#     centroid=np.array([weld_pool.x,weld_pool.y]).astype(int)
#     #create 5x5 bbox around the centroid
#     bbox=np.array([centroid[0]-2,centroid[1]-2,5,5])
    

#     # ##############################################display for debugging#########################################################
#     # #plot out the convex hull and the template bottom center
#     # # plt.plot(*poly.exterior.xy)
#     # # plt.scatter(*point.xy, c='r')
#     # # plt.plot(*downward_line.xy)
#     # # plt.show()
#     # ir_normalized = ((raw_img - np.min(raw_img)) / (np.max(raw_img) - np.min(raw_img))) * 255
#     # ir_normalized=np.clip(ir_normalized, 0, 255)
#     # # Convert the IR image to BGR format with the inferno colormap
#     # ir_bgr = cv2.applyColorMap(ir_normalized.astype(np.uint8), cv2.COLORMAP_INFERNO)
#     # cv2.rectangle(ir_bgr, tuple(bbox[:2]), tuple(bbox[:2]+bbox[2:]), (0,255,0), 2)
#     # #change convex hull vertices to blue
#     # for i in range(len(hull)):
#     #     cv2.circle(ir_bgr, tuple(hull[i][0]), 1, (255,0,0), thickness=2)
#     # #make template bottom center red
#     # cv2.circle(ir_bgr, tuple(map(int, template_bottom_center)), 1, (0,0,255), thickness=2)


#     # cv2.imshow('ir_bgr',ir_bgr)
#     # cv2.waitKey(0)
#     # ##############################################display for debugging END#########################################################


#     return centroid, bbox, torch_centroid, torch_bbox

def weld_detection_steel(raw_img,torch_model,tip_model):

    ## Torch detection
    torch_centroid, torch_bbox=torch_detect_yolo(raw_img,torch_model)
    ## Tip detection
    tip_centroid, tip_bbox=tip_wire_detect_yolo(raw_img,tip_model)

    return tip_centroid, tip_bbox, torch_centroid, torch_bbox

def form_vector(c,r,flir_intrinsic,rotated=True):
    #form a 3D vector from imaging sensor to pixel coordinate c,r
	if rotated:
		c_original=r
		r_original=flir_intrinsic['height']-c
		vector=np.array([(c_original-flir_intrinsic['c0'])/flir_intrinsic['fsx'],(r_original-flir_intrinsic['r0'])/flir_intrinsic['fsy'],1])
	else:
		vector=np.array([(c-flir_intrinsic['c0'])/flir_intrinsic['fsx'],(r-flir_intrinsic['r0'])/flir_intrinsic['fsy'],1])
	return vector/np.linalg.norm(vector)

def torch_detect(ir_image,template,template_threshold=0.3,pixel_threshold=1e4):
    ###template matching for torch, return the upper left corner of the matched region
    #threshold and normalize ir image
    ir_torch_tracking=ir_image.copy()
    ir_torch_tracking[ir_torch_tracking>pixel_threshold]=pixel_threshold
    ir_torch_tracking_normalized = ((ir_torch_tracking - np.min(ir_torch_tracking)) / (np.max(ir_torch_tracking) - np.min(ir_torch_tracking))) * 255

    # run edge detection
    edges = cv2.Canny(ir_torch_tracking_normalized.astype(np.uint8), threshold1=20, threshold2=50)
    # bolden all edges
    edges=cv2.dilate(edges,None,iterations=1)

    # cv2.imshow('ir_torch_tracking',ir_torch_tracking_normalized.astype(np.uint8))
    # cv2.waitKey(0)
    # cv2.imshow('edges',edges)
    # cv2.waitKey(0)
    

    ###template matching with normalized image
    res = cv2.matchTemplate(edges,template,cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    
    if max_val<template_threshold:
        return None
    
    return max_loc

def torch_detect_yolo(ir_image,yolo_model,pixel_threshold=1e4,percentage_threshold=0.5):
    ir_torch_tracking=copy.deepcopy(ir_image)
    pixel_threshold = max(pixel_threshold, percentage_threshold*np.max(ir_torch_tracking))
    ir_torch_tracking[ir_torch_tracking>pixel_threshold]=pixel_threshold
    ir_torch_tracking_normalized = ((ir_torch_tracking - np.min(ir_torch_tracking)) / (np.max(ir_torch_tracking) - np.min(ir_torch_tracking))) * 255
    ir_torch_tracking_normalized = ir_torch_tracking_normalized.astype(np.uint8)
    ir_torch_tracking = cv2.cvtColor(ir_torch_tracking_normalized, cv2.COLOR_GRAY2BGR)

    #run yolo
    result= yolo_model.predict(ir_torch_tracking,verbose=False, conf=0.5)[0]
    conf_all = result.boxes.conf.cpu().numpy()  #find the most confident torch prediction
    if len(conf_all)>0:
        max_conf_idx=np.argmax(conf_all)
        bbox = result.boxes.cpu().xyxy[max_conf_idx].numpy()
        centroid = np.array([(bbox[0]+bbox[2])/2,(bbox[1]+bbox[3])/2])
        #change bbox to opencv format
        bbox[2]=bbox[2]-bbox[0]
        bbox[3]=bbox[3]-bbox[1]
        return centroid, bbox.astype(int)
    else:
        return None, None
    
def tip_detect_yolo(ir_image,tip_model):
    ir_tip_tracking=copy.deepcopy(ir_image)
    ir_tip_tracking = ((ir_tip_tracking - np.min(ir_tip_tracking)) / (np.max(ir_tip_tracking) - np.min(ir_tip_tracking))) * 255
    ir_tip_tracking = ir_tip_tracking.astype(np.uint8)
    ir_tip_tracking = cv2.cvtColor(ir_tip_tracking, cv2.COLOR_GRAY2BGR)
    #run yolo
    result= tip_model.predict(ir_tip_tracking,verbose=False)[0]
    if result.boxes.cls.cpu().numpy()==0:
        bbox = result.boxes.cpu().xyxy[0].numpy()\
        #change bbox to opencv format
        bbox[2]=bbox[2]-bbox[0]
        bbox[3]=bbox[3]-bbox[1]
        centroid = np.array([(bbox[0]+bbox[2])/2,(bbox[1]+bbox[3])/2])
        return centroid, bbox.astype(int)
    else:
        return None, None

def tip_wire_detect_yolo(ir_image,tip_wire_model):
    ir_tip_wire_tracking=copy.deepcopy(ir_image)
    pixel_threshold=0.77*np.max(ir_tip_wire_tracking)
    ir_tip_wire_tracking[ir_tip_wire_tracking>pixel_threshold]=pixel_threshold
    ir_tip_wire_tracking = ((ir_tip_wire_tracking - np.min(ir_tip_wire_tracking)) / (np.max(ir_tip_wire_tracking) - np.min(ir_tip_wire_tracking))) * 255
    ir_tip_wire_tracking = ir_tip_wire_tracking.astype(np.uint8)
    ir_tip_wire_tracking = cv2.cvtColor(ir_tip_wire_tracking, cv2.COLOR_GRAY2BGR)
    #run yolo
    result= tip_wire_model.predict(ir_tip_wire_tracking,verbose=False,conf=0.5)[0]
    cls_list = result.boxes.cls.cpu().numpy().astype(int)
    counts = np.bincount(cls_list)
    # Check if there is exactly one 0 and one 1
    if len(counts) >= 2 and counts[0] == 1 and counts[1] == 1:
        idx0=np.where(cls_list==0)[0][0]
        idx1=np.where(cls_list==1)[0][0]
        #get the lower right corner of the bounding box of cls 0
        x0_left,y0_left,x1_left,y1_left = result.boxes.cpu().xyxy[idx0].numpy()
        #get the lower left corner of the bounding box of cls 1
        x0_right,y0_right,x1_right,y1_right = result.boxes.cpu().xyxy[idx1].numpy()
        #get the average of the two points
        centroid_x = (x1_left+x0_right)/2
        centroid_y = (y1_left+y1_right)/2
        
        return np.array([centroid_x,centroid_y]), (int(centroid_x - 1.5), int(centroid_y - 1.5), 3, 3)

    return None, None

def get_pixel_value(ir_image,coord,window_size):
    ###get pixel value larger than avg within the window
    window = ir_image[coord[1]-window_size//2:coord[1]+window_size//2+1,coord[0]-window_size//2:coord[0]+window_size//2+1]
    pixel_avg = np.mean(window)
    mask = (window > pixel_avg) # filter out background 
    pixel_avg = np.mean(window[mask])
    return pixel_avg

# def line_intersect(p1,v1,p2,v2):
#     #calculate the intersection of two lines, on line 1
#     #find the closest point on line1 to line2
#     w = p1 - p2
#     a = np.dot(v1, v1)
#     b = np.dot(v1, v2)
#     c = np.dot(v2, v2)
#     d = np.dot(v1, w)
#     e = np.dot(v2, w)

#     sc = (b*e - c*d) / (a*c - b*b)
#     closest_point = p1 + sc * v1

#     return closest_point