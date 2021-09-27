# ---------------------------------------------------------------------
# Project "Track 3D-Objects Over Time"
# Copyright (C) 2020, Dr. Antje Muntzinger / Dr. Andreas Haja.
#
# Purpose of this file : Evaluate performance of object detection
#
# You should have received a copy of the Udacity license together with this program.
#
# https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
# ----------------------------------------------------------------------
#

# general package imports
from student.objdet_detect import detections_by_row
import numpy as np
import matplotlib
matplotlib.use('wxagg') # change backend so that figure maximizing works on Mac as well     
import matplotlib.pyplot as plt

import torch
from shapely.geometry import Polygon, Point 
from operator import itemgetter

# add project directory to python path to enable relative imports
import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

# object detection tools and helper functions
import misc.objdet_tools as tools


# compute various performance measures to assess object detection
def measure_detection_performance(detections, labels, labels_valid, configs): #min_iou=0.5):
    
     # find best detection for each valid label 
    true_positives = 0 # no. of correctly detected objects
    center_devs = []
    ious = []
    # print () 
    # print (f"type(labels) {type(labels)}")   # type(labels) <class 'google.protobuf.pyext._message.RepeatedCompositeContainer'>
    # print () 
    # print (f"type(labels_valid) {type(labels_valid)}") # type(labels_valid) <class 'numpy.ndarray'> 
    # print () 
    # print (f"labels {labels}") 
    # print () 
    # print (f"labels_valid {labels_valid}")
    # labels_valid [False False False False False  True False  True False False False False
    # True False False False False]
    # print () 
    # input ('get back to work')
    #matches_lab_det = dict([]) # why did they intentionally give broken code & make this a **list** inside the loop when it should be a dictionary outside the loop????????!!!
    # matches_lab_det = [] # it isn't a dict they just put in the wrong place on purpose ...   
    for label, valid in zip(labels, labels_valid):
        matches_lab_det = []
        if valid: # exclude all labels from statistics which are not considered valid
            
            # compute intersection over union (iou) and distance between centers

            ####### ID_S4_EX1 START #######     
            #######
            print("student task ID_S4_EX1 ")

            ## step 1 : extract the four corners of the current label bounding-box
            # print () 
            # print ()
            # print (f"label {label}") 
            # label box {
            # center_x: 29.06191018172467
            # center_y: 0.7503580339998734
            # center_z: 0.8711946193956237
            # width: 2.035441412505256
            # length: 4.307956063242486
            # height: 1.9099999999999966
            # heading: -0.01140328283988623
            # }
            # metadata {
            # speed_x: 16.194098193945763
            # speed_y: 0.19548010062521826
            # accel_x: -0.4126408362878919
            # accel_y: -0.11501838135075103
            # }
            # type: TYPE_VEHICLE
            # id: "mV--0dAvjt4OyGlzlXvfPw"
            # print (f"type(label) {type(label)}")
            # type(label) <class 'simple_waymo_open_dataset_reader.label_pb2.Label'>

            # print ()
            # print ()
            
            # from line 324 & 325 & 290-291 of objdet_tools.py: 
            candidate = [label.type, label.box.center_x, label.box.center_y, label.box.center_z,
                         label.box.height, label.box.width, label.box.length, label.box.heading] 
            _, x, y, z, _, w, l, yaw = candidate
            # It seems like label is in meters instead of pixels, let's try converting it. . . 
            ########################################################################################
            #_id, _x, _y, _z, _h, _w, _l, _yaw = detections 

            # print (f"type(y): {type(y)}")
            # input ('what is y"s type?') # type(y): <class 'float'>

            # convert from metric into pixel coordinates


            # EXPERIMENTAL CODE!!!!!!!!!!!!!!!!!!!!!!!!!

            # x = (x - configs.lim_x[0]) / (configs.lim_x[1] - configs.lim_x[0]) * configs.bev_height 
            # y = (y - configs.lim_y[0]) / (configs.lim_y[1] - configs.lim_y[0]) * configs.bev_width

            # EXPERIMENTAL CODE!!!!!!!!!!!!!!!!!!!!!!!!!


            x_ = (y - configs.lim_y[0]) / (configs.lim_y[1] - configs.lim_y[0]) * configs.bev_width # why are we making x in terms of y and y in terms of x??? 
            y_ = (x - configs.lim_x[0]) / (configs.lim_x[1] - configs.lim_x[0]) * configs.bev_height
            z = z - configs.lim_z[0]
            w = w / (configs.lim_y[1] - configs.lim_y[0]) * configs.bev_width
            l = l / (configs.lim_x[1] - configs.lim_x[0]) * configs.bev_height
            yaw = -yaw

            x = x_
            y = y_ 

            #--------------------------

            # x = (x - configs.lim_x[0]) / (configs.lim_x[1] - configs.lim_x[0]) * configs.bev_width  # why are we making x in terms of y and y in terms of x??? 

            ########################################################################################

            # print (f"x: {x}")
            # print (f"y: {y}")
            # print (f"z: {z}")  
            # print (f"w: {w}")
            # print (f"l: {l}")
            # print (f"yaw: {yaw}")

            # x: 313.04114946004813
            # y: 3806.5803774341857
            # z: 1.8929607095304846
            # w: 24.750967576049547
            # l: 52.38474572902863
            # yaw: 0.010818934012628567
            
            #label_obj_corners = compute_box_corners(x,y,w,l,yaw)
            label_obj_corners = tools.compute_box_corners(x,y,w,l,yaw) # from tools.is_label_inside_detection_area() EJS 
            label_obj_poly = Polygon(label_obj_corners) # from tools.is_label_inside_detection_area()  EJS 
            label_center_point_xyz = Point(x, y, z) 

            # print (f"label_obj_corners: {label_obj_corners}")
            # print (f"detections: {detections}")
            # input('press whatever you want to conintue, bob') 

            # so to me at the **moment** label_obj_corners... it looks like **all** the "x" coordinates (are these tuples xy-coordinates?) 
            # look like they are probably spot on, e.g. 340, 365. BUT, *all* the "y" coordinates look like they are off by about 4,000. 
            # e.g. 4321.xxx, 4266.xxx, etc.  

            


            # label_obj_corners: [(340.8322294236808, 4321.364901888402), (340.8657878837116, 4266.48519327646), (365.40667696637615, 4266.500199814692), (365.37311850634535, 4321.379908426633)]
            # detections: [[1, tensor(313.0328), tensor(353.7452), 0.0, 1.5, tensor(24.0984), tensor(49.5910), tensor(-0.0037)], [1, tensor(350.6573), tensor(218.0103), 0.0, 1.5, tensor(23.0669), tensor(51.8288), tensor(0.0017)], [1, tensor(353.3030), tensor(600.6396), 0.0, 1.5, tensor(23.5785), tensor(51.1130), tensor(0.0515)]]
            # press whatever you want to conintue, bob


            
            ## step 2 : loop over all detected objects

            # print () 
            # print () 
            # print (f"len(detections): {len(detections)}")
            # print ()
            # print (f"detections: {detections}")

            # len(detections): 3

            # detections: [[1, tensor(2961.2461), tensor(4271.3052), 1.0, 1.5, tensor(280.6897), tensor(632.9459), tensor(0.0072)], 
            # [1, tensor(4588.4912), tensor(3807.2178), 1.0, 1.5, tensor(294.5393), tensor(579.0916), tensor(-0.0052)], 
            # [1, tensor(7604.0771), tensor(4295.9116), 1.0, 1.5, tensor(286.0607), tensor(618.6083), tensor(-0.0680)]]
            # print ()
            # print () 
            # input ("you can quit & get back to programming now")
            
            for each_object_detected in detections: 

                ## step 3 : extract the four corners of the current detection
                #object = [_id, x, y, z, _h, w, l, yaw] # another line of code i wrote somewhere else -- EJS 
                _, x, y, z, _, w, l, yaw = each_object_detected

                # print (f"x: {x}")
                # print (f"y: {y}")
                # print (f"z: {z}")
                # print (f"w: {w}")
                # print (f"l: {l}")
                # print (f"yaw: {yaw}")
               
                
                cur_detection_obj_corners = tools.compute_box_corners(x,y,w,l,yaw) # here are the 4 corners requested 
                # print (f"label_obj_corners: {label_obj_corners}")
                # print (f"cur_detection_obj_corners: {cur_detection_obj_corners}")
                # input ('proceed at your own risk!!!') 
                point_xyz_detection = Point(x, y, z) # center point of this detection 

                #---^--- the #'s for cur_detection_obj_corners seem far, far, far, far off...  

                # websites that might help [tomorrow] . . . 
                # https://programmerah.com/using-shapely-geometry-polygon-to-calculate-the-iou-of-any-two-quadrilaterals-28395/
                # https://codereview.stackexchange.com/questions/204017/intersection-over-union-for-rotated-rectangles 
                
                ## step 4 : compute the center distance between label and detection bounding-box in x, y, and z

                center_distance = point_xyz_detection.distance(label_center_point_xyz) # hypothetically the center distance between label and detection bounding-box in x, y, and z

                # EJS: idea: if the distance between label & detection bounding box in x, y, & z is greater than the length
                # of either "vehicle" (length of either bounding box), then the *intersection* is *zero* which means *iou* is zero. 

                # for 'length' of vehicle, i'll stay in the xy-plane as does `tools.compute_box_corners` but instead of `l` i'll use diagonal length
                # of the vehicle / label boxes returned by `tools.compute_box_corners` -- EJS 

                _, rl, _, fr = cur_detection_obj_corners # four tuples: xy coordinates of the front left, rear left, rear right, front right 
                vehicle_length = Point(rl).distance(Point(fr)) 

                _, rl, _, fr = label_obj_corners 
                label_length = Point(rl).distance(Point(fr)) 

                longest_length = max(vehicle_length, label_length) # whichever is longer is "longest" 

                # print (f"longest_length: {longest_length}")
                # print (f"point_xyz_detection.distance(label_center_point_xyz): {point_xyz_detection.distance(label_center_point_xyz):}")
                

                # longest_length: 60.116877876896254
                # point_xyz_detection.distance(label_center_point_xyz): 3940.3912839559107 
                # it looks like taht distance is off by about 4000 (minus the "length" of the vehicle as i called it) 

                #input ('error "bout to happen!') 


                ##### EXPERIMENTALLY NOW!!!!!!!!!!!!!!! ################# 
                # if longest_length < point_xyz_detection.distance(label_center_point_xyz): # definitely they cannot overlap in the plane then 
                #     # print (f"longest_length: {longest_length}")
                #     # print (f"point_xyz_detection.distance(label_center_point_xyz): {point_xyz_detection.distance(label_center_point_xyz)}")
                #     iou = 0 
                #     # input ('keep trying')

                # else: 
                
                ##### EXPERIMENTALLY NOW!!!!!!!!!!!!!!! ################# 
                    
                cur_det_obj_poly = Polygon(cur_detection_obj_corners) 
                _i = cur_det_obj_poly.intersection(label_obj_poly).area  
                _u = cur_det_obj_poly.union(label_obj_poly).area  
                iou = (_i / (_u + 1e-16))
                # https://knowledge.udacity.com/questions/677101 
                # g <- gt_area 
                # p <- pred_area 
                # g = cur_det_obj_poly.area 
                # p = label_obj_poly.area 
                # iou = _i / (g + p - _i + 1e-16) # i don't think this changes anything & i like my above answer better . . . 

                # print(f"_i: {_i}") # _i: POLYGON EMPTY
                # input('continue continue continue continue')
                
                ## step 5 : compute the intersection over union (IOU) between label and detection bounding-box
                
                ## step 6 : if IOU exceeds min_iou threshold, store [iou,dist_x, dist_y, dist_z] in matches_lab_det and increase the TP count

                if iou > configs.min_iou: 
                    dist_x = point_xyz_detection.x - label_center_point_xyz.x 
                    dist_y = point_xyz_detection.y - label_center_point_xyz.y
                    dist_z = point_xyz_detection.z - label_center_point_xyz.z 
                    # dist_x = label_center_point_xyz.x # AttributeError: 'float' object has no attribute 'x' # LOL!!!! 
                    # dist_y = label_center_point_xyz.y 
                    # dist_z = label_center_point_xyz.z 
                    matches_lab_det.append([iou, dist_x, dist_y, dist_z]) # center_distance.x, center_distance.y, center_distance.z]) 
                    true_positives += 1 # increase the TP count
                
            #######
            ####### ID_S4_EX1 END #######     
            
        # find best match and compute metrics
        if matches_lab_det: # reassigned to `[]` at each iteration of the containing for() loop ...   no... that **has to be wrong** they intentionally provided wrong code 
            best_match = max(matches_lab_det,key=itemgetter(0))  #1)) # retrieve entry with max iou in case of multiple candidates   ## I changed it to key=itemgetter(0) -- EJS 
            # print (f"best_match: {best_match}")
            # print ()
            # input ('keep trying') 
            #best_match: [0.8658362183256311, 0.18358635575276594, -2.1950195030926807, -2.0292643213596193] 
            ious.append(best_match[0])
            center_devs.append(best_match[1:])

    # print ()

    # print (f"ious: {ious}")
    # print (f"center_devs: {center_devs}")
    # print () 
    # input ('was it right???')
    # print () 
    # ious: []
    # center_devs: []

    # was it right???

    ####### ID_S4_EX2 START #######     
    #######
    print("student task ID_S4_EX2")
    
    # compute positives and negatives for precision/recall
    
    ## step 1 : compute the total number of positives present in the scene
    all_positives = 0 
    all_positives = np.array(labels_valid).sum()  
    

    ## step 2 : compute the number of false negatives
    false_negatives = 0
    for label, valid in zip(labels, labels_valid):
        if valid:
            candidate = [label.type, label.box.center_x, label.box.center_y, label.box.center_z,
                         label.box.height, label.box.width, label.box.length, label.box.heading] 
            _, x, y, z, _, w, l, yaw = candidate
            x_ = (y - configs.lim_y[0]) / (configs.lim_y[1] - configs.lim_y[0]) * configs.bev_width # why are we making x in terms of y and y in terms of x??? 
            y_ = (x - configs.lim_x[0]) / (configs.lim_x[1] - configs.lim_x[0]) * configs.bev_height
            z = z - configs.lim_z[0]
            w = w / (configs.lim_y[1] - configs.lim_y[0]) * configs.bev_width
            l = l / (configs.lim_x[1] - configs.lim_x[0]) * configs.bev_height
            yaw = -yaw
            x = x_
            y = y_ 
            label_obj_corners = tools.compute_box_corners(x,y,w,l,yaw) # from tools.is_label_inside_detection_area() EJS 
            label_obj_poly = Polygon(label_obj_corners) # from tools.is_label_inside_detection_area()  EJS 
            FN = True 
            for each_object_detected in detections: 
                _, x, y, z, _, w, l, yaw = each_object_detected
                cur_detection_obj_corners = tools.compute_box_corners(x,y,w,l,yaw) 
                cur_det_obj_poly = Polygon(cur_detection_obj_corners) 
                _i = cur_det_obj_poly.intersection(label_obj_poly).area  
                _u = cur_det_obj_poly.union(label_obj_poly).area  
                iou = (_i / (_u + 1e-16))
                if iou > configs.min_iou: 
                    FN = False 
                    break 
            if FN:
                false_negatives += 1 

    ## step 3 : compute the number of false positives
    false_positives = 0
    for each_object_detected in detections: 
        _, x, y, z, _, w, l, yaw = each_object_detected
        cur_detection_obj_corners = tools.compute_box_corners(x,y,w,l,yaw) 
        cur_det_obj_poly = Polygon(cur_detection_obj_corners)
        FP = True 
        for label, valid in zip(labels, labels_valid):
            if valid:
                candidate = [label.type, label.box.center_x, label.box.center_y, label.box.center_z,
                            label.box.height, label.box.width, label.box.length, label.box.heading] 
                _, x, y, z, _, w, l, yaw = candidate
                x_ = (y - configs.lim_y[0]) / (configs.lim_y[1] - configs.lim_y[0]) * configs.bev_width # why are we making x in terms of y and y in terms of x??? 
                y_ = (x - configs.lim_x[0]) / (configs.lim_x[1] - configs.lim_x[0]) * configs.bev_height
                z = z - configs.lim_z[0]
                w = w / (configs.lim_y[1] - configs.lim_y[0]) * configs.bev_width
                l = l / (configs.lim_x[1] - configs.lim_x[0]) * configs.bev_height
                yaw = -yaw
                x = x_
                y = y_ 
                label_obj_corners = tools.compute_box_corners(x,y,w,l,yaw) # from tools.is_label_inside_detection_area() EJS 
                label_obj_poly = Polygon(label_obj_corners) # from tools.is_label_inside_detection_area()  EJS 
                _i = cur_det_obj_poly.intersection(label_obj_poly).area  
                _u = cur_det_obj_poly.union(label_obj_poly).area  
                iou = (_i / (_u + 1e-16))
                if iou > configs.min_iou: 
                    FP = False 
                    break 
        if FP:
            false_positives += 1 






    
    #######
    ####### ID_S4_EX2 END #######     
    
    pos_negs = [all_positives, true_positives, false_negatives, false_positives]
    det_performance = [ious, center_devs, pos_negs]

    # print (f"checking ID_S4_EX2")
    # print (f"pos_negs: {pos_negs}")
    # print (f"det_performance: {det_performance}")
    # input ('so sad') 
    
    return det_performance


# evaluate object detection performance based on all frames
def compute_performance_stats(det_performance_all):

    # extract elements
    ious = []
    center_devs = []
    pos_negs = []
    for item in det_performance_all:
        ious.append(item[0]) # a list of ious at each entry, mine will have 3 cause my ious are somehow magically too high... 
        center_devs.append(item[1]) # i think this will be a list of list of list 
        pos_negs.append(item[2]) 
    
    ####### ID_S4_EX3 START #######     
    #######    
    print('student task ID_S4_EX3')

    ## step 1 : extract the total number of positives, true positives, false negatives and false positives
    total_positives = 0 
    true_positives = 0 
    false_negatives = 0 
    false_positives = 0 
    for pos_neg_list in pos_negs: # each pos_neg_list will look like: [all_positives, true_positives, false_negatives, false_positives] 
        total_positives += pos_neg_list[0] 
        true_positives += pos_neg_list[1] 
        false_negatives += pos_neg_list[2] 
        false_positives += pos_neg_list[3] 

    
    ## step 2 : compute precision
    precision = 0.0
    precision = true_positives / (true_positives + false_positives) 

    ## step 3 : compute recall 
    recall = 0.0 
    recall =  true_positives  / total_positives 

    #######    
    ####### ID_S4_EX3 END #######     
    print('precision = ' + str(precision) + ", recall = " + str(recall))   

    input ('precision = 0.996, recall = 0.8137254901960784???????')



    # serialize intersection-over-union and deviations in x,y,z
    ious_all = [element for tupl in ious for element in tupl]
    devs_x_all = []
    devs_y_all = []
    devs_z_all = []
    for tuple in center_devs:
        for elem in tuple:
            dev_x, dev_y, dev_z = elem
            devs_x_all.append(dev_x)
            devs_y_all.append(dev_y)
            devs_z_all.append(dev_z)
    

    # compute statistics
    stdev__ious = np.std(ious_all)
    mean__ious = np.mean(ious_all)

    stdev__devx = np.std(devs_x_all)
    mean__devx = np.mean(devs_x_all)

    stdev__devy = np.std(devs_y_all)
    mean__devy = np.mean(devs_y_all)

    stdev__devz = np.std(devs_z_all)
    mean__devz = np.mean(devs_z_all)
    #std_dev_x = np.std(devs_x)

    # plot results
    data = [precision, recall, ious_all, devs_x_all, devs_y_all, devs_z_all]
    titles = ['detection precision', 'detection recall', 'intersection over union', 'position errors in X', 'position errors in Y', 'position error in Z']
    textboxes = ['', '', '',
                 '\n'.join((r'$\mathrm{mean}=%.4f$' % (np.mean(devs_x_all), ), r'$\mathrm{sigma}=%.4f$' % (np.std(devs_x_all), ), r'$\mathrm{n}=%.0f$' % (len(devs_x_all), ))),
                 '\n'.join((r'$\mathrm{mean}=%.4f$' % (np.mean(devs_y_all), ), r'$\mathrm{sigma}=%.4f$' % (np.std(devs_y_all), ), r'$\mathrm{n}=%.0f$' % (len(devs_x_all), ))),
                 '\n'.join((r'$\mathrm{mean}=%.4f$' % (np.mean(devs_z_all), ), r'$\mathrm{sigma}=%.4f$' % (np.std(devs_z_all), ), r'$\mathrm{n}=%.0f$' % (len(devs_x_all), )))]

    f, a = plt.subplots(2, 3)
    a = a.ravel()
    num_bins = 20
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    for idx, ax in enumerate(a):
        ax.hist(data[idx], num_bins)
        ax.set_title(titles[idx])
        if textboxes[idx]:
            ax.text(0.05, 0.95, textboxes[idx], transform=ax.transAxes, fontsize=10,
                    verticalalignment='top', bbox=props)
    plt.tight_layout()
    plt.show()

