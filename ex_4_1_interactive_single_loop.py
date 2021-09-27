# for running a loop from loop_over_dataset.py interactively 
# this is for ID_S4_EX1 

import os
import sys
import numpy as np
import math
import cv2
import matplotlib.pyplot as plt
import copy

## Add current working directory to path
sys.path.append(os.getcwd())

## Waymo open dataset reader
from tools.waymo_reader.simple_waymo_open_dataset_reader import utils as waymo_utils
from tools.waymo_reader.simple_waymo_open_dataset_reader import WaymoDataFileReader, dataset_pb2, label_pb2

## 3d object detection
import student.objdet_pcl as pcl
import student.objdet_detect as det
import student.objdet_eval as eval

import misc.objdet_tools as tools 
from misc.helpers import save_object_to_file, load_object_from_file, make_exec_list

# for the kalman filter 'final' project --->  
## Tracking
from student.filter import Filter
from student.trackmanagement import Trackmanagement
from student.association import Association
from student.measurements import Sensor, Measurement
from misc.evaluation import plot_tracks, plot_rmse, make_movie
import misc.params as params 
 
##################
## Set parameters and perform initializations

## Select Waymo Open Dataset file and frame numbers
data_filename = 'training_segment-1005081002024129653_5313_150_5333_150_with_camera_labels.tfrecord' # Sequence 1
# data_filename = 'training_segment-10072231702153043603_5725_000_5745_000_with_camera_labels.tfrecord' # Sequence 2
# data_filename = 'training_segment-10963653239323173269_1924_000_1944_000_with_camera_labels.tfrecord' # Sequence 3
show_only_frames = [50, 51] # show only frames in interval for debugging

datafile = WaymoDataFileReader('dataset/' + data_filename) 
datafile_iter = iter(datafile)  # initialize dataset iterator

## Initialize object detection
configs_det = det.load_configs(model_name='darknet') # options are 'darknet', 'fpn_resnet'
#configs_det = det.load_configs(model_name="fpn_resnet") 
model_det = det.create_model(configs_det)

configs_det.use_labels_as_objects = False # True = use groundtruth labels as objects, False = use model-based detection

## Uncomment this setting to restrict the y-range in the final project
# configs_det.lim_y = [-25, 25] 

## Initialize tracking
KF = Filter() # set up Kalman filter 
association = Association() # init data association
manager = Trackmanagement() # init track manager
lidar = None # init lidar sensor object
camera = None # init camera sensor object
np.random.seed(10) # make random values predictable

## Selective execution and visualization
# https://youtu.be/JLKDm3J4Ojs?t=305 
exec_detection = ['bev_from_pcl', 'detect_objects', 'validate_object_labels', 'measure_detection_performance'] #['bev_from_pcl', 'detect_objects'] # ['bev_from_pcl'] # ['bev_from_pcl', 'detect_objects', 'validate_object_labels', 'measure_detection_performance'] # options are 'bev_from_pcl', 'detect_objects', 'validate_object_labels', 'measure_detection_performance'; options not in the list will be loaded from file
exec_tracking = [] # options are 'perform_tracking'
exec_visualization = ['show_detection_performance'] # ['show_objects_in_bev_labels_in_camera'] # ['show_pcl'] # ['show_range_image'] # [] # ['show_pcl'] # options are 'show_range_image', 'show_bev', 'show_pcl', 'show_labels_in_image', 'show_objects_and_labels_in_bev', 'show_objects_in_bev_labels_in_camera', 'show_tracks', 'show_detection_performance', 'make_tracking_movie'
exec_list = make_exec_list(exec_detection, exec_tracking, exec_visualization)
vis_pause_time = 0 # set pause time between frames in ms (0 = stop between frames until key is pressed)


##################
## Perform detection & tracking over all selected frames

cnt_frame = 50  ######## this line is different to save 50 iterations of the main loop as this file only represents 1 iteration of it
all_labels = []
det_performance_all = [] 
np.random.seed(0) # make random values predictable

frame = next(datafile_iter)

#################################
## Perform 3D object detection

## Extract calibration data and front camera image from frame
lidar_name = dataset_pb2.LaserName.TOP # `1` 
camera_name = dataset_pb2.CameraName.FRONT # `1` 
lidar_calibration = waymo_utils.get(frame.context.laser_calibrations, lidar_name)        
camera_calibration = waymo_utils.get(frame.context.camera_calibrations, camera_name)

lidar_pcl = tools.pcl_from_range_image(frame, lidar_name)

lidar_bev = pcl.bev_from_pcl(lidar_pcl, configs_det)

# detections = det.detect_objects(lidar_bev, model_det, configs_det)

# ---^--- I'm thinking *that* is broken... hmmm...
detections = load_object_from_file('results/', data_filename, 'detections_' + configs_det.arch + '_' + str(configs_det.conf_thresh), cnt_frame) #EJS



valid_label_flags = tools.validate_object_labels(frame.laser_labels, lidar_pcl, configs_det, 0 if configs_det.use_labels_as_objects==True else 10)


