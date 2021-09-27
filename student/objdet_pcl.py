# ---------------------------------------------------------------------
# Project "Track 3D-Objects Over Time"
# Copyright (C) 2020, Dr. Antje Muntzinger / Dr. Andreas Haja.
#
# Purpose of this file : Process the point-cloud and prepare it for object detection
#
# You should have received a copy of the Udacity license together with this program.
#
# https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
# ----------------------------------------------------------------------
#

# general package imports
import cv2
import numpy as np
import torch

# Ezra's imports 
import zlib 
import open3d as o3d

# add project directory to python path to enable relative imports
import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

# waymo open dataset reader
from tools.waymo_reader.simple_waymo_open_dataset_reader import utils as waymo_utils
from tools.waymo_reader.simple_waymo_open_dataset_reader import dataset_pb2, label_pb2

# object detection tools and helper functions
import misc.objdet_tools as tools


# visualize lidar point-cloud
def show_pcl(pcl):

    ####### ID_S1_EX2 START #######     
    #######
    print("student task ID_S1_EX2")

    # step 1 : initialize open3d with key callback and create window
    #vis = o3d.Visualizer() # s.l.o.c. from : https://www.programcreek.com/python/example/110517/open3d.Vector3dVector 
    #vis.create_window("3D Map")  # s.l.o.c. from : https://www.programcreek.com/python/example/110517/open3d.Vector3dVector  
    # ---^--- AttributeError: module 'open3d' has no attribute 'Visualizer' 
    # step 2 : create instance of open3d point-cloud class
    pcd = o3d.geometry.PointCloud() # from: GitHub for classroom materials https://github.com/udacity/nd013-c2-fusion-exercises/blob/main/lesson-1-lidar-sensor/examples/l1_examples.py

    # step 3 : set points in pcd instance by converting the point-cloud into 3d vectors (using open3d function Vector3dVector)
    # print (type(pcl)) # <class 'numpy.ndarray'>
    # print (pcl.shape) # (135174, 4) ... darn pickle 
    # print ((pcl < 0).any()) # True ... well... gosh darnit! 
    # print ((pcl < 0).sum()) # 158810 # well, if they are less than 0 how many are? ans is about 29% of them... 
    # print (pcl[0:100, 0:4]) # looks very similar to print (pcl[0:100, 0:3]) i wrote in `l1_examples.py` from github exercises 
    pcl = pcl[:, 0:3] # throw away the last column because it seemed extraneous 




    pcd.points = o3d.utility.Vector3dVector(pcl) # from: GitHub for classroom materials https://github.com/udacity/nd013-c2-fusion-exercises/blob/main/lesson-1-lidar-sensor/examples/l1_examples.py 
    # step 4 : for the first frame, add the pcd instance to visualization using add_geometry; for all other frames, use update_geometry instead
    
    # step 5 : visualize point cloud and keep window open until right-arrow is pressed (key-code 262)



    #### HMMMM???
    o3d.visualization.draw_geometries([pcd]) 

    input("Press Enter to continue...") # https://stackoverflow.com/questions/983354/how-to-make-a-script-wait-for-a-pressed-key 
    # from: GitHub for classroom materials https://github.com/udacity/nd013-c2-fusion-exercises/blob/main/lesson-1-lidar-sensor/examples/l1_examples.py
    #######
    ####### ID_S1_EX2 END #######     
       

# visualize range image
def show_range_image(frame, lidar_name):

    ####### ID_S1_EX1 START #######     
    #######
    print("student task ID_S1_EX1")

    # step 1 : extract lidar data and range image for the roof-mounted lidar
    ## EJS --> the following lines of code this EJS-block from Example C1-5-1 `load_range_image` 
    # hopefully lidar_name is the roof-mounted lidar! 
    lidar = [obj for obj in frame.lasers if obj.name == lidar_name][0] # get laser data structure from frame
    ri = []
    if len(lidar.ri_return1.range_image_compressed) > 0: # use first response
        ri = dataset_pb2.MatrixFloat()
        ri.ParseFromString(zlib.decompress(lidar.ri_return1.range_image_compressed))
        ri = np.array(ri.data).reshape(ri.shape.dims)
    ## <-- EJS 
    
    # step 2 : extract the range and the intensity channel from the range image
    ## EJS --> 
    # next 2 lines of code from the classroom: https://bit.ly/3ka2hsG 
    ri_range = ri[:,:,0]
    ri_intensity = ri[:, :, 1] 
    ## <-- EJS 
    
    # step 3 : set values <0 to zero
    ## EJS -->
    # next 1 lines of code from the classroom: https://bit.ly/3ka2hsG 
    ri[ri < 0] = 0.0 
    ## <-- EJS 

    # step 4 : map the range channel onto an 8-bit scale and make sure that the full range of values is appropriately considered
    ## EJS -->
    ri_range = ri_range * 255 / (np.amax(ri_range) - np.amin(ri_range))
    img_range = ri_range.astype(np.uint8)
    ## <-- EJS 

    # EJS: 
    # the rubric says, Crop range image to +/- 90 deg. left and right of the forward-facing x-axis, however, I think they mean
    # +/- 45 deg as that was what was referenced as per the Waymo paper in the Classroom materials 
    deg45 = int(img_range.shape[1] / 8) # s.l.o.c. from the Udacity Classroom here: https://bit.ly/3zogPt1 
    ri_center = int(img_range.shape[1]/2) # s.l.o.c. from the Udacity Classroom here: https://bit.ly/3zogPt1 
    img_range = img_range[:,ri_center-deg45:ri_center+deg45] # s.l.o.c. from the Udacity Classroom here: https://bit.ly/3zogPt1  
    img_range = img_range.astype(np.uint8) # probably don't need this, added it just in case <--- ejs 

    # step 5 : map the intensity channel onto an 8-bit scale and normalize with the difference between the 1- and 99-percentile 
    # to mitigate the influence of outliers
    # now that ri[ri < 0] "==" 0.0, 1st percentile is going to equal np.min(ri_intesity) "==" 0.0 
    # so just use ri_intensity_99 == ri_intensity_99 - ri_intensity_00 
    # ri_intensity_99 = np.percentile(ri_intensity, 99)  # https://numpy.org/doc/stable/reference/generated/numpy.percentile.html 
    # ri_intensity /= ri_intensity_99 
    # img_intensity = ri_intensity.astype(np.uint8)  
    ri_intensity = np.amax(ri_intensity)/2 * ri_intensity * 255 / (np.amax(ri_intensity) - np.amin(ri_intensity))
    img_intensity = ri_intensity.astype(np.uint8)

    # following 3 lines of code from the classroom at: https://bit.ly/3kqZ2gJ 
    deg45 = int(img_intensity.shape[1] / 8)
    ri_center = int(img_intensity.shape[1]/2)
    img_intensity = img_intensity[:,ri_center-deg45:ri_center+deg45] 
    
    # step 6 : stack the range and intensity image vertically using np.vstack and convert the result to an unsigned 8-bit integer
    img_range_intensity = np.vstack((img_range, img_intensity)) 
    
    #img_range_intensity = [] # remove after implementing all steps
    #######
    ####### ID_S1_EX1 END #######     
    
    return img_range_intensity


# create birds-eye view of lidar data
def bev_from_pcl(lidar_pcl, configs):

    # remove lidar points outside detection area and with too low reflectivity
    mask = np.where((lidar_pcl[:, 0] >= configs.lim_x[0]) & (lidar_pcl[:, 0] <= configs.lim_x[1]) &
                    (lidar_pcl[:, 1] >= configs.lim_y[0]) & (lidar_pcl[:, 1] <= configs.lim_y[1]) &
                    (lidar_pcl[:, 2] >= configs.lim_z[0]) & (lidar_pcl[:, 2] <= configs.lim_z[1]))
    lidar_pcl = lidar_pcl[mask] # e.g. lidar_pcl.shape|:>(148457, 4)-->(65821, 4) 
    
    # shift level of ground plane to avoid flipping from 0 to 255 for neighboring pixels
    lidar_pcl[:, 2] = lidar_pcl[:, 2] - configs.lim_z[0]  

    # convert sensor coordinates to bev-map coordinates (center is bottom-middle)
    ####### ID_S2_EX1 START #######     
    #######
    print("student task ID_S2_EX1")

    ## step 1 :  compute bev-map discretization by dividing x-range by the bev-image height (see configs) 
    # print ()
    # print ('--------------------------------------------')
    # print (f"lim_x {configs.lim_x}") 
    # print (f"lim_y {configs.lim_y}")
    # print (f"lim_z {configs.lim_z}")
    # print ('--------------------------------------------')
    # print () 
    # print ('what is in this config?') 
    # print (configs)
    # print ()
    # print ('--------------------------------------------')

    configs.bev_width # 608
    configs.bev_height # 608 # <--- bev image height 
    bev_discret = (configs.lim_x[1] - configs.lim_x[0]) / configs.bev_height # from Udacity classroom https://bit.ly/3kjcuDc 

    ## step 2 : create a copy of the lidar pcl and transform all metrix x-coordinates into bev-image coordinates   
    lidar_pcl_cpy = np.copy(lidar_pcl) # from Udacity classroom: https://bit.ly/2XvPsQN 
    lidar_pcl_cpy[:, 0] = np.int_(np.floor(lidar_pcl_cpy[:, 0] / bev_discret))  # from Udacity classroom: https://bit.ly/2XvPsQN  

    # step 3 : perform the same operation as in step 2 for the y-coordinates but make sure that no negative bev-coordinates occur
    # transform all matrix y-coordinates as well but center the foward-facing x-axis on the middle of the image
    lidar_pcl_cpy[:, 1] = np.int_(np.floor(lidar_pcl_cpy[:, 1] / bev_discret) + (configs.bev_width + 1) / 2)
    #print ("do negative y coordinates happen?")
    #print ((lidar_pcl_cpy[:, 1][lidar_pcl_cpy[:, 1] < 0]).any()) # False (as desired) 

    # step 4 : visualize point-cloud using the function show_pcl from a previous task
    #show_pcl(lidar_pcl_cpy) 
    #################---^----- NOTE UNCOMMENT THE ABOVE LINE BUT I CANNOT LOOK AT 100 FRAMES in o3d at the last step of this project!!! 


    # print (f"lidar_pcl_cpy.shape {lidar_pcl_cpy.shape}")

    # input ('press ENTER to continue...')
    
    #######
    ####### ID_S2_EX1 END #######     
    
    
    # Compute intensity layer of the BEV map
    ####### ID_S2_EX2 START #######     
    #######
    print("student task ID_S2_EX2")

    ## step 1 : create a numpy array filled with zeros which has the same dimensions as the BEV map
    #height_map = np.zeros((configs.bev_height + 1, configs.bev_width + 1)) # sloc from thhttps://bit.ly/2XvPsQN 
    intensity_map = np.zeros((configs.bev_height + 1, configs.bev_width + 1)) # sloc from https://bit.ly/2XvPsQN 

    # step 2 : re-arrange elements in lidar_pcl_cpy by sorting first by x, then y, then -z (use numpy.lexsort) 

    # steps not included but talked about in the project discussion in the udacity classroom 
    # verbatim project instructions from the classroom, quote [note, i didn't understand they were spreading this 
    # across rubric points but left this like this & copy / pasted it lower]:
    # "Use numpy.lexsort in step 2 to sort the point cloud. Sorting shall be performed in such a 
    # way that first, all points are sorted according to their x-coordinate in BEV space. Then, 
    # for points with the same x-coordinate, sorting shall again be performed by their y-coordinates 
    # in BEV space. In case there are points with both x and y identical, sort these by z in sensor space. 
    # Make sure to invert z as sorting is performed in ascending order and we want the top-most point for each cell." -- https://bit.ly/3klF86H
    idx_height = np.lexsort((-lidar_pcl_cpy[:, 2], lidar_pcl_cpy[:, 1], lidar_pcl_cpy[:, 0])) # s.l.o.c  from Udacity classroom: https://bit.ly/2XvPsQN 
    lidar_pcl_top  = lidar_pcl_cpy[idx_height] # s.l.o.c from Udacity classroom: https://bit.ly/2XvPsQN 

    ## step 3 : extract all points with identical x and y such that only the top-most z-coordinate is kept (use numpy.unique)
    ##          also, store the number of points per x,y-cell in a variable named "counts" for use in the next task
    #_, idx_height_unique, counts = np.unique(lidar_pcl_top[:, 0:2], axis=0, return_index=True, return_counts=True) # s.l.o.c from Udacity classroom: https://bit.ly/2XvPsQN 
    _, idx_height_unique = np.unique(lidar_pcl_top[:, 0:2], axis=0, return_index=True) 
    lidar_pcl_top = lidar_pcl_top[idx_height_unique] # s.l.o.c from Udacity classroom: https://bit.ly/2XvPsQN 
    #_, _, counts = np.unique(lidar_pcl_cpy[:, 0:2], axis=0, return_index=True, return_counts=True) 
    # ---^--- last s.l.o.c. from *this* file [commented out, below] 
    # extract all points with identical x and y such that only the top-most z-coordinate is kept (use numpy.unique)

    ## step 4 : assign the intensity value of each unique entry in lidar_top_pcl to the intensity map 
    ##          make sure that the intensity is scaled in such a way that objects of interest (e.g. vehicles) are clearly visible    
    ##          also, make sure that the influence of outliers is mitigated by normalizing intensity on the difference between the max. and min. value within the point cloud

    # Following 10 s.l.o.c. including quoted comments from the Udacity Classroom: https://bit.ly/2XvPsQN  
    # "sort points such that in case of identical BEV grid coordinates, the points in each grid cell are arranged based on their intensity"
    lidar_pcl_cpy[lidar_pcl_cpy[:,3]>1.0,3] = 1.0 #  s.l.o.c. from the Udacity Classroom: https://bit.ly/2XvPsQN  
    idx_intensity = np.lexsort((-lidar_pcl_cpy[:, 3], lidar_pcl_cpy[:, 1], lidar_pcl_cpy[:, 0]))  #  s.l.o.c. from the Udacity Classroom: https://bit.ly/2XvPsQN  
    lidar_pcl_cpy = lidar_pcl_cpy[idx_intensity]  #  s.l.o.c. from the Udacity Classroom: https://bit.ly/2XvPsQN  

    # "only keep one point per grid cell"
    _, indices = np.unique(lidar_pcl_cpy[:, 0:2], axis=0, return_index=True)  #  s.l.o.c. from the Udacity Classroom: https://bit.ly/2XvPsQN  
    lidar_pcl_int = lidar_pcl_cpy[indices]  #  s.l.o.c. from the Udacity Classroom: https://bit.ly/2XvPsQN  

    # "create the intensity map"
    #intensity_map = np.zeros((configs.bev_height + 1, configs.bev_width + 1))  #  s.l.o.c. from the Udacity Classroom: https://bit.ly/2XvPsQN  
    intensity_map[np.int_(lidar_pcl_int[:, 0]), np.int_(lidar_pcl_int[:, 1])] = lidar_pcl_int[:, 3] / (np.amax(lidar_pcl_int[:, 3])-np.amin(lidar_pcl_int[:, 3])) 
    # ---^--- s.l.o.c. from the Udacity Classroom: https://bit.ly/2XvPsQN   

    # EJS: next 3 lines **including** quoted comments on next 2 lines are directly from the 
    # classroom at: https://bit.ly/2XAo2ZL 
    # "sort points such that in case of identical BEV grid coordinates, the points in each grid cell are arranged based on their intensity"
    # lidar_pcl_cpy[lidar_pcl_cpy[:,3]>1.0,3] = 1.0 # from the classroom at https://bit.ly/2XAo2ZL
    # idx_intensity = np.lexsort((-lidar_pcl_cpy[:, 3], lidar_pcl_cpy[:, 1], lidar_pcl_cpy[:, 0])) # from the classroom at https://bit.ly/2XAo2ZL
    # lidar_pcl_cpy = lidar_pcl_cpy[idx_intensity] # from the classroom at https://bit.ly/2XAo2ZL

    ## step 5 : temporarily visualize the intensity map using OpenCV to make sure that vehicles separate well from the background

    img_intensity = intensity_map * 256 # s.l.o.c. from the Udacity Classroom: https://bit.ly/2XvPsQN   
    img_intensity = img_intensity.astype(np.uint8) # s.l.o.c. from the Udacity Classroom: https://bit.ly/2XvPsQN   
    ################ NOTE: uncomment out the next 6 lines of code BUT I cannot look at over 100 images while going through the last step of the project!!!!!
    # while (1): # s.l.o.c. from the Udacity Classroom: https://bit.ly/2XvPsQN   
    #     cv2.imshow('img_intensity', img_intensity) # s.l.o.c. from the Udacity Classroom: https://bit.ly/2XvPsQN   
    #     #if cv2.waitKey(10) & 0xFF == 27: # s.l.o.c. from the Udacity Classroom: https://bit.ly/2XvPsQN   
    #     if cv2.waitKey(10) & 0xFF == 113: # ord('q'): # https://stackoverflow.com/questions/57690899/how-cv2-waitkey1-0xff-ordq-works 
    #         break # s.l.o.c. from the Udacity Classroom: https://bit.ly/2XvPsQN   
    # cv2.destroyAllWindows() # s.l.o.c. from the Udacity Classroom: https://bit.ly/2XvPsQN    


    #######
    ####### ID_S2_EX2 END ####### 


    # Compute height layer of the BEV map
    ####### ID_S2_EX3 START #######     
    #######
    print("student task ID_S2_EX3")

    ## step 1 : create a numpy array filled with zeros which has the same dimensions as the BEV map
    #intensity_map = np.zeros((configs.bev_height + 1, configs.bev_width + 1)) # https://bit.ly/2XvPsQN # IDENTICAL to above!!! they said the same thing twice, i feel like i'm in the matrix & just another black cat a few seconds later
    height_map = np.zeros((configs.bev_height + 1, configs.bev_width + 1))

    ## step 2 : assign the height value of each unique entry in lidar_top_pcl to the height map 
    ##          make sure that each entry is normalized on the difference between the upper and lower height defined in the config file
    ##          use the lidar_pcl_top data structure from the previous task to access the pixels of the height_map



    # EJS: next 4 lines **including** quoted comments on next 2 lines are directly from the 
    # classroom at: https://bit.ly/2XAo2ZL 
    # "assign the height value of each unique entry in lidar_top_pcl to the height map and" 
    # "make sure that each entry is normalized on the difference between the upper and lower height defined in the config file"
    height_map = np.zeros((configs.bev_height + 1, configs.bev_width + 1)) # from the classroom at https://bit.ly/2XAo2ZL 
    height_map[np.int_(lidar_pcl_top[:, 0]), np.int_(lidar_pcl_top[:, 1])] = lidar_pcl_top[:, 2] / float(np.abs(configs.lim_z[1] - configs.lim_z[0]))
    # ---^--- from the classroom at: https://bit.ly/2XAo2ZL 


    ## step 3 : temporarily visualize the intensity map using OpenCV to make sure that vehicles separate well from the background
    img_height = height_map * 256 # s.l.o.c. from the Udacity Classroom: https://bit.ly/2XvPsQN   
    img_height = img_height.astype(np.uint8) # s.l.o.c. from the Udacity Classroom: https://bit.ly/2XvPsQN   
    ################ NOTE: uncomment out the next 6 lines of code BUT I cannot look at over 100 images while going through the last step of the project!!!!!
    while (1): # s.l.o.c. from the Udacity Classroom: https://bit.ly/2XvPsQN   
        cv2.imshow('img_height', img_height) # s.l.o.c. from the Udacity Classroom: https://bit.ly/2XvPsQN   
        # if cv2.waitKey(10) & 0xFF == 27: # s.l.o.c. from the Udacity Classroom: https://bit.ly/2XvPsQN   
        if cv2.waitKey(10) & 0xFF == 113: #ord('q'): # https://stackoverflow.com/questions/57690899/how-cv2-waitkey1-0xff-ordq-works 
            break # s.l.o.c. from the Udacity Classroom: https://bit.ly/2XvPsQN   
    cv2.destroyAllWindows() # s.l.o.c. from the Udacity Classroom: https://bit.ly/2XvPsQN  

    #######
    ####### ID_S2_EX3 END #######       

    # TODO remove after implementing all of the above steps
    # lidar_pcl_cpy = []
    # lidar_pcl_top = []
    # height_map = []
    # intensity_map = []

    # Compute density layer of the BEV map
    density_map = np.zeros((configs.bev_height + 1, configs.bev_width + 1))
    _, _, counts = np.unique(lidar_pcl_cpy[:, 0:2], axis=0, return_index=True, return_counts=True)
    normalizedCounts = np.minimum(1.0, np.log(counts + 1) / np.log(64)) 
    density_map[np.int_(lidar_pcl_top[:, 0]), np.int_(lidar_pcl_top[:, 1])] = normalizedCounts
        
    # assemble 3-channel bev-map from individual maps
    bev_map = np.zeros((3, configs.bev_height, configs.bev_width))
    bev_map[2, :, :] = density_map[:configs.bev_height, :configs.bev_width]  # r_map
    bev_map[1, :, :] = height_map[:configs.bev_height, :configs.bev_width]  # g_map
    bev_map[0, :, :] = intensity_map[:configs.bev_height, :configs.bev_width]  # b_map

    # expand dimension of bev_map before converting into a tensor
    s1, s2, s3 = bev_map.shape
    bev_maps = np.zeros((1, s1, s2, s3))
    bev_maps[0] = bev_map

    bev_maps = torch.from_numpy(bev_maps)  # create tensor from birds-eye view
    input_bev_maps = bev_maps.to(configs.device, non_blocking=True).float()
    return input_bev_maps


