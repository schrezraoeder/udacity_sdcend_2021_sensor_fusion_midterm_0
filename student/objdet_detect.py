# ---------------------------------------------------------------------
# Project "Track 3D-Objects Over Time"
# Copyright (C) 2020, Dr. Antje Muntzinger / Dr. Andreas Haja.
#
# Purpose of this file : Detect 3D objects in lidar point clouds using deep learning
#
# You should have received a copy of the Udacity license together with this program.
#
# https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
# ----------------------------------------------------------------------
#

# general package imports
import numpy as np
import torch
from easydict import EasyDict as edict

# EJS import: 
from argparse import ArgumentParser 

# add project directory to python path to enable relative imports
import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

# model-related
from tools.objdet_models.resnet.models import fpn_resnet
from tools.objdet_models.resnet.utils.evaluation_utils import decode, post_processing

from tools.objdet_models.resnet.utils.torch_utils import _sigmoid # i added torch utils for _sigmoid -- EJS  


from tools.objdet_models.darknet.models.darknet2pytorch import Darknet as darknet
from tools.objdet_models.darknet.utils.evaluation_utils import post_processing_v2

# this function `parse_test_configs` is almost entirely from: https://github.com/maudzung/SFA3D/blob/master/sfa/test.py 
# with just a **few** changes -- EJS 
def parse_test_configs():
    #parser = argparse.ArgumentParser(description='Testing config for the Implementation')
    parser = ArgumentParser(description='Testing config for the Implementation')
    parser.add_argument('--saved_fn', type=str, default='fpn_resnet_18', metavar='FN',
                        help='The name using for saving logs, models,...')
    parser.add_argument('-a', '--arch', type=str, default='fpn_resnet_18', metavar='ARCH',
                        help='The name of the model architecture')
    parser.add_argument('--pretrained_path', type=str,
                        default='../checkpoints/fpn_resnet_18/fpn_resnet_18_epoch_300.pth', metavar='PATH',
                        help='the path of the pretrained checkpoint')
    parser.add_argument('--K', type=int, default=50,
                        help='the number of top K')
    parser.add_argument('--no_cuda', action='store_true',
                        help='If true, cuda is not used.')
    parser.add_argument('--gpu_idx', default=0, type=int,
                        help='GPU index to use.')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='Take a subset of the dataset to run and debug')
    parser.add_argument('--num_workers', type=int, default=1,
                        help='Number of threads for loading data')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='mini-batch size (default: 4)')
    parser.add_argument('--peak_thresh', type=float, default=0.2)
    parser.add_argument('--save_test_output', action='store_true',
                        help='If true, the output image of the testing phase will be saved')
    parser.add_argument('--output_format', type=str, default='image', metavar='PATH',
                        help='the type of the test output (support image or video)')
    parser.add_argument('--output_video_fn', type=str, default='out_fpn_resnet_18', metavar='PATH',
                        help='the video filename if the output format is video')
    parser.add_argument('--output-width', type=int, default=608,
                        help='the width of showing output, the height maybe vary')

    configs = edict(vars(parser.parse_args()))
    configs.pin_memory = True
    configs.distributed = False  # For testing on 1 GPU only ## <--- even comments are from: https://github.com/maudzung/SFA3D/blob/master/sfa/test.py 

    configs.input_size = (608, 608)
    configs.hm_size = (152, 152)
    configs.down_ratio = 4
    configs.max_objects = 50

    configs.imagenet_pretrained = False
    configs.head_conv = 64
    configs.num_classes = 3
    configs.num_center_offset = 2
    configs.num_z = 1
    configs.num_dim = 3
    configs.num_direction = 2  # sin, cos

    configs.heads = {
        'hm_cen': configs.num_classes,
        'cen_offset': configs.num_center_offset,
        'direction': configs.num_direction,
        'z_coor': configs.num_z,
        'dim': configs.num_dim
    }
    configs.num_input_features = 4

    ####################################################################
    ##############Dataset, Checkpoints, and results dir configs#########
    ####################################################################
    configs.root_dir = '../'
    configs.dataset_dir = os.path.join(configs.root_dir, 'dataset', 'kitti')
    #configs.dataset_dir = 

    if configs.save_test_output:
        configs.results_dir = os.path.join(configs.root_dir, 'results', configs.saved_fn)
        make_folder(configs.results_dir)

    return configs

# load model-related parameters into an edict
def load_configs_model(model_name='darknet', configs=None):

    # init config file, if none has been passed
    if configs==None:
        configs = edict()  

    # get parent directory of this file to enable relative paths
    curr_path = os.path.dirname(os.path.realpath(__file__))
    parent_path = configs.model_path = os.path.abspath(os.path.join(curr_path, os.pardir))    
    
    # set parameters according to model type
    if model_name == "darknet":
        configs.model_path = os.path.join(parent_path, 'tools', 'objdet_models', 'darknet')
        configs.pretrained_filename = os.path.join(configs.model_path, 'pretrained', 'complex_yolov4_mse_loss.pth')
        configs.arch = 'darknet'
        configs.batch_size = 4
        configs.cfgfile = os.path.join(configs.model_path, 'config', 'complex_yolov4.cfg')
        configs.conf_thresh = 0.5
        configs.distributed = False
        configs.img_size = 608
        configs.nms_thresh = 0.4
        configs.num_samples = None
        configs.num_workers = 4
        configs.pin_memory = True
        configs.use_giou_loss = False

    elif model_name == 'fpn_resnet':
        ####### ID_S3_EX1-3 START #######     
        #######
        print("student task ID_S3_EX1-3")

        configs = parse_test_configs() 


        configs.model_path = os.path.join(parent_path, 'tools', 'objdet_models', 'resnet') 
        configs.pretrained_filename = os.path.join(configs.model_path, 'pretrained', 'fpn_resnet_18_epoch_300.pth')
        configs.arch = 'fpn_resnet' # PRESUMABLY... -- EJS [see below to wit  `elif 'fpn_resnet' in configs.arch`]  ... note c.f. line 45 here: https://github.com/maudzung/SFA3D/blob/master/sfa/test.py
        
        
        
        # Following 24 lines of code from: https://github.com/maudzung/SFA3D/blob/master/sfa/test.py -- EJS 
        # configs.pin_memory = True
        # configs.distributed = False  # For testing on 1 GPU only
        # configs.input_size = (608, 608)
        # configs.hm_size = (152, 152)
        # configs.down_ratio = 4
        # configs.max_objects = 50
        # configs.imagenet_pretrained = False
        # configs.head_conv = 64
        # configs.num_classes = 3
        # configs.num_center_offset = 2
        # configs.num_z = 1
        # configs.num_dim = 3
        # configs.num_direction = 2  # sin, cos
        # configs.heads = {
        #     'hm_cen': configs.num_classes,
        #     'cen_offset': configs.num_center_offset,
        #     'direction': configs.num_direction,
        #     'z_coor': configs.num_z,
        #     'dim': configs.num_dim
        # }
        # configs.num_input_features = 4

        #arch_parts = configs.arch.split('_') # from: https://github.com/maudzung/SFA3D/blob/5f042b9d194b63d47d740c42ad04243b02c2c26a/sfa/models/model_utils.py#L25 
        configs.num_layers = 18  # from: https://github.com/maudzung/SFA3D/blob/5f042b9d194b63d47d740c42ad04243b02c2c26a/sfa/models/model_utils.py#L25 
        configs.conf_thresh = 0.5 

        #######
        ####### ID_S3_EX1-3 END #######     

    else:
        raise ValueError("Error: Invalid model name")

    # GPU vs. CPU
    configs.no_cuda = True # if true, cuda is not used
    configs.gpu_idx = 0  # GPU index to use.
    configs.device = torch.device('cpu' if configs.no_cuda else 'cuda:{}'.format(configs.gpu_idx))

    return configs


# load all object-detection parameters into an edict
def load_configs(model_name='fpn_resnet', configs=None):

    # init config file, if none has been passed
    if configs==None:
        configs = edict()    
    
    # add model-dependent parameters
    configs = load_configs_model(model_name, configs)

    # birds-eye view (bev) parameters
    configs.lim_x = [0, 50] # detection range in m
    configs.lim_y = [-25, 25]
    configs.lim_z = [-1, 3]
    configs.lim_r = [0, 1.0] # reflected lidar intensity
    configs.bev_width = 608  # pixel resolution of bev image
    configs.bev_height = 608 

    # visualization parameters
    configs.output_width = 608 # width of result image (height may vary)
    configs.obj_colors = [[0, 255, 255], [0, 0, 255], [255, 0, 0]] # 'Pedestrian': 0, 'Car': 1, 'Cyclist': 2

    configs.min_iou = 0.5 # added by EJS for `ID_S4_EX1` 

    return configs


# create model according to selected model type
def create_model(configs):

    # check for availability of model file
    assert os.path.isfile(configs.pretrained_filename), "No file at {}".format(configs.pretrained_filename)

    # create model depending on architecture name
    if (configs.arch == 'darknet') and (configs.cfgfile is not None):
        print('using darknet')
        model = darknet(cfgfile=configs.cfgfile, use_giou_loss=configs.use_giou_loss)    
    
    elif 'fpn_resnet' in configs.arch:
        print('using ResNet architecture with feature pyramid')
        
        ####### ID_S3_EX1-4 START #######     
        #######
        print("student task ID_S3_EX1-4")
        model = fpn_resnet.get_pose_net(configs.num_layers, configs.heads, configs.head_conv, configs.imagenet_pretrained)

        #######
        ####### ID_S3_EX1-4 END #######     
    
    else:
        assert False, 'Undefined model backbone'

    # load model weights
    model.load_state_dict(torch.load(configs.pretrained_filename, map_location='cpu'))
    print('Loaded weights from {}\n'.format(configs.pretrained_filename))

    # set model to evaluation state
    configs.device = torch.device('cpu' if configs.no_cuda else 'cuda:{}'.format(configs.gpu_idx))
    model = model.to(device=configs.device)  # load model to either cpu or gpu
    model.eval()          

    return model

def detections_by_row(row_of_detections, objects, configs): 

    print ()
    print (f"row_of_detections: {row_of_detections}")
    print ()

    detections = row_of_detections 
    #detections = detections[0] 
    print (f"detections: {detections}")
    print (f"type(detections) {type(detections)}") 
    

    ## step 2 : loop over all detections
    #for each_detection in detections: 
    # ALL lines of code this for() [except last two] loop from: `project_detections_into_bev` function in https://github.com/udacity/nd013-c2-fusion-exercises/blob/main/misc/objdet_tools.py
    # for ix, a_row in enumerate(detections):
    # for a_row, _ in enumerate(detections): 
        # extract detection
        
        # print () 
        # print ('--------------------------------------')
        # print ("a_row :-)") 
        # print (a_row)
        # print (f"type(row): {type(a_row)}")  # type(row): <class 'dict'> 
        # print (f"row.keys(): {a_row.keys()}")
        # print ('--------------------------------------')
        # print () 

        # for ix, row in enumerate(a_row): 
    # for row, ix in enumerate(detections): 
    # for ix, row in detections.items(): 

        

    #     if len(row) == 0:
    #         continue 
    #     for each_row in row: # row is a numpy array, but each_row of row is a row of detections [make sense?] 
    _id, _x, _y, _z, _h, _w, _l, _yaw = detections 

    # convert from metric into pixel coordinates
    x = (_y - configs.lim_y[0]) / (configs.lim_y[1] - configs.lim_y[0]) * configs.bev_width
    y = (_x - configs.lim_x[0]) / (configs.lim_x[1] - configs.lim_x[0]) * configs.bev_height
    z = _z - configs.lim_z[0]
    w = _w / (configs.lim_y[1] - configs.lim_y[0]) * configs.bev_width
    l = _l / (configs.lim_x[1] - configs.lim_x[0]) * configs.bev_height
    yaw = -_yaw

    ## step 3 : perform the conversion using the limits for x, y and z set in the configs structure

    ## step 4 : append the current object to the 'objects' array

    #id = 1 

    object = [1, x, y, z, _h, w, l, yaw]  # <--- my line of code -- e.j.s. 
    objects.append(object) # <--- my line of code -- e.j.s. 
    return objects 


# detect trained objects in birds-eye view
def detect_objects(input_bev_maps, model, configs):

    # deactivate autograd engine during test to reduce memory usage and speed up computations
    with torch.no_grad():  

        # perform inference
        outputs = model(input_bev_maps)

        # decode model output into target object format
        if 'darknet' in configs.arch:

            # perform post-processing
            output_post = post_processing_v2(outputs, conf_thresh=configs.conf_thresh, nms_thresh=configs.nms_thresh) 
            print (f"darknet's output_post: {output_post}") 
            #input ("continue")
            detections = []
            for sample_i in range(len(output_post)):
                if output_post[sample_i] is None:
                    continue
                detection = output_post[sample_i] 
                for obj in detection:
                    x, y, w, l, im, re, _, _, _ = obj
                    yaw = np.arctan2(im, re)
                    detections.append([1, x, y, 0.0, 1.50, w, l, yaw])  
            print (f"detections: {detections}") 
            #input ('continue...')  

        elif 'fpn_resnet' in configs.arch:
            # decode output and perform post-processing
            
            ####### ID_S3_EX1-5 START #######     
            #######
            print("student task ID_S3_EX1-5")

            #outputs = model(input_bev_maps) # s.l.o.c. from line 133 here: https://github.com/maudzung/SFA3D/blob/master/sfa/test.py
            # following 7 s.l.o.c.'s from lines 134-140 here: https://github.com/maudzung/SFA3D/blob/master/sfa/test.py 
            # _sigmoid lives at: C:\Users\seneca_wolf\Desktop\sdcend_udacity_2021\nd013-c2-fusion-starter-main\tools\objdet_models\resnet\utils\torch_utils.py 
            outputs['hm_cen'] = _sigmoid(outputs['hm_cen'])
            outputs['cen_offset'] = _sigmoid(outputs['cen_offset'])
            # detections size (batch_size, K, 10)
            detections = decode(outputs['hm_cen'], outputs['cen_offset'], outputs['direction'], outputs['z_coor'],
                                outputs['dim'], K=configs.K)
            detections = detections.cpu().numpy().astype(np.float32)
            post_processed_detections = post_processing(detections, configs) #.num_classes, configs.down_ratio, configs.peak_thresh) 
            # print (f"fpn_resnet's post_processing results: {post_processed_detections}")
            # input ("continue") 
            # fpn_resnet's post_processing results: 
                # [{0: array([], shape=(0, 8), dtype=float32), 1: array([[9.7616243e-01, 3.5113776e+02, 2.1874646e+02, 1.0536969e+00,
                # 1.5929611e+00, 2.0124949e+01, 4.7386971e+01, 1.1588168e-02],
                # [6.9274604e-01, 3.1218921e+02, 3.5528729e+02, 1.1284245e+00,
                # 1.7885958e+00, 2.0844662e+01, 4.6806496e+01, 1.2181865e-02]],
            #     dtype=float32), 2: array([], shape=(0, 8), dtype=float32)}] 

            # print (f"type(post_processed_detections): {type(post_processed_detections)}") # 
            # print (f"len(post_processed_detections): {len(post_processed_detections)}") # len(post_processed_detections): 1 
            # print (f"post_processed_detections[0]: {post_processed_detections[0]}") #
            # {0: array([], shape=(0, 8), dtype=float32), 1: array([[9.7616243e-01, 3.5113776e+02, 2.1874646e+02, 1.0536969e+00,
            # 1.5929611e+00, 2.0124949e+01, 4.7386971e+01, 1.1588168e-02],
            # [6.9274604e-01, 3.1218921e+02, 3.5528729e+02, 1.1284245e+00,
            # 1.7885958e+00, 2.0844662e+01, 4.6806496e+01, 1.2181865e-02]],
            # dtype=float32), 2: array([], shape=(0, 8), dtype=float32)} 
            # print (f"type(post_processed_detections[0]): {type(post_processed_detections[0])}") # type(post_processed_detections[0]): <class 'dict'>
            
            detections = [] 
            for each_pp_detections in post_processed_detections: 
                for ix, entry in each_pp_detections.items():  # entry is going to be a numpy array which may be empty garbage or each row is the 'detections' for ID_S3_EX1 
                    if len(entry) > 0: # the 'defunct' ones have len(entry) == 0, but somehow have shape == (0, 8) 
                        for row in entry: # each row of the numpy array `entry` (shape == (2, 8) that I've seen) is a 'detection' as per the VS Code inspector bit in the classroom with the wrong directions 
                            detections.append(row) 
            detections = np.array(detections) 
            # print ()
            # print (f"detections: {detections}") 
            # detections: [array([9.7616243e-01, 3.5113776e+02, 2.1874646e+02, 1.0536969e+00,
            # 1.5929611e+00, 2.0124949e+01, 4.7386971e+01, 1.1588168e-02],
            # dtype=float32), array([6.9274604e-01, 3.1218921e+02, 3.5528729e+02, 1.1284245e+00,
            # 1.7885958e+00, 2.0844662e+01, 4.6806496e+01, 1.2181865e-02],
            # dtype=float32)]
            # print () 
            # input ('go back & work again')
            





              


            
            #######
            ####### ID_S3_EX1-5 END #######     

            

    ####### ID_S3_EX2 START #######     
    #######
    # Extract 3d bounding boxes from model response
    print("student task ID_S3_EX2")
    objects = [] 

    ## step 1 : check whether there are any detections

    # I feel that the instructions are **wrong** -- EJS. So, I'm doing it my way for now. 
    if 'darknet' in configs.arch: 
        #for row_of_detections in detections: 
        #    objects = detections_by_row(row_of_detections, objects, configs)
        objects = detections 
    else: # 'fp_resnet' and there is still work to do . . .  
        detections = list(detections) 
        if detections: 
            # print (f"detections: {detections}")
            # print (f"len(detections) {len(detections)}") 
            #if len(detections) > 1:
            for row_of_detections in detections: 
                objects = detections_by_row(row_of_detections, objects, configs) 



        # if len(detections) == 1: 
        #     detections = detections[0] 
        #     # print (f"detections: {detections}")
        #     # print (f"type(detections) {type(detections)}") 
        #     row_of_detections = detections 

        #     objects = detections_by_row(row_of_detections, objects, configs)  # this line not called on darknet during `ID_S4_EX1` 
            

            ## step 2 : loop over all detections
            #for each_detection in detections: 
            # ALL lines of code this for() [except last two] loop from: `project_detections_into_bev` function in https://github.com/udacity/nd013-c2-fusion-exercises/blob/main/misc/objdet_tools.py
            # for ix, a_row in enumerate(detections):
            # for a_row, _ in enumerate(detections): 
                # extract detection
                
                # print () 
                # print ('--------------------------------------')
                # print ("a_row :-)") 
                # print (a_row)
                # print (f"type(row): {type(a_row)}")  # type(row): <class 'dict'> 
                # print (f"row.keys(): {a_row.keys()}")
                # print ('--------------------------------------')
                # print () 

                # for ix, row in enumerate(a_row): 
            # for row, ix in enumerate(detections): 




            ########################################################## refactoring . . . 
            # for ix, row in detections.items(): 

                

            #     if len(row) == 0:
            #         continue 
            #     for each_row in row: # row is a numpy array, but each_row of row is a row of detections [make sense?] 
            #         _id, _x, _y, _z, _h, _w, _l, _yaw = each_row

            #         # convert from metric into pixel coordinates
            #         x = (_y - configs.lim_y[0]) / (configs.lim_y[1] - configs.lim_y[0]) * configs.bev_width
            #         y = (_x - configs.lim_x[0]) / (configs.lim_x[1] - configs.lim_x[0]) * configs.bev_height
            #         z = _z - configs.lim_z[0]
            #         w = _w / (configs.lim_y[1] - configs.lim_y[0]) * configs.bev_width
            #         l = _l / (configs.lim_x[1] - configs.lim_x[0]) * configs.bev_height
            #         yaw = -_yaw
                
            #         ## step 3 : perform the conversion using the limits for x, y and z set in the configs structure
                
            #         ## step 4 : append the current object to the 'objects' array

            #         object = [_id, x, y, z, _h, w, l, yaw]  # <--- my line of code -- e.j.s. 
            #         objects.append(object) # <--- my line of code -- e.j.s.  [see comment above for loop for attribution of other lines here]
        
    #######
    ####### ID_S3_EX2 START #######   
    
    return objects    

