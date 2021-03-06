import rs_snapshot
import transform
from multiprocessing import Process, Queue
import votenet_inference
from playsound import playsound


# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

""" Demo of using VoteNet 3D object detector to detect objects from a point cloud.
"""

import os
import sys
import numpy as np
import argparse
import importlib
import time

import torch
import torch.nn as nn
import torch.optim as optim

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
from pc_util import random_sampling, read_ply
from ap_helper import parse_predictions
from multiprocessing import Queue

#import torch.autograd.profiler as profiler

def preprocess_point_cloud(point_cloud):
    ''' Prepare the numpy point cloud (N,3) for forward pass '''
    point_cloud = point_cloud[:,0:3] # do not use color for now
    floor_height = np.percentile(point_cloud[:,2],0.99)
    height = point_cloud[:,2] - floor_height
    point_cloud = np.concatenate([point_cloud, np.expand_dims(height, 1)],1) # (N,4) or (N,7)
    point_cloud = random_sampling(point_cloud, 20000)
    pc = np.expand_dims(point_cloud.astype(np.float32), 0) # (1,40000,4)
    return pc

def _votenet_inference(queue):
    
    # Set file paths and dataset config
    demo_dir = os.path.join(BASE_DIR, 'demo_files') 
    
    # Use sunrgbd
    sys.path.append(os.path.join(ROOT_DIR, 'sunrgbd'))
    from sunrgbd_detection_dataset import DC # dataset config
    checkpoint_path = os.path.join(demo_dir, 'pretrained_votenet_on_sunrgbd.tar')
    
    
    eval_config_dict = {'remove_empty_box': True, 'use_3d_nms': True, 'nms_iou': 0.25,
        'use_old_type_nms': False, 'cls_nms': False, 'per_class_proposal': False,
        'conf_thresh': 0.5, 'dataset_config': DC}

    # Init the model and optimzier
    MODEL = importlib.import_module('votenet') # import network module
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = MODEL.VoteNet(num_proposal=256, input_feature_dim=1, vote_factor=1,
        #sampling='seed_fps', num_class=DC.num_class,
        sampling='vote_fps', num_class=DC.num_class,
        num_heading_bin=DC.num_heading_bin,
        num_size_cluster=DC.num_size_cluster,
        mean_size_arr=DC.mean_size_arr).to(device)
    print('Constructed model.')
    
    

    # Load checkpoint
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    checkpoint = torch.load(checkpoint_path)
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    print("Loaded checkpoint %s (epoch: %d)"%(checkpoint_path, epoch))
   
    # Load and preprocess input point cloud 
    net.eval() # set model to eval mode (for bn and dp)

    filename = queue.get()
    print(filename)
    pc_dir = os.path.join(BASE_DIR, 'point_cloud')
    pc_path = os.path.join(pc_dir, filename)

    point_cloud = read_ply(pc_path)
    pc = preprocess_point_cloud(point_cloud)
    print('Loaded point cloud data: %s'%(pc_path))
   
    # Model inference
    inputs = {'point_clouds': torch.from_numpy(pc).to(device)}
    tic = time.time()
    with torch.no_grad():
        #with profiler.profile(with_stack=True, profile_memory=True) as prof:
        end_points = net(inputs)
        toc = time.time()
        print('Inference time: %f'%(toc-tic))

    end_points['point_clouds'] = inputs['point_clouds']
    pred_map_cls = parse_predictions(end_points, eval_config_dict)
    print('Finished detection. %d object detected.'%(len(pred_map_cls[0])))

    #dump_dir = os.path.join(demo_dir, '%s_results'%('sunrgbd'))
    #if not os.path.exists(dump_dir): os.mkdir(dump_dir) 
    #MODEL.dump_results(end_points, dump_dir, DC, True)
    #print('Dumped detection results to folder %s'%(dump_dir))

    #return pred_map_cls  

    queue.put(pred_map_cls)



def play_sound():
    instructions = "I will find it for you, please wait around 20 seconds"
    filename =   'finditforyou'
    print(instructions)                                  
    playsound(filename + '.mp3') 


if __name__ == '__main__':
    
    start = time.time()
    queue = Queue()
    p0 = Process(target=votenet_inference.votenet_inference, args=(queue,))
    p0.start()
    
    p1 = Process(target=play_sound)
    p1.start()
    p1.join()

    ply_filename = rs_snapshot.take_snapshot()
    rotated_ply = transform.rotation_ply(ply_filename)
    queue.put(rotated_ply)
    #predicted_class = votenet_inference.votenet_inference(rotated_ply)

    p0.join()
    
    predicted_class = queue.get()
    end = time.time()
    print("Total runtime multi processing", end - start)
    print(predicted_class)
    
    '''No multiprocessing
    start = time.time()
    play_sound()
    ply_filename = rs_snapshot.take_snapshot()
    rotated_ply = transform.rotation_ply(ply_filename)
    predicted_class = votenet_inference(rotated_ply)
    end = time.time()
    print("Total runtime multi processing", end - start)
    print(predicted_class)
    '''
    ''' multiprocessing only for sound
    start = time.time()    
    p1 = Process(target=play_sound)
    p1.start()

    ply_filename = rs_snapshot.take_snapshot()
    rotated_ply = transform.rotation_ply(ply_filename)
    predicted_class = votenet_inference(rotated_ply)    
    #predicted_class = votenet_inference.votenet_inference(rotated_ply)

    p1.join()
    
    end = time.time()
    print("Total runtime multi processing", end - start)
    print(predicted_class)
    '''    
