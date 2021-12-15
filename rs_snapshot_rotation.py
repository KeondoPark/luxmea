#import pyrealsense2.pyrealsense2 as rs
import pyrealsense2 as rs
import numpy as np
import time

from scipy.spatial.transform import Rotation
from numpy.linalg import norm
import math
import time
import numpy as np
import array
import csv
import os
import imageio
from PIL import Image

import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

save_file = False

def take_snapshot_rotation(queue):
    pipeline = rs.pipeline() 
    pc = rs.pointcloud()   
    config = rs.config()
    config.enable_stream(stream_type=rs.stream.depth, width=640, height=480, format=rs.format.z16, framerate=30)
    config.enable_stream(stream_type=rs.stream.color, width=640, height=480, format=rs.format.bgr8, framerate=30)
    pipeline.start(config)

    cnt = 0

    #try:
    while True:
        frames = pipeline.wait_for_frames()
        cnt += 1
        if cnt > 5:
            depth_frame = frames.get_depth_frame()   
            color_frame = frames.get_color_frame()      
        
            if not depth_frame or not color_frame:
                continue
            print("Get depth frame")

            color_image = np.asanyarray(color_frame.get_data())
            color_image = color_image[..., ::-1]

            #Get intrinsic
            depth_intrinsics = rs.video_stream_profile(
                depth_frame.profile).get_intrinsics()
            w, h = depth_intrinsics.width, depth_intrinsics.height
            Rtilt = np.reshape(np.eye(3), -1)
            K = np.array([depth_intrinsics.fx, 0, 0, 0, depth_intrinsics.fy, 0, depth_intrinsics.ppx, depth_intrinsics.ppy, 1])
            print(Rtilt, K)
            

            if save_file:            
                filename = 'snapshot_' + str(currentTime) + '.ply'
                if not os.path.exists('point_cloud'): os.mkdir('point_cloud')
                ply = rs.save_to_ply('point_cloud/' + filename)
                ply.set_option(rs.save_to_ply.option_ply_binary, False)
                ply.set_option(rs.save_to_ply.option_ignore_color, True)
                ply.set_option(rs.save_to_ply.option_ply_normals, False)
                ply.set_option(rs.save_to_ply.option_ply_mesh, False)
                print("Saving point cloud...")
                ply.process(depth_frame)
                print("Point cloud is created as {}".format(filename))

                rgb_filename = filename[:-3] + 'jpg'
                if not os.path.exists('rgb_image'): os.mkdir('rgb_image')
                imageio.imwrite('rgb_image/' + rgb_filename, color_image)
                print("RGB Image is created as {}".format(rgb_filename))

            else:
                
                decimate = rs.decimation_filter()
                
                decimate.set_option(rs.option.filter_magnitude, 2 ** 1)
                depth_frame = decimate.process(depth_frame)

                points = pc.calculate(depth_frame)
                pc.map_to(depth_frame)

            pipeline.stop()                
            break

    if save_file:
        with open('point_cloud/' + filename,'r') as input_file:            
            header_cnt = 8
            cnt = 0
            all_lines = input_file.readlines()        
            #Get header and data
            header = all_lines[:header_cnt]
            split_line = header[3].strip().split(' ')
            vertex_cnt = int(split_line[2])
            data = all_lines[header_cnt:(vertex_cnt + header_cnt)]
                    
            point_cloud = []
            
            # Random sampling
            replace = True if vertex_cnt < 20000 else False
            sampled_int = np.random.choice(vertex_cnt, 20000, replace=replace)

            
            if vertex_cnt < 20000:
                vertex_cnt = vertex_cnt
            else:
                vertex_cnt = 20000
            header[3] = ' '.join(split_line[:2] + [str(vertex_cnt) + '\n']) #We will randomly choose 20000 points, so change the point cloud info


            #Get point cloud
            if vertex_cnt >= 20000:
                idx = -1
                for line in data:
                    idx += 1
                    if idx not in sampled_int:
                        continue

                    line_split = line.strip().split(' ')
                    new_line = []
                    
                    for item in line_split:
                        new_line.append(float(item))
                    point_cloud.append(new_line)   
            else:         
                for line in data:                
                    line_split = line.strip().split(' ')
                    new_line = []
                    
                    for item in line_split:
                        new_line.append(float(item))
                    point_cloud.append(new_line)   
                   

            #Apply rotation by 90 degree along x-axis
            axis = [1,0,0]
            axis = axis / norm(axis)
            theta = math.pi/2
            rot = Rotation.from_rotvec(theta * axis)

            new_point_cloud = rot.apply(point_cloud)

            # Saving point cloud
            out_filename = filename[:-4] + '_r.ply'
            with open('point_cloud/' + out_filename,'w') as output_file:
                output_file.writelines(header)
                for line in new_point_cloud.tolist():
                    print_line = ''
                    for item in line:
                        print_line += "{:.5f}".format(item) + ' '
                    print_line += '\n'
                    output_file.write(print_line)
            print('Rotation completed')

            #queue.put(out_filename)
    else:
        v = points.get_vertices()
        verts = np.asanyarray(v).view(np.float32).reshape(-1, 3)  # xyz        
        verts = verts[verts[:,2] < 5] # clip where z is farther than 10
        verts = verts[verts[:,2] > 0.1] # clip where z is closer than 0.1        
        print(verts)
        
        data = verts
        vertex_cnt = data.shape[0]        
        
        # Random sampling
        replace = True if vertex_cnt < 20000 else False
        sampled_int = np.random.choice(vertex_cnt, 20000, replace=replace)
        point_cloud = verts[sampled_int]
        
        if vertex_cnt < 20000:
            vertex_cnt = vertex_cnt
        else:
            vertex_cnt = 20000

        #Apply rotation by 90 degree along x-axis
        axis = [1,0,0]
        axis = axis / norm(axis)
        theta = - math.pi/2
        rot = Rotation.from_rotvec(theta * axis)

        new_point_cloud = rot.apply(point_cloud)  
        pc_path = os.path.join(BASE_DIR, 'point_cloud', 'pc.npy')
        np.save(pc_path, new_point_cloud)
        print('Rotation completed')   

        img_path = os.path.join(BASE_DIR, 'rgb_image', 'rgb.npy')
        np.save(img_path, color_image)

        queue.put(pc_path)
        queue.put(img_path)
        queue.put((Rtilt, K))        
        print("Sending queue finished")

    
    

    
            
    #finally:
    #    pipeline.stop()
    #    print("Pipeline finished")

def rotation_only(queue):
    filename = queue.get()
    start = time.time()

    header_cnt = 8
    data = np.genfromtxt('point_cloud/' + filename, skip_header=header_cnt, delimiter=' ', usecols=[0,1,2])
    with open('point_cloud/' + filename,'r') as input_file:
        header_cnt = 8
        cnt = 0
        all_lines = input_file.readlines()        
        #Get header and data
        header = all_lines[:header_cnt]
        split_line = header[3].strip().split(' ')
        vertex_cnt = int(split_line[2])
        #data = np.array([list(map(float,line.strip().split(' '))) for line in all_lines[header_cnt:(vertex_cnt + header_cnt)]])
                
        point_cloud = []
        
        # Random sampling
        replace = True if vertex_cnt < 20000 else False
        sampled_int = np.random.choice(vertex_cnt, 20000, replace=replace)

        
        if vertex_cnt < 20000:
            vertex_cnt = vertex_cnt
        else:
            vertex_cnt = 20000
        header[3] = ' '.join(split_line[:2] + [str(vertex_cnt) + '\n']) #We will randomly choose 20000 points, so change the point cloud info

        point_cloud = data[sampled_int]
        
        #Apply rotation by 90 degree along x-axis
        axis = [1,0,0]
        axis = axis / norm(axis)
        theta = math.pi/2
        rot = Rotation.from_rotvec(theta * axis)

        new_point_cloud = rot.apply(point_cloud[:,:3])

        # Saving point cloud
        out_filename = 'rotated_pc.ply'
        with open('point_cloud/' + out_filename,'w') as output_file:
            output_file.writelines(header)
            for line in new_point_cloud.tolist():
                print_line = ''
                for item in line:
                    print_line += "{:.5f}".format(item) + ' '
                print_line += '\n'
                output_file.write(print_line)
        print('Rotation completed')
        
        #return out_filename
    
    print("Rotation time:", time.time() - start)
    queue.put(out_filename)

    
            
    #finally:
    #    pipeline.stop()
    #    print("Pipeline finished")


if __name__ == '__main__':
    out_file = take_snapshot_rotation()
