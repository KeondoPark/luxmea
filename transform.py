from scipy.spatial.transform import Rotation
from numpy.linalg import norm
import math
import time
import numpy as np
import array

def rotation_ply(filename):
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
        
        return out_filename
                
if __name__ == '__main__':
    start = time.time()
    rotation_ply('snapshot_1616566593.1071548.ply')
    end = time.time()
    print("Transformation:", end - start)



'''
import numpy as np
import quaternion as quat
import math

v = [3,5,0]
axis = [0,0,1]
theta = math.pi()/2 # radian

vector = np.array([0.]+v)
rot_axis = np.array([0.]+axis)
axis_angle = (theta*0.5) * rot_axis / np.linalg.norm(rot_axis)

vec = quat.quaternion(*v)
qlog = quat.quaternion(*axis_angle)
q = np.exp(qlog)

v_prime = q * vec * np.conjugate(q)

v_prime_vec = v_prime.imag
print(v_prime_vec)
'''
