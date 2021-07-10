from flask import Flask, render_template, Response, jsonify
import time
import cv2
import math
import time
import numpy as np
import pyrealsense2 as rs


from scipy.spatial.transform import Rotation
from numpy.linalg import norm
import array

app = Flask(__name__)

class AppState:

    def __init__(self, *args, **kwargs):
        self.WIN_NAME = 'RealSense'
        self.pitch, self.yaw = math.radians(0), math.radians(0)
        self.translation = np.array([0, 0, -1], dtype=np.float32)
        self.distance = 2
        self.prev_mouse = 0, 0
        self.mouse_btns = [False, False, False]
        self.paused = False
        self.decimate = 1
        self.scale = True
        self.color = True

    def reset(self):
        self.pitch, self.yaw, self.distance = 0, 0, 2
        self.translation[:] = 0, 0, -1

    @property
    def rotation(self):
        Rx, _ = cv2.Rodrigues((self.pitch, 0, 0))
        Ry, _ = cv2.Rodrigues((0, self.yaw, 0))
        return np.dot(Ry, Rx).astype(np.float32)

    @property
    def pivot(self):
        return self.translation + np.array((0, 0, self.distance), dtype=np.float32)


state = AppState()
# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.depth, width=480, height=270, format=rs.format.z16, framerate=15)
config.enable_stream(rs.stream.color, rs.format.bgr8, framerate=15)

# Start streaming
pipeline.start(config)

# Get stream profile and camera intrinsics
profile = pipeline.get_active_profile()
depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
depth_intrinsics = depth_profile.get_intrinsics()
w, h = depth_intrinsics.width, depth_intrinsics.height

# Processing blocks
pc = rs.pointcloud()
decimate = rs.decimation_filter()
decimate.set_option(rs.option.filter_magnitude, 2 ** state.decimate)
colorizer = rs.colorizer()

out = np.empty((h, w, 3), dtype=np.uint8)


def project(v):
    """project 3d vector array to 2d"""
    h, w = out.shape[:2]
    view_aspect = float(h)/w

    # ignore divide by zero for invalid depth
    with np.errstate(divide='ignore', invalid='ignore'):
        proj = v[:, :-1] / v[:, -1, np.newaxis] * \
            (w*view_aspect, h) + (w/2.0, h/2.0)

    # near clipping
    znear = 0.03
    proj[v[:, 2] < znear] = np.nan
    return proj


def view(v):
    """apply view transformation on vector array"""
    return np.dot(v - state.pivot, state.rotation) + state.pivot - state.translation


def line3d(out, pt1, pt2, color=(0x80, 0x80, 0x80), thickness=1):
    """draw a 3d line from pt1 to pt2"""
    p0 = project(pt1.reshape(-1, 3))[0]
    p1 = project(pt2.reshape(-1, 3))[0]
    if np.isnan(p0).any() or np.isnan(p1).any():
        return
    p0 = tuple(p0.astype(int))
    p1 = tuple(p1.astype(int))
    rect = (0, 0, out.shape[1], out.shape[0])
    inside, p0, p1 = cv2.clipLine(rect, p0, p1)
    if inside:
        cv2.line(out, p0, p1, color, thickness, cv2.LINE_AA)


def grid(out, pos, rotation=np.eye(3), size=1, n=10, color=(0x80, 0x80, 0x80)):
    """draw a grid on xz plane"""
    pos = np.array(pos)
    s = size / float(n)
    s2 = 0.5 * size
    for i in range(0, n+1):
        x = -s2 + i*s
        line3d(out, view(pos + np.dot((x, 0, -s2), rotation)),
               view(pos + np.dot((x, 0, s2), rotation)), color)
    for i in range(0, n+1):
        z = -s2 + i*s
        line3d(out, view(pos + np.dot((-s2, 0, z), rotation)),
               view(pos + np.dot((s2, 0, z), rotation)), color)


def axes(out, pos, rotation=np.eye(3), size=0.075, thickness=2):
    """draw 3d axes"""
    line3d(out, pos, pos +
           np.dot((0, 0, size), rotation), (0xff, 0, 0), thickness)
    line3d(out, pos, pos +
           np.dot((0, size, 0), rotation), (0, 0xff, 0), thickness)
    line3d(out, pos, pos +
           np.dot((size, 0, 0), rotation), (0, 0, 0xff), thickness)


def frustum(out, intrinsics, color=(0x40, 0x40, 0x40)):
    """draw camera's frustum"""
    orig = view([0, 0, 0])
    w, h = intrinsics.width, intrinsics.height

    for d in range(1, 6, 2):
        def get_point(x, y):
            p = rs.rs2_deproject_pixel_to_point(intrinsics, [x, y], d)
            line3d(out, orig, view(p), color)
            return p

        top_left = get_point(0, 0)
        top_right = get_point(w, 0)
        bottom_right = get_point(w, h)
        bottom_left = get_point(0, h)

        line3d(out, view(top_left), view(top_right), color)
        line3d(out, view(top_right), view(bottom_right), color)
        line3d(out, view(bottom_right), view(bottom_left), color)
        line3d(out, view(bottom_left), view(top_left), color)


def pointcloud(out, verts, texcoords, color, painter=True):
    """draw point cloud with optional painter's algorithm"""
    if painter:
        # Painter's algo, sort points from back to front

        # get reverse sorted indices by z (in view-space)
        # https://gist.github.com/stevenvo/e3dad127598842459b68
        v = view(verts)
        s = v[:, 2].argsort()[::-1]
        proj = project(v[s])
    else:
        proj = project(view(verts))

    if state.scale:
        proj *= 0.5**state.decimate

    h, w = out.shape[:2]

    # proj now contains 2d image coordinates
    j, i = proj.astype(np.uint32).T

    # create a mask to ignore out-of-bound indices
    im = (i >= 0) & (i < h)
    jm = (j >= 0) & (j < w)
    m = im & jm

    cw, ch = color.shape[:2][::-1]
    if painter:
        # sort texcoord with same indices as above
        # texcoords are [0..1] and relative to top-left pixel corner,
        # multiply by size and add 0.5 to center
        v, u = (texcoords[s] * (cw, ch) + 0.5).astype(np.uint32).T
    else:
        v, u = (texcoords * (cw, ch) + 0.5).astype(np.uint32).T
    # clip texcoords to image
    np.clip(u, 0, ch-1, out=u)
    np.clip(v, 0, cw-1, out=v)

    # perform uv-mapping
    out[i[m], j[m]] = color[u[m], v[m]]

def gen_frames():  # generate frame by frame from camera    
    while True:        
        # Grab camera data
        if not state.paused:
            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()

            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()

            depth_frame = decimate.process(depth_frame)

            # Grab new intrinsics (may be changed by decimation)
            depth_intrinsics = rs.video_stream_profile(
                depth_frame.profile).get_intrinsics()
            w, h = depth_intrinsics.width, depth_intrinsics.height

            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            depth_colormap = np.asanyarray(
                colorizer.colorize(depth_frame).get_data())

            if state.color:
                mapped_frame, color_source = color_frame, color_image
            else:
                mapped_frame, color_source = depth_frame, depth_colormap

            points = pc.calculate(depth_frame)
            pc.map_to(mapped_frame)

            # Pointcloud data to arrays
            v, t = points.get_vertices(), points.get_texture_coordinates()
            verts = np.asanyarray(v).view(np.float32).reshape(-1, 3)  # xyz
            texcoords = np.asanyarray(t).view(np.float32).reshape(-1, 2)  # uv

        # Render
        now = time.time()

        out.fill(0)

        grid(out, (0, 0.5, 1), size=1, n=10)
        frustum(out, depth_intrinsics)
        axes(out, view([0, 0, 0]), state.rotation, size=0.1, thickness=1)

        if not state.scale or out.shape[:2] == (h, w):
            pointcloud(out, verts, texcoords, color_source)
        else:
            tmp = np.zeros((h, w, 3), dtype=np.uint8)
            pointcloud(tmp, verts, texcoords, color_source)
            tmp = cv2.resize(
                tmp, out.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
            np.putmask(out, tmp > 0, tmp)
        
        dt = time.time() - now
        
        

        ret, buffer = cv2.imencode('.jpg', out)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result
    
    # Stop streaming
    pipeline.stop()


@app.route('/video_feed')
def video_feed():
    #Video streaming route. Put this in the src attribute of an img tag
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/depth_snapshot', methods=['GET'])
def send_depth_snapshot():
    # Send back the file name of created depth image
    print("Entered depth snapshot function")
    state.paused = True        
    while True:
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()         
    
        if not depth_frame:
            continue

        print("Get depth frame")

        currentTime = time.time()
        filename = 'snapshot_' + str(currentTime) + '.ply'
        ply = rs.save_to_ply('point_cloud/' + filename)
        ply.set_option(rs.save_to_ply.option_ply_binary, False)
        ply.set_option(rs.save_to_ply.option_ignore_color, True)
        ply.set_option(rs.save_to_ply.option_ply_normals, False)
        ply.set_option(rs.save_to_ply.option_ply_mesh, False)
        print("Saving point cloud...")
        ply.process(depth_frame)
        print("Point cloud is created as {}".format(filename))
        
        break

    print(filename)
    return jsonify({'filename':filename})

    #TODO: Modifry the code to use rotation

    '''
    with open('point_cloud/' + filename,'r') as input_file:
        print("Opened point cloud")
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
    
    #state.paused=False

    return jsonify({'filename':filename})
    '''
@app.route('/puase_onoff')
def puase_onoff():
    """Pause on off"""
    if state.paused:
        state.paused = False
    else:
        state.paused = True
    return jsonify({'success':True})


@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')



if __name__ == '__main__':
    app.run(host="0.0.0.0",debug=False)



