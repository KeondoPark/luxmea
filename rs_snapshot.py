#import pyrealsense2.pyrealsense2 as rs
import pyrealsense2 as rs
import numpy as np
import time

def take_snapshot():
    pipeline = rs.pipeline()    
    config = rs.config()
    config.enable_stream(stream_type=rs.stream.depth, width=480, height=270, format=rs.format.z16, framerate=15)
    pipeline.start(config)

    cnt = 0

    try:
        while True:
            frames = pipeline.wait_for_frames()
            cnt += 1
            if cnt > 5:
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
                print("Saving point cloud")
                ply.process(depth_frame)
                print("Point cloud is created as {}".format(filename))

                pipeline.stop()
                return filename
                break
    except:
        print("error occurred")
        pipeline.stop()
            
    #finally:
    #    pipeline.stop()
    #    print("Pipeline finished")


if __name__ == '__main__':
    out_file = take_snapshot()
