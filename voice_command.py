import random
import time

import speech_recognition as sr
import os
import sys


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

import math
import numpy as np
from gtts import gTTS
from playsound import playsound
import multiprocessing as mp
from multiprocessing import Process, Queue, Pipe, Array
import rs_snapshot_rotation
import requests
import argparse
from PIL import Image
import json

sys.path.append(os.path.join(ROOT_DIR, 'votenet_tf'))
import votenet_inference

SERVER_ADDRESS = "127.0.0.1"
SERVER_PORT = 5000

SERVER_ADDRESS_PORT = SERVER_ADDRESS + ":" + str(SERVER_PORT)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AUDIO_DIR = os.path.join(BASE_DIR, 'audio_files')
PROMPT_LIMIT = 5
WAKE_UP_WORD = ['lux mia', 'lux mea', 'knox mia', 'luxmia', 'luxmea', 'knoxmia', 'Lochmere', 'la mia', 'Locs mia']
CLASS2TYPE_DICT = {'bed':0, 'table':1, 'sofa':2, 'chair':3, 'toilet':4, 'desk':5, 'dresser':6, 'night_stand':7, 'bookshelf':8, 'bathtub':9,'person':10}
TYPE2CLASS_DICT = {0:'bed', 1:'table', 2:'sofa', 3:'chair', 4:'toilet', 5:'desk', 6:'dresser', 7:'night_stand', 8:'bookshelf', 9:'bathtub',10:'person'}
SUPPORTED_CLASS = ['bed','table','chair','toilet','desk','dresser','night_stand','bookshelf','bathtub','person']


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ignore-wakeup', action='store_true', help='Jump to find target without wake up word')
    parser.add_argument('--target-obj', type=str, default=None, help='Input target object if you do not want to trigger by voice')
    FLAGS = parser.parse_args()
    return FLAGS

def send_request_wrapper(queue):
    res = requests.get("http://" + SERVER_ADDRESS_PORT + "/depth_snapshot")
    print(res)
    pc_path = res.json()['pc_path']
    img_path = res.json()['img_path']
    calibs = res.json()['calibs']

    queue.put(pc_path)
    queue.put(img_path)
    queue.put(calibs)

def recognize_speech_from_mic(recognizer, microphone):
    """Transcribe speech from recorded from `microphone`.

    Returns a dictionary with three keys:
    "success": a boolean indicating whether or not the API request was
               successful
    "error":   `None` if no error occured, otherwise a string containing
               an error message if the API could not be reached or
               speech was unrecognizable
    "transcription": `None` if speech could not be transcribed,
               otherwise a string containing the transcribed text
    """
    
    # check that recognizer and microphone arguments are appropriate type
    if not isinstance(recognizer, sr.Recognizer):
        raise TypeError("`recognizer` must be `Recognizer` instance")

    if not isinstance(microphone, sr.Microphone):
        raise TypeError("`microphone` must be `Microphone` instance")

    #print("Running voice recognition module...")
    # adjust the recognizer sensitivity to ambient noise and record audio
    # from the microphone

    with microphone as source:
        recognizer.adjust_for_ambient_noise(source, duration=0.5)
        audio = recognizer.listen(source)

    # set up the response object
    response = {
        "success": True,
        "error": None,
        "transcription": None
    }

    # try recognizing the speech in the recording
    # if a RequestError or UnknownValueError exception is caught,
    #     update the response object accordingly
    try:
        response["transcription"] = recognizer.recognize_google(audio)
    except sr.RequestError:
        # API was unreachable or unresponsive
        response["success"] = False
        response["error"] = "API unavailable"
    except sr.UnknownValueError:
        # speech was unintelligible
        response["error"] = "Unable to recognize speech"

    return response

def playsound_intruction(instruction_id, make_file=False, **kwargs):
    instruction_dict = {
        'wakeup': "Please wake me up by saying Lux Mea",
        'noobject': "Sorry no object was found",        
        'look_for': "What are you looking for?",
        'speaknow': 'Speak now!',
        'bye': "Exiting program. Bye!",
        'finditforyou': "I will find it for you, please wait for a moment",
        'findaroundyou': "Hi, I am Lux Mea. I will find objects around you.",
        'saidobject': lambda target_obj: "You said: {}".format(target_obj),
        'guidance': lambda target_obj, distance, direction: \
                   'There is a ' + target_obj + " {:.2f}".format(distance) + 'meter away at ' + str(direction) + "o'clock direction"
    }

    if instruction_id == 'guidance':
        instructions = instruction_dict[instruction_id](kwargs['target_obj'], kwargs['distance'], kwargs['direction'])
    elif instruction_id == 'saidobject':
        instructions = instruction_dict[instruction_id](kwargs['target_obj'])
    else:
        instructions = instruction_dict[instruction_id]

    audio_file_path = os.path.join(AUDIO_DIR, instruction_id + '.mp3')
    if make_file:
        myOutput=gTTS(text=instructions, lang='en', slow=False)
        myOutput.save(audio_file_path)
    print(instructions)
    playsound(audio_file_path)             

def decode_results_with_target(predicted_class, target_obj, server_on=False):
    targets = []
    person_found = []
    boxes = []

    if server_on:
        #Turn on streaming again                    
        pause_res = requests.get("http://" + SERVER_ADDRESS_PORT + "/pause_onoff")

    for item in predicted_class[0]:
        if target_obj == 'desk' and item[0] in [CLASS2TYPE_DICT['desk'], CLASS2TYPE_DICT['table']] \
        or item[0] == CLASS2TYPE_DICT[target_obj]:        
            found = {}
            found['center'] = np.mean(item[1], axis=0)
            found['dist'] = np.linalg.norm(found['center'])
            found['score'] = item[2]
            found['heading_angle'] = item[3]
            targets.append(found)
            boxes.append(item[1].tolist())
    
    #Find a person
    for item in predicted_class[0]:
        if item[0] == CLASS2TYPE_DICT['person']:
            found = {}
            found['center'] = np.mean(item[1], axis=0)
            found['dist'] = np.linalg.norm(found['center'])
            found['score'] = item[2]
            #found['heading_angle'] = item[3]                                
            found['heading_angle'] = - math.pi / 2 # Person is assumed to head toward camera
            if len(person_found) > 0 and person_found[0]['score'] < found['score'] :
                person_found[0] = found
            elif len(person_found) == 0:
                person_found.append(found)

    targets.sort(key=lambda x: x['score'])

    if len(targets) == 0:
        playsound_intruction('noobject', make_file=False)                          

    for target, box in zip(targets, boxes):
        if len(person_found) == 0:
            #WHen there is no person found, Guide is based on camera location
            found = {}
            found['center'] = np.array([0.0, 0.0, 0.0])
            found['heading_angle'] = math.pi / 2 
            person_found.append(found)
            
        # x, y difference from target to person
        rel_x = target['center'][0] - person_found[0]['center'][0] 
        rel_y = target['center'][2] - person_found[0]['center'][2]

        distance = (rel_x **2  + rel_y **2) ** 0.5
        
        # Calculate direction from person to target
        tan = rel_y / rel_x
        deg = math.atan(tan)
        if rel_x < 0: deg += math.pi
        #tan = target['center'][2] / target['center'][0]
        rel_deg = deg - person_found[0]['heading_angle']
        
        if rel_deg < 0 : rel_deg + 2 * math.pi

        rel_deg += math.pi / 12  # 15 degree / half of each hour
        if rel_deg >= 2 * math.pi * 2: rel_deg -= 2 * math.pi                                
        direction = 12 - int(rel_deg / (math.pi / 6))

        if server_on:
            requests.post("http://" + SERVER_ADDRESS_PORT + "/get_boxes", json={'boxes':[box]})

        playsound_intruction('guidance', make_file=True, target_obj=target_obj, distance=distance, direction=direction)

def decode_results_around(predicted_class, server_on=False):
    objects = []
    person_found = []
    boxes = []

    if server_on:
        #Turn on streaming again                    
        pause_res = requests.get("http://" + SERVER_ADDRESS_PORT + "/pause_onoff")

    for item in predicted_class[0]:        
        found = {}
        found['center'] = np.mean(item[1], axis=0)
        found['dist'] = np.linalg.norm(found['center'])
        found['score'] = item[2]            
        if item[0] == CLASS2TYPE_DICT['person']:            
            found['heading_angle'] = - math.pi / 2 # Person is assumed to head toward camera
            if len(person_found) > 0 and person_found[0]['score'] < found['score'] :
                person_found[0] = found
            elif len(person_found) == 0:
                person_found.append(found)
        else:
            found['class'] = TYPE2CLASS_DICT[item[0]]
            found['heading_angle'] = item[3]
            objects.append(found)
            boxes.append(item[1].tolist())

    objects.sort(key=lambda x: x['score'])

    if len(objects) == 0:
        playsound_intruction('noobject', make_file=False)                          

    for obj, box in zip(objects, boxes):
        if len(person_found) == 0:
            #WHen there is no person found, Guide is based on camera location
            found = {}
            found['center'] = np.array([0.0, 0.0, 0.0])
            found['heading_angle'] = math.pi / 2 
            person_found.append(found)
            
        # x, y difference from obj to person
        rel_x = obj['center'][0] - person_found[0]['center'][0] 
        rel_y = obj['center'][2] - person_found[0]['center'][2]

        distance = (rel_x **2  + rel_y **2) ** 0.5
        
        # Calculate direction from person to obj
        tan = rel_y / rel_x
        deg = math.atan(tan)
        if rel_x < 0: deg += math.pi        
        rel_deg = deg - person_found[0]['heading_angle']
        
        if rel_deg < 0 : rel_deg + 2 * math.pi

        rel_deg += math.pi / 12  # 15 degree / half of each hour
        if rel_deg >= 2 * math.pi * 2: rel_deg -= 2 * math.pi                                
        direction = 12 - int(rel_deg / (math.pi / 6))

        if server_on:
            requests.post("http://" + SERVER_ADDRESS_PORT + "/get_boxes", json={'boxes':[box]})

        playsound_intruction('guidance', make_file=True, target_obj=obj['class'], distance=distance, direction=direction)

def adjust_similar_words(captured):
    target_obj = captured
    if captured in 'chair tear care cher':
        target_obj = 'chair'  
    if captured in 'desk task deck that':
        target_obj = 'desk'
    if captured in 'table fable':
        target_obj = 'table'   
    return target_obj  

def proceedToFindTarget(recognizer, microphone, ignore_wakeup=False, cnt=0):
    cnt += 1
    if ignore_wakeup:
        return True    
    instructions = "Please wake me up by saying Lux Mea"                                    
    playsound('audio_files/wakeup.mp3')                
    listening = recognize_speech_from_mic(recognizer, microphone)
    result = listening['transcription'] and listening['transcription'].lower() in WAKE_UP_WORD or cnt >= 1
    return result, cnt

def check_server_on():
    # if streaming server is on, send request and get depth image
    # Otherwise, control the camera directly
    import socket
    a_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)                    
    location = (SERVER_ADDRESS, SERVER_PORT)
    res = a_socket.connect_ex(location)
    if res == 0:
        return True
    else:
        return False

def find_mic():
    m, idx = -1, 0
    for device_name in sr.Microphone().list_microphone_names():
        if device_name.startswith("ReSpeaker"):
            m = idx
            print("Respeaker found, device_index=", m)
            break
        idx += 1

    if m < 0:
        print("Respeaker not found")        
    else:
        return sr.Microphone(device_index=m)  

def find_target(ignore_wakeup=False, target_obj=None):
    # create recognizer and mic instances
    recognizer = sr.Recognizer()            
    microphone = find_mic() 
    cnt = 0    

    # Start votenet sub process
    p0_parent_conn, p0_child_conn = Pipe()
    p0 = Process(target=votenet_inference.run_inference, \
                    args=(p0_child_conn,))
    p0.start()
    
    while True:            
        proceed, cnt = proceedToFindTarget(recognizer, microphone, ignore_wakeup, cnt)
        if proceed:                 
            while True:       
                if not target_obj in SUPPORTED_CLASS:     
                    playsound_intruction('look_for', make_file=False)                    
                    
                    for j in range(PROMPT_LIMIT): 
                        playsound_intruction('speaknow', make_file=False)                        
                        
                        response = recognize_speech_from_mic(recognizer, microphone)
                        if response['transcription']:
                            break
                        if not response['success']:
                            break
                            
                    if response["error"]:
                        print("Error:{}".format(guess["error"]))
                    elif not response["success"]:
                        print("I didn't catch that. Exiting program..\n")
                    else:                               
                        captured = response["transcription"].lower()
                        target_obj = adjust_similar_words(captured)          
                        playsound_intruction('saidobject', make_file=True, target_obj=target_obj)  
                        
                        if target_obj == 'exit':
                            playsound_intruction('bye', make_file=False)  
                            break
                else:
                    target_obj = FLAGS.target_obj
                
                
                if target_obj in SUPPORTED_CLASS:                            
                    start = time.time()  
                    """
                    Using shared array for multiprocessing... Replaced
                    point_cloud_shared = np.frombuffer(Array('d', 20000 * 3).get_obj(), dtype='d').reshape((20000,3))
                    img_shared = np.frombuffer(Array('i', 480 * 640 * 3).get_obj(), dtype='i').reshape((480,640,3))
                    Rtilt_shared = np.frombuffer(Array('d', 9).get_obj(), dtype='d')
                    K_shared = np.frombuffer(Array('d', 9).get_obj(), dtype='d')
                    """
                    
                    #q0, p0: Queue/Process for votenet
                    #q1, p1: Queue/Process for depth image(camera)
                    print("Starting to take depth snapshot...")
                    
                    #Take depth image from Realsense camera
                    q1 = Queue()                   

                    server_on = check_server_on()

                    # and then check the response...
                    if server_on:                        
                        print("Server Active")                    
                        p1 = Process(target=send_request_wrapper, args=(q1,))
                    else:
                        print("Server Inactive")                    
                        p1 = Process(target=rs_snapshot_rotation.take_snapshot_rotation, args=(q1,))  
                    
                    p1.start()                    
                    p0_parent_conn.send(q1.get())
                    p0_parent_conn.send(q1.get())
                    p0_parent_conn.send(q1.get()) 

                    playsound_intruction('finditforyou', make_file=False)                                       
                    
                    predicted_class = p0_parent_conn.recv()                    
                    end = time.time()
                    print("Total runtime multi processing", end - start)                      
                    #p0.terminate()     

                    decode_results_with_target(predicted_class, target_obj, server_on=True)

                    if p1.is_alive():
                        print("P1 still alive, kill")
                        p1.terminate()
                        p1.join()                    
                    
                    time.sleep(3)
        break      

def detect_person(image_input):

    from pycoral.adapters import common
    from pycoral.adapters import detect
    from pycoral.utils.dataset import read_label_file
    from pycoral.utils.edgetpu import make_interpreter
    label_path = os.path.join(BASE_DIR, 'coral_files', 'coco_labels.txt')
    model_path = os.path.join(BASE_DIR, 'coral_files', 'ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite')
    print(model_path)
    image = Image.fromarray(image_input)
    print(image)

    labels = read_label_file(label_path) 
    print("labels", labels)
    interpreter = make_interpreter(model_path)
    print("INterpreter made")
    interpreter.allocate_tensors()    
    print("Tensor allocated")
    _, scale = common.set_resized_input(
        interpreter, image.size, lambda size: image.resize(size, Image.ANTIALIAS))
    print("Before invoke")
    interpreter.invoke()
    objs = detect.get_objects(interpreter, 0.4, scale)
    print(objs)
    for obj in objs:
        print(labels.get(obj.id, obj.id))
        print('  id:    ', obj.id)
        print('  score: ', obj.score)
        print('  bbox:  ', obj.bbox)

    return False


def auto_detect(queue):
    # create recognizer and mic instances
    recognizer = sr.Recognizer()            
    microphone = find_mic()     

    # Start votenet sub process
    p0_parent_conn, p0_child_conn = Pipe()
    p0 = Process(target=votenet_inference.run_inference, \
                    args=(p0_child_conn,))
    p0.start()


    while True:
        if queue.get() == 'FIND_PERSON':
            #person_found = detect_person(queue.get())
            person_found = True
            if not person_found:
                #Turn on streaming again                    
                res = requests.get("http://" + SERVER_ADDRESS_PORT + "/end_detecting")
            
            else:
                start = time.time()   
                
                #q0, p0: Queue/Process for votenet
                #q1, p1: Queue/Process for depth image(camera)
                print("Starting to take depth snapshot...")
                
                #Take depth image from Realsense camera
                q1 = Queue()                   

                server_on = check_server_on()

                # and then check the response...
                if server_on:                        
                    print("Server Active")                    
                    p1 = Process(target=send_request_wrapper, args=(q1,))
                else:
                    print("Server Inactive")                    
                    p1 = Process(target=rs_snapshot_rotation.take_snapshot_rotation, args=(q1,))  
                
                p1.start()                    
                p0_parent_conn.send(q1.get())
                p0_parent_conn.send(q1.get())
                p0_parent_conn.send(q1.get())                                                    
                
                playsound_intruction('findaroundyou', make_file=True)
                
                predicted_class = p0_parent_conn.recv()                    
                end = time.time()
                print("Total runtime multi processing", end - start)                      
                #p0.terminate()     

                decode_results_around(predicted_class, server_on=True)

                if p1.is_alive():
                    print("P1 still alive, kill")
                    p1.terminate()
                    p1.join()   

                if server_on:
                    #Turn on streaming again                    
                    res = requests.get("http://" + SERVER_ADDRESS_PORT + "/end_detecting")

                """
                if p0.is_alive():
                    print("P0 still alive, kill")
                    p0.terminate()
                    p0.join()                     
                """

if __name__ == "__main__":
    
    FLAGS = parse_opt()
    #find_target(ignore_wakeup=FLAGS.ignore_wakeup, target_obj=FLAGS.target_obj)
    auto_detect()

    
