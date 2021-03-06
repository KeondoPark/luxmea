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

if __name__ == "__main__":

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    VOICE_DIR = os.path.join(BASE_DIR, 'voice_guide')
    PROMPT_LIMIT = 5
    WAKE_UP_WORD = ['lux mia', 'lux mea', 'knox mia', 'luxmia', 'luxmea', 'knoxmia', 'Lochmere', 'la mia', 'Locs mia']
    CLASS2TYPE_DICT = {'bed':0, 'table':1, 'sofa':2, 'chair':3, 'toilet':4, 'desk':5, 'dresser':6, 'night_stand':7, 'bookshelf':8, 'bathtub':9,'person':10}
    TYPE2CLASS_DICT = {0:'bed', 1:'table', 2:'sofa', 3:'chair', 4:'toilet', 5:'desk', 6:'dresser', 7:'night_stand', 8:'bookshelf', 9:'bathtub',10:'person'}
    SUPPORTED_CLASS = ['bed','table','chair','toilet','desk','dresser','night_stand','bookshelf','bathtub','person']
    FLAGS = parse_opt()

    # create recognizer and mic instances
    recognizer = sr.Recognizer()
    
    #microphone = sr.Microphone(device_index=2)
    m, idx = -1, 0
    for device_name in sr.Microphone().list_microphone_names():
        if device_name.startswith("ReSpeaker"):
            m = idx
            print("Respeaker found, device_index=", m)
            break
        idx += 1

    cnt = 0    

    if m < 0:
        print("Respeaker not found")        
    else:
        microphone = sr.Microphone(device_index=m)
        
        #for i in range(NUM_GUESSES):
            # get the guess from the user
            # if a transcription is returned, break out of the loop and
            #     continue
            # if no transcription returned and API request failed, break
            #     loop and continue
            # if API request succeeded but no transcription was returned,
            #     re-prompt the user to say their guess again. Do this up
            #     to PROMPT_LIMIT times

    p0_parent_conn, p0_child_conn = Pipe()
    p0 = Process(target=votenet_inference.run_inference, \
                    args=(p0_child_conn,))
    p0.start()
    
    while True:            
        proceedToFindTarget = False

        if not FLAGS.ignore_wakeup:                            
            cnt += 1
            instructions = "Please wake me up by saying Lux Mea"                                    
            playsound('audio_files/wakeup.mp3')                
            listening = recognize_speech_from_mic(recognizer, microphone)
            proceedToFindTarget = listening['transcription'] and listening['transcription'].lower() in WAKE_UP_WORD or cnt >=1
            #proceedToFindTarget = listening['transcription']
        else:
            proceedToFindTarget = True

        if proceedToFindTarget:                            
            while True:       
                if not FLAGS.target_obj in SUPPORTED_CLASS:                    
                    
                    # format the instructions string                    
                    instructions = "What are you looking for?"
                    print(instructions)

                    # show instructions and wait 1 seconds before receiving instruction                                        
                    playsound('audio_files/look_for.mp3')
                    
                    for j in range(PROMPT_LIMIT): 
                        instructions = 'Speak now!'                                                     
                        print(instructions)                         
                        playsound('audio_files/speaknow.mp3')    
                        
                        response = recognize_speech_from_mic(recognizer, microphone)
                        if response['transcription']:
                            break
                        if not response['success']:
                            break
                        if not response["success"]:
                            print("I didn't catch that. What did you say?\n")
                            
                    if response["error"]:
                        print("Error:{}".format(guess["error"]))
                    elif not response["success"]:
                        print("I didn't catch that. Exiting program..\n")
                    else:                               
                        target_obj = response["transcription"].lower()

                        if target_obj in 'chair tear care cher':
                            target_obj = 'chair'  
                        if target_obj in 'desk task deck that':
                            target_obj = 'desk'
                        if target_obj in 'table fable':
                            target_obj = 'table'                         

                        # show the user the transcription                        
                        instructions = "You said: {}".format(target_obj)                        
                        myOutput=gTTS(text=instructions, lang='en', slow=False)
                        myOutput.save('audio_files/saidobject.mp3')
                        print(instructions)                         
                        playsound('audio_files/saidobject.mp3')
                        
                        if target_obj == 'exit':
                            instructions = "Exiting program. Bye!"                                                    
                            print(instructions)                         
                            playsound('audio_files/bye.mp3')                            
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

                    # if streaming server is on, send request and get depth image
                    # Otherwise, control the camera directly
                    import socket
                    a_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

                    #location = (SERVER_ADDRESS, SERVER_PORT)
                    location = (SERVER_ADDRESS, SERVER_PORT)
                    res = a_socket.connect_ex(location)

                    server_on = True if res == 0 else False

                    # and then check the response...
                    if server_on:                        
                        print("Server Active")                    
                        p1 = Process(target=send_request_wrapper, args=(q1,))
                    else:
                        print("Server Inactive")                    
                        p1 = Process(target=rs_snapshot_rotation.take_snapshot_rotation, args=(q1,))                                            
                        
                    
                    p1.start()
                    
                    instructions = "I will find it for you, please wait for a moment"                                                                                    
                    #myOutput=gTTS(text=instructions, lang='en', slow=False)
                    #myOutput.save('audio_files/finditforyou.mp3')             
                    print(instructions)                         
                    playsound('audio_files/finditforyou.mp3')   
                    
                    
                    #p1.join()
                    print("P1 joined")                    
                    #point_cloud = q1.get()                    
                    #img_path = q1.get()
                    #(Rtilt, K) = q1.get()
                    #q1.close()                                       
                                        
                    p0_parent_conn.send(q1.get())
                    p0_parent_conn.send(q1.get())
                    p0_parent_conn.send(q1.get())
                    #p0_parent_conn.send("START_INFERENCE")

                    
                    #p0.join()                    
                    #predicted_class = q0.get()     
                    predicted_class = p0_parent_conn.recv()                    
                    end = time.time()
                    print("Total runtime multi processing", end - start)                      
                    #p0.terminate()         
                    
                    
                    targets = []
                    person_found = []
                    boxes = []

                    if server_on:
                        #Turn on streaming again                    
                        pause_res = requests.get("http://" + SERVER_ADDRESS_PORT + "/pause_onoff")

                    for item in predicted_class[0]:
                        if target_obj == 'desk' and item[0] in [CLASS2TYPE_DICT['desk'], CLASS2TYPE_DICT['table']] \
                        or item[0] == CLASS2TYPE_DICT[target_obj]:
                        #if item[0] == 'chair':
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
                        instructions = "Sorry no object was found"
                        print(instructions)
                        playsound('audio_files/noobject.mp3')                       

                    for target in targets:
                        if len(person_found) == 0:
                            #WHen there is no person found, Guide is based on camera location
                            found = {}
                            found['center'] = np.array([0.0, 0.0, 0.0])
                            found['heading_angle'] = math.pi / 2 
                            person_found.append(found)
                            print("Person Found!!")
                            
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

                        instructions = 'There is a ' + target_obj + " {:.2f}".format(distance) + 'meter away at ' + str(direction) + "o'clock direction"
                        
                        myOutput=gTTS(text=instructions, lang='en', slow=False)
                        myOutput.save('audio_files/guidance.mp3')             
                        print(instructions)

                        if server_on:
                            requests.post("http://" + SERVER_ADDRESS_PORT + "/get_boxes", json={'boxes':boxes})
                        
                        playsound('audio_files/guidance.mp3')

                    if p1.is_alive():
                        print("P1 still alive, kill")
                        p1.terminate()
                        p1.join()                    
                    
                    time.sleep(3)

            
                        
            break
    
