import random
import time

import speech_recognition as sr
import rs_snapshot
import transform
import votenet_inference
import math
import numpy as np
import os
from gtts import gTTS
from playsound import playsound
from multiprocessing import Process, Queue
import rs_snapshot_rotation
import requests


#def play_sound():
#    play_finditforyou()


def send_request_wrapper(queue):
    res = requests.get("http://127.0.0.1:5000/depth_snapshot")
    print(res)
    filename = res.json()['filename']
    queue.put(filename)



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
    CLASS2TYPE_DICT = {'bed':0, 'table':1, 'sofa':2, 'chair':3, 'toilet':4, 'desk':5, 'dresser':6, 'night_stand':7, 'bookshelf':8, 'bathtub':9}
    TYPE2CLASS_DICT = {0:'bed', 1:'table', 2:'sofa', 3:'chair', 4:'toilet', 5:'desk', 6:'dresser', 7:'night_stand', 8:'bookshelf', 9:'bathtub'}
    SUPPORTED_CLASS = ['bed','table','chair','toilet','desk','dresser','night_stand','bookshelf','bathtub']
    #SUPPORTED_CLASS = ['chair']
    
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
        while True:
            cnt += 1            
            instructions = "Please wake me up by saying Lux Mea"                        
            myOutput=gTTS(text=instructions, lang='en', slow=False)
            myOutput.save('audio_files/wakeup.mp3')
            print(instructions)            
            playsound('audio_files/wakeup.mp3')
            
            listening = recognize_speech_from_mic(recognizer, microphone)
            #if listening['transcription']:
                #print("Captured voice:", listening['transcription'])

                #myText="You said " + listening['transcription']
                #myOutput=gTTS(text=myText, lang='en', slow=False)
                #myOutput.save('audio_files/captured_voice.mp3')
                #print(myText)
                #playsound('audio_files/captured_voice.mp3')

            if listening['transcription']:
            #if listening['transcription'] and listening['transcription'].lower() in WAKE_UP_WORD or cnt >=3: 
                           
                while True:
                    # format the instructions string                    
                    instructions = "What are you looking for?"
                    
                    myOutput=gTTS(text=instructions, lang='en', slow=False)
                    myOutput.save('audio_files/look_for.mp3')                    
                    print(instructions)

                    # show instructions and wait 1 seconds before receiving instruction                                        
                    playsound('audio_files/look_for.mp3')
                    
                    for j in range(PROMPT_LIMIT): 
                        instructions = 'Speak now!'                        
                        myOutput=gTTS(text=instructions, lang='en', slow=False)
                        myOutput.save('audio_files/speaknow.mp3')                   
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
                            myOutput=gTTS(text=instructions, lang='en', slow=False)
                            myOutput.save('audio_files/bye.mp3')
                            print(instructions)                         
                            playsound('audio_files/bye.mp3')
                            
                            break

                        if target_obj in SUPPORTED_CLASS:                            
                            start = time.time()   
                            q0 = Queue()                             
                            p0 = Process(target=votenet_inference.votenet_inference, args=(q0,))
                            p0.start()  

                            print("Starting to take depth snapshot")
                            q1 = Queue()
                            #p1 = Process(target=rs_snapshot_rotation.take_snapshot_rotation, args=(q1,))
                            p1 = Process(target=send_request_wrapper, args=(q1,))
                            p1.start()

                            
                            instructions = "I will find it for you, please wait around 20 seconds"                                               
                            myOutput=gTTS(text=instructions, lang='en', slow=False)
                            myOutput.save('audio_files/finditforyou.mp3')                   
                            print(instructions)                         
                            playsound('audio_files/finditforyou.mp3')   
                            
                            p1.join()
                            snapshot_filename = q1.get()
                            
                            q2 = Queue()
                            q2.put(snapshot_filename)
                            p2 = Process(target=rs_snapshot_rotation.rotation_only, args=(q2,))
                            p2.start()
                            p2.join()
                            rotated_ply = q2.get()                           

                            q0.put(rotated_ply)                            
                            p0.join()
                            
                            predicted_class = q0.get()

                            end = time.time()
                            print("Total runtime multi processing", end - start)                            
                            
                            targets = []

                            for item in predicted_class[0]:
                                if target_obj == 'table' and item[0] in [CLASS2TYPE_DICT['desk'], CLASS2TYPE_DICT['table']] \
                                or item[0] == CLASS2TYPE_DICT[target_obj]:
                                #if item[0] == 'chair':
                                    found = {}
                                    found['center'] = np.mean(item[1], axis=0)
                                    found['dist'] = np.linalg.norm(found['center'])
                                    found['score'] = item[2]
                                    targets.append(found)

                            targets.sort(key=lambda x: x['score'])

                            if len(targets) == 0:
                                instructions = "Sorry no object was found"
                                myOutput=gTTS(text=instructions, lang='en', slow=False)
                                myOutput.save('audio_files/noobject.mp3')
                                print(instructions)                               
                               
                                playsound('audio_files/noobject.mp3')                       

                            for target in targets:
                                tan = target['center'][2] / target['center'][0]
                                if tan > math.tan(5 * math.pi/12) or tan < math.tan(7 * math.pi/12):
                                    direction = 12
                                elif tan > math.tan(3 * math.pi/12):
                                    direction = 1
                                elif tan < math.tan(9 * math.pi/12):
                                    direction = 11
                                elif tan > math.tan(1 * math.pi/12):
                                    direction = 2
                                elif tan < math.tan(11 * math.pi/12):
                                    direction = 10
                                elif tan > 0:
                                    direction = 3
                                elif tan < 0:
                                    direction = 9

                                instructions = 'There is a ' + target_obj + " {:.2f}".format(target['dist']) + 'meter away at ' + str(direction) + "o'clock direction"
                                
                                myOutput=gTTS(text=instructions, lang='en', slow=False)
                                myOutput.save('audio_files/guidance.mp3')             
                                print(instructions)
                                
                                playsound('audio_files/guidance.mp3')

                            if p1.is_alive():
                                p1.terminate()
                                p1.join()
                            else:
                                p1.join()
                            
                            #Turn on streaming again
                            pause_res = requests.get("http://127.0.0.1:5000/puase_onoff")
                            time.sleep(3)
                                
                            print(predicted_class)
                            
                break
    
