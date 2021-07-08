# LUXMEA
LuxMea("My Lihgt" in latin) is developed to help visually impared people find objects indoors. 


## Devices
Several devices are used to run LuxMea.
- Depth camera(Intel Realsense D435i): Distance information is acquired from depth camera and it is not affected by the brightness of the environment, therefore suitable for our purpose. 
- Edge devices(Jetson Nano): It is cheap and powerful to run deep learning applications. 
- Mic/Speaker(Respeaker USB Mic Array / Britz Orion speaker): Required for sound interface. Remember that our target users are visually impaired people.
- Coral EdgeTPU(M.2 or USB accelerator)

## Prerequisites
Followings should be installed on Jetson Nano
- Realsense SDK
- pytorch
- TFlite
- flask(If use streaming server)
- numpy, scipy
- gtts, playsound(For sound interface)

## How to start LuxMea
1. Without Streaming server

    You can start LuxMea with the following command.
    ```
    python3 voice_command.py
    ```

    It will ask you to wake up. "LuxMea" is our wake-up word.
    If you made it to wake up, you can say objects you are looking for. 10 Objects are supported: 'bed','table','chair','toilet','desk','dresser','night_stand','bookshelf','bathtub'.

    You can give `--ignore-wakeup` to ignore wake-up and `--target-obj [SUPPORTED_OBJ]` to give the target from command.

2. With Straeming server

    You can stream the video from depth camera to webpage. This is useful in test stage to see whether depth camera is watching the appropriate scene. Use following command to turn on the streaming server.
    ```
    python3 streaming_server.py
    ```
    Video frame will be streamed through port 5000. You can watch the video from browser with address as `192.168.0.50:5000` the ip address might be different depending on Jetson Nano's address.




