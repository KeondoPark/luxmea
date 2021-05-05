from playsound import playsound
from gtts import gTTS

def play_sound(queue):
    
    instructions = queue.get()
    filename = queue.get()    
    myOutput=gTTS(text=instructions, lang='en', slow=False)
    myOutput.save(filename + '.mp3')          
    print(instructions)                                  
    playsound(filename + '.mp3') 

if __name__=='__main__':
    play_finditforyou()