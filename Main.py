
#this sends a notification if someone unauthorized tries to access the PC

# Importing the libraries
import cv2
import numpy as np
import pickle
import os
os.chdir(r"C:\Users\HP\Desktop\New Folder")

labels = {}
with open("label.pickle","rb") as f:
    og_labels = pickle.load(f)
    labels = {v:k for k,v in og_labels.items()}
    
    
from twilio.rest import Client



account_sid = '' 

auth_token = '' 



myPhone = '************' #admin(s) phone number to which the security notification is to be sent

TwilioNumber = '**********' #twilio account number



client = Client(account_sid, auth_token)
                  

# Loading the cascades
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

recognizer  = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainner.yml")  # this is trained weights of face pic of the admin(s) provided in the images/admin directory


cnt = 0
# Defining a function that will do the detections
def detect(gray, frame):
    global cnt
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        id_, conf = recognizer.predict(roi_gray)
        if conf>=45 and conf<=85:
            if(labels[id_] == 'admin'):
                cnt = 0
            else:
                cnt += 1
    return (frame)

# Doing  Face Recognition with the webcam
video_capture = cv2.VideoCapture(0)
id_cnt = 0
while True:
    _, frame = video_capture.read()
    if(cnt >= 8):
        print('sleep')
        client.messages.create(

          to=myPhone,

          from_=TwilioNumber,

          body='Someone accesing your PC' + u'\U0001f680')
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    canvas = detect(gray, frame)
  
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()
