import os
import numpy as np
from PIL import Image
import cv2
import pickle

 #this is .py for training the model using images provided by admins
# Loading the cascades
face_cascade = cv2.CascadeClassifier(r'haarcascade_frontalface_alt.xml')

BASE_DIR = os.getcwd()
image_dir = os.path.dirname(r'.\images')


recognizer  = cv2.face.LBPHFaceRecognizer_create()


current_id = 0
label_ids = {} #creating a dictionary which stores the integer labels for admin and non-admin

y_labels = []
x_train = []
for  root,dirs,files in os.walk(image_dir):
    for file in files:
        if file.endswith("png") or file.endswith("jpg"):
            path = os.path.join(root,file)
            label = os.path.basename(os.path.dirname(path)).replace(" ","-").lower()
            print(path)
            if label in label_ids:
                pass
            else:
                label_ids[label] = current_id
                current_id+=1

            id_ = label_ids[label]
            pil_image = Image.open(path) .convert("L")#give inmage on that path and convert it into gray
            image_array = np.array(pil_image,"uint8")
            faces = face_cascade.detectMultiScale(image_array, 1.3, 5)

            for (x,y,w,h) in faces:
                 roi = image_array[y:y+h,x:x+w]
                 x_train.append(roi)
                 y_labels.append(id_)


with open("label.pickle","wb") as f:
    pickle.dump(label_ids,f)   #creates file with the labels of admin and non-admin



recognizer.train(x_train,np.array(y_labels))      #training of using local binary histogram pattern recognition
recognizer.save("trainner.yml")
