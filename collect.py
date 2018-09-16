# import libraries
import cv2, json
import numpy as np

# crate the haar cascade
face_classifier = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')

# read config file
with open('config.json') as conf:
    config = json.load(conf)

name = config['name']
samples = config['samples']

# function to detect faces and return the cropped face
def face_extractor(img):
    
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    
    if faces is ():
        return None
    
    for (x,y,w,h) in faces:
        cropped_face = img[y:y+h, x:x+w]
    
    return cropped_face

# initialize video
cap = cv2.VideoCapture(0)
count = 0

while True:
    # read frames
    ret, frame = cap.read()
    # get faces
    if face_extractor(frame) is not None:
        count += 1
        face = cv2.resize(face_extractor(frame), (200, 200))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        
        # prepare the path ! MAKE SURE THE FOLDERS ARE ALREADY CREATED AS OPENCV WILL NOT CREATE THEM !
        file_name_path = './faces/'+name+'/'+ str(count) + '.jpg'
        cv2.imwrite(file_name_path, face)
        
        # inform the user about how many samples were collected
        cv2.putText(face, str(count), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
        cv2.imshow('Sampler Collector', face)
    
    else:
        print("Face not found")
        pass
    
    if cv2.waitKey(1) == 13 or count == samples: # running this until the user presses enter or 
        break                                    # we have collected the number of sample data provided in the config.json

cap.release()
cv2.destroyAllWindows()
