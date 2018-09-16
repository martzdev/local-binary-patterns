# Face recognition - developed by Sturza Mihai
# Using photos provided by the collection software,
# I'm using a local binary pattern histogram to detect known faces

# import libraries
import cv2, json
import numpy as np
from os import listdir
from os.path import isfile, join

# §§§§§§§§§§§§§
# TRAINING PART
# §§§§§§§§§§§§§

# get data from config.json
with open('config.json') as conf:
    config = json.load(conf)

name = config['name']

# find the images needed to train the classifier
data_path = './faces/'+name+'/'
onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path, f))]

# initialize the training matrix
Training_Data, Labels = [], []

# insert training data
for i, files in enumerate(onlyfiles):
    image_path = data_path + onlyfiles[i]
    images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    Training_Data.append(np.asarray(images, dtype=np.uint8))
    Labels.append(i)

Labels = np.asarray(Labels, dtype=np.int32)

# create and train the model
model = cv2.face.LBPHFaceRecognizer_create(radius=3,neighbors=9,grid_x=10,grid_y=10)
model.train(np.asarray(Training_Data), np.asarray(Labels))

# print confirmation
print("Model trained sucessefully")

# §§§§§§§§§§§§§§§§§§§
# CLASSIFICATION PART
# §§§§§§§§§§§§§§§§§§§

# crate the haar cascade
face_classifier = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')

# function for detecting faces in the image and return only the part with a face
def face_detector(img, size=0.5):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    if faces is ():
        return img, []
    
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),2)
        roi = img[y:y+h, x:x+w]
        roi = cv2.resize(roi, (200, 200))
    return img, roi

# initialize the video feed
cap = cv2.VideoCapture(0)

while True:
    # read frames from the camera
    ret, frame = cap.read()
    image, face = face_detector(frame)
    
    try:
        # make a prediction using the trained model
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        results = model.predict(face)
        
        # get results
        if results[1] < 500:
            # the 'confidence' can be negative 
            confidence = int( 100 * (1 - (results[1])/400) )
            display_string = str(confidence) + '% confident it is '+name # prepare the string that will be printed
        
        # print info and display image
        cv2.putText(image, display_string, (100, 120), cv2.FONT_HERSHEY_COMPLEX, 1, (255,120,150), 2)
        cv2.imshow('Face Recognition', image )

    except:
        # no face found!
        cv2.putText(image, "No Face Found", (220, 120) , cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
        cv2.imshow('Face Recognition', image )
        pass
        
    if cv2.waitKey(1) == 13: # space key
        break
        
cap.release()
cv2.destroyAllWindows()