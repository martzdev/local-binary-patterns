# person-recognition
software that detects specific persons using your webcam
### config.json
- you can change the name of the person that you will be classifing
- you can change the number of samples ( pictures that are taken using 'collector.py' )
![alt text](https://imgur.com/YOVHkXz)
### collector.py
- scans for faces using haarcascades and saves only the part where the face is
#### you need to create the folder 'faces' and inside it another folder named like in the config.json
![alt text](https://imgur.com/AjJgH0c)
### train_and_detect.py
#### If you get a 'NoneType' error from the numpy.asarray function, make sure to delete the .DS_Store as the program sees it as None
- using the samples provided by collector.py i am using a local binary pattern histogram to detect a specific face
- after the classification was made, inside the window will be printed a confidence score
![alt text](https://imgur.com/N9mgKVT)
- it still doesn't provide the most accurate prediction but in the future i might use a cnn as the classifier
