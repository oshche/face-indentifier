
# coding: utf-8

# In[11]:

import cv2,os
import cv2.face
import numpy as np
from PIL import Image

cascadePath = "D:\\shche\\OpenCVDemo\\haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cascadePath)
recognizer = cv2.face.LBPHFaceRecognizer_create(1,8,8,8,123)

path = 'D:\\face\\yalefaces'
image_paths_train = [os.path.join(path, f) for f in os.listdir(path) if not f.endswith('.happy')]
image_paths_test = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.happy')]


# In[14]:

print(len(image_paths_train))
print(len(image_paths_test))


# In[16]:

images = []
names = []
for im_path in image_paths_train:
    image = Image.open(im_path)
    grayscale_image = image.convert('L')
    array_image = np.array(grayscale_image,'uint8') 
    
    person_on_image = int((os.path.split(im_path)[1].split(".")[0].replace("subject", "")))
    faces = face_cascade.detectMultiScale(array_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    for (x, y, w, h) in faces:
            #cv2.rectangle(array_image,(x,y),(x+w,y+h),(255,0,0),2)
            images.append(array_image[y: y + h, x: x + w])
            names.append(person_on_image)
            cv2.imshow("Adding faces to traning set",array_image[y: y + h, x: x + w])
            cv2.waitKey(50)
            cv2.destroyAllWindows()
            
    #image.show()


# In[17]:

recognizer.train(images,np.array(names))


# In[20]:

for im_path in image_paths_test:
    gray = Image.open(im_path).convert('L')
    image = np.array(gray, 'uint8')
    faces_test = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    
    for (x, y, w, h) in faces_test:
        number_predicted, conf = recognizer.predict(image[y: y + h, x: x + w])
        number_real = int(os.path.split(im_path)[1].split(".")[0].replace("subject", ""))
        #print(number_real,number_predicted, conf)
        if number_real == number_predicted:
            print( "{} is Correctly indentified with confidence {}".format(number_real, conf))
        else:
            print ("{} is Incorrect indentified as {}".format(number_actual, number_predicted))
        cv2.imshow("Recognizing Face", image[y: y + h, x: x + w])
        cv2.waitKey(50)
        cv2.destroyAllWindows()


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



