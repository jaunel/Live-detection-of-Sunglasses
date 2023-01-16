# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 14:49:43 2023

@author: Jaunel
"""


#Importing the libraries
import tensorflow as tf
import numpy as np
import cv2
import datetime
from tensorflow.keras.utils import load_img
from tensorflow import keras
from keras.layers import Dense,Conv2D,MaxPooling2D,Flatten,BatchNormalization

from keras.preprocessing.image import ImageDataGenerator


#Live detection of Sunglasses
mymodel= tf.keras.models.load_model("D:\Jaunel\Learning\DL\computer vision\sunglasses/mymodel.h5")

cap=cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')

while cap.isOpened():
    ret,img = cap.read()
    face = face_cascade.detectMultiScale(img,scaleFactor = 1.1,minNeighbors=5
            )
    for(x,y,w,h) in face:
        face_img = img[y:y+h, x:x+w] #cropping the image with given coordinates
        cv2.imwrite('D:/Jaunel/Learning/DL/computer vision/sunglasses/images/temp.jpg',face_img)
        test_image=load_img("D:/Jaunel/Learning/DL/computer vision/sunglasses/images/temp.jpg",target_size=(256,256,3))
        test_image=tf.keras.utils.img_to_array(test_image)
        test_image=np.expand_dims(test_image,axis=0)
        pred = mymodel.predict(test_image)[0][0]
        if pred ==1:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),3)
            cv2.putText(img,'With Sunglasses',((x+w)//2,y+h+20),cv2.FONT_HERSHEY_SIMPLEX,
                        1,(255,0,0),3)
        else:
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)
            cv2.putText(img,'Without Sunglasses',((x+w)//2,y+h+20),cv2.FONT_HERSHEY_SIMPLEX,
                        1,(0,255,0),3)

        
    cv2.imshow("img",img)
    if cv2.waitKey(1)==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()