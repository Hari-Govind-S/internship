from ultralytics import *
import math
import numpy as np
import cv2
import cvzone
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D

'''
#model = tf.keras.models.load_model('my_model.keras')
model = Sequential([
    Conv2D(filters=32, kernel_size=(3,3), padding='Same', activation='relu', input_shape=(180,180,3)),
    MaxPool2D(pool_size=(2,2)),
    Conv2D(filters=64, kernel_size=(3,3), padding='Same', activation='relu'),
    MaxPool2D(pool_size=(2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.load_weights('model_weight.data-00000-of-00001') 

def preprocess_frame(frame):
    # Resize the frame to match the model's input size
    resized_frame = cv2.resize(frame, (224, 224))
    img = np.expand_dims(resized_frame, axis=0)
    img = preprocess_input(img)
    return img

# Classify the image
def classify_frame(frame):
    img = preprocess_frame(frame)
    predictions = model.predict(img)
    decoded_predictions = decode_predictions(predictions)
    return decoded_predictions


video = cv2.VideoCapture('uploads/Source.mp4')
frame_width = int(video.get(3))
frame_height = int(video.get(4))
y = np.zeros([frame_height,frame_width], dtype=np.uint8)
n = 0
while True:
    ret, frame = video.read()
    gray1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    a = np.asarray(gray1)
    #print(a)
    if(n>50):
        for i in range(frame_width):
            for j in range(frame_height):
                if(a[j][i] >= b[j][i]):
                    y[j][i] = a[j][i]-b[j][i]
                else:
                    y[j][i] = b[j][i] - a[j][i]
                if(y[j][i]>45):
                     y[j][i] = 244
                else:
                     y[j][i] = 0
        #y = np.subtract(a,b)
        #print(y)

        #threshed = cv2.adaptiveThreshold(y, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 0)
        #diff = cv2.subtract(gray1, gray2)
        ksize = (10, 10) 
  
        # Using cv2.blur() method  
        image = cv2.blur(y, ksize)
        adjusted = cv2.convertScaleAbs(image,alpha = 20,beta = 0.2)
        #cv2.imshow("diff(img1, img2)", adjusted)
        objectdetector = cv2.createBackgroundSubtractorMOG2(history=400, varThreshold=200)
        mask = objectdetector.apply(gray2)
        mask = objectdetector.apply(gray1)
        #cv2.imshow('frame', mask)
        contours, _ = cv2.findContours(adjusted, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        prt = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        for cnt in contours:
                    area = cv2.contourArea(cnt)
                    x1, y1, w, h = cv2.boundingRect(cnt)
                    x2 = x1 + w
                    y2 = y1 + h
                    cx = x1 + (w / 2 )
                    cy = y1 + (h / 2)
                    if area > 100 and area < 20000:# and w > 50 and h > 50 and w < 300 and h < 300:
                        region_of_interest = frame[y1:y1+h, x1:x1+w]
                        predictions = classify_frame(region_of_interest)
                        if(predictions[0][0] < 0.5):
                            cvzone.cornerRect(prt, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))
        cv2.imshow('frame', prt)
        cv2.waitKey() 
    gray2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    b = np.asarray(gray2)
    n += 1
'''
def objdet(cur_frame, gray2, h, w):
    y = np.zeros([h,w], dtype=np.uint8)
    gray1 = cv2.cvtColor(cur_frame, cv2.COLOR_BGR2GRAY)
    a = np.asarray(gray1)
    b = np.asarray(gray2)
    for i in range(w):
            for j in range(h):
                if(a[j][i] >= b[j][i]):
                    y[j][i] = a[j][i]-b[j][i]
                else:
                    y[j][i] = b[j][i] - a[j][i]
                if(y[j][i]>45):
                     y[j][i] = 244
                else:
                     y[j][i] = 0
    ksize = (10, 10)  
    image = cv2.blur(y, ksize)
    adjusted = cv2.convertScaleAbs(image,alpha = 20,beta = 0.2)
    return y
