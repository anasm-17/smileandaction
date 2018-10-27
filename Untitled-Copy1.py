#!/usr/bin/env python
# coding: utf-8

# In[2]:


import cv2
import numpy as np
from keras.models import load_model
from statistics import mode
from utils.datasets import get_labels
from utils.inference import detect_faces
from utils.inference import draw_text
from utils.inference import draw_bounding_box
from utils.inference import apply_offsets
from utils.inference import load_detection_model
from utils.preprocessor import preprocess_input


# In[1]:


USE_WEBCAM = False # If false, loads video file source


# In[4]:


# parameters for loading data and images
emotion_model_path = 'models/emotion_model.hdf5'
emotion_labels = get_labels('fer2013')


# In[5]:


# hyper-parameters for bounding boxes shape
frame_window = 10
emotion_offsets = (20, 40)


# In[ ]:


# loading models
face_cascade = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')
emotion_classifier = load_model(emotion_model_path)


# In[ ]:


# getting input model shapes for inference
emotion_target_size = emotion_classifier.input_shape[1:3]


# In[ ]:


# starting lists for calculating modes
emotion_window = []


# In[ ]:


import time
import os
cwd = os.getcwd()
cwd
smile_count=5


# In[ ]:


height = 768
width = 1280

# height, width, number of channels in image
outter_rect_ix = int(0.15*width)
outter_rect_iy = int(0.96*height)
outter_rect_jx = int(0.85*width)
outter_rect_jy = int(0.94*height)

inner_rect_ix = outter_rect_ix
inner_rect_iy = outter_rect_iy
inner_rect_jx = int(smile_count*(1/1000)*outter_rect_jx)
inner_rect_jy = outter_rect_jy


# In[ ]:


# Select video or webcam feed
cap = None
if (USE_WEBCAM == True):
    cap = cv2.VideoCapture(0) # Webcam source
else:
    cap = cv2.VideoCapture('demo/Happy_Face.mp4') # Video file source
    cv2.namedWindow('window_frame')

t0=time.time()
t1=time.time()
print_count=0
while cap.isOpened(): # True:
    ret, bgr_image = cap.read()

    #bgr_image = video_capture.read()[1]

    gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5,
			minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

    for face_coordinates in faces:

        x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
        gray_face = gray_image[y1:y2, x1:x2]
        try:
            gray_face = cv2.resize(gray_face, (emotion_target_size))
        except:
            continue

        gray_face = preprocess_input(gray_face, True)
        gray_face = np.expand_dims(gray_face, 0)
        gray_face = np.expand_dims(gray_face, -1)
        emotion_prediction = emotion_classifier.predict(gray_face)
        emotion_probability = np.max(emotion_prediction)
        emotion_label_arg = np.argmax(emotion_prediction)
        emotion_text = emotion_labels[emotion_label_arg]
        emotion_window.append(emotion_text)

        if len(emotion_window) > frame_window:
            emotion_window.pop(0)
        try:
            emotion_mode = mode(emotion_window)
        except:
            continue

        if emotion_text == 'angry':
            color = emotion_probability * np.asarray((255, 0, 0))
        elif emotion_text == 'sad':
            color = emotion_probability * np.asarray((0, 0, 255))
        elif emotion_text == 'happy':
            color = emotion_probability * np.asarray((255, 255, 0))
        elif emotion_text == 'surprise':
            color = emotion_probability * np.asarray((0, 255, 255))
        else:
            color = emotion_probability * np.asarray((0, 255, 0))

        color = color.astype(int)
        color = color.tolist()
        e_p = str(round(emotion_probability*100,2))
        draw_bounding_box(face_coordinates, rgb_image, color)
        draw_text(face_coordinates, rgb_image, emotion_mode+" "+e_p+"%",
                  color, 0, -45, 0.5, 1)



    try:     

        inner_rect_jx = int(smile_count*(1/1000)*outter_rect_jx)
        
        if (emotion_text =='happy'):
            if (smiles_count == 1000):
                    break
            smiles_count +=5

    except Exception as e:
        continue
        
    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    cv2.rectangle(bgr_image,(outter_rect_ix,outter_rect_iy),(outter_rect_jx,outter_rect_jy),(0,255,255),3)
    cv2.rectangle(bgr_image,(inner_rect_ix,inner_rect_iy),(inner_rect_jx,inner_rect_jy),(0,255,0),-1)
    cv2.imshow('window_frame', bgr_image)    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


# In[ ]:
