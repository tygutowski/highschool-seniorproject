import asyncio
import io
import glob
import os
import sys
import pathlib
import time
import uuid
import requests
import json
from config import *
from PIL import Image
from urllib.parse import urlparse
from io import BytesIO
from PIL import Image, ImageDraw
from azure.cognitiveservices.vision.face import FaceClient
from msrest.authentication import CognitiveServicesCredentials
from azure.cognitiveservices.vision.face.models import TrainingStatusType, Person, SnapshotObjectType, OperationStatusType
import numpy as np
import cv2

cam = cv2.VideoCapture(0)

face_client = FaceClient(ENDPOINT, CognitiveServicesCredentials(KEY))

TARGET_GROUP_ID = str(uuid.uuid4())

PATH = pathlib.Path().absolute()
ALL_PATH = os.path.join(PATH, "all")
FRIENDLY_PATH = os.path.join(PATH, "friendly1")

GROUP_ID = 'authorized'
print('Person group:', GROUP_ID)
#This is to create (NOT RETRAIN) a new persongroup (if ever needed)
'''
face_client.person_group.delete(person_group_id=GROUP_ID,name=GROUP_ID)
face_client.person_group.create(person_group_id=GROUP_ID, name=GROUP_ID)
friendly_group = face_client.person_group_person.create(GROUP_ID, "friendly_group")
'''
#Gets all images of friendly
friendly_images = os.listdir(FRIENDLY_PATH)
all_images = os.listdir(ALL_PATH)
#Gets paths of all images of friendly
friendly_list = []
all_list = []

for index, image in enumerate(friendly_images):
    friendly_list.append(os.path.join(FRIENDLY_PATH, friendly_images[index]))
    
for index, image in enumerate(all_images):
    all_list.append(os.path.join(ALL_PATH, all_images[index]))

name = "webcam.jpg"
x=0
while(True):
    # Capture frame-by-frame
    ret, frame = cam.read()
    # Display the resulting frame
    cv2.imshow('Original', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    time.sleep(.1)
    if(x==30):
        x=0
        cv2.imwrite(os.path.join(ALL_PATH, name), frame)
        face_ids = []
        for index, image in enumerate(all_list):
            open_image = open(all_list[index], 'r+b')
            faces = face_client.face.detect_with_stream(open_image)
            for face in faces:
                face_ids.append(face.face_id)
        
        if(len(face_ids)>0):
            results = face_client.face.identify(face_ids, GROUP_ID)
            for person in results:
                if(len(person.candidates)>0):
                    print('Person for face ID {} is identified with a confidence of {:0.2f}%.'.format(person.face_id, person.candidates[0].confidence*100))
                else:
                    print('Person for face ID {} is identified in and is not authorized.'.format(person.face_id))
        else:
            print('No person identified.'.format(ALL_PATH))
    x+=1


# When everything done, release the capture
cam.release()
cv2.destroyAllWindows()
