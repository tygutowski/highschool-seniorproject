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
from myconfig import *
from PIL import Image
from urllib.parse import urlparse
from io import BytesIO
from PIL import Image, ImageDraw
from azure.cognitiveservices.vision.face import FaceClient
from msrest.authentication import CognitiveServicesCredentials
from azure.cognitiveservices.vision.face.models import TrainingStatusType, Person, SnapshotObjectType, OperationStatusType
import numpy as np
import cv2

face_client = FaceClient(ENDPOINT, CognitiveServicesCredentials(KEY))

TARGET_GROUP_ID = str(uuid.uuid4())

PATH = pathlib.Path().absolute()
ALL_PATH = os.path.join(PATH, "all")
FRIENDLY_PATH = os.path.join(PATH, "friendly1")

GROUP_ID = 'authorized'
print('Person group:', GROUP_ID)
face_client.person_group.delete(person_group_id=GROUP_ID,name=GROUP_ID)
face_client.person_group.create(person_group_id=GROUP_ID, name=GROUP_ID)
friendly_group = face_client.person_group_person.create(GROUP_ID, "friendly_group")

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
    
for image in friendly_list:
    friendly_group_image = open(image, 'r+b')
    face_client.person_group_person.add_face_from_stream(GROUP_ID, friendly_group.person_id, friendly_group_image)

print("Training person group.")

face_client.person_group.train(GROUP_ID)

while(True):
    training_status = face_client.person_group.get_training_status(GROUP_ID)
    print("Training Status: {}.".format(training_status.status))
    if(training_status.status is TrainingStatusType.succeeded):
        break
    elif(training_status.status is TrainingStatusType.failed):
        sys.exit("Training the person group has failed.")
    time.sleep(5)
