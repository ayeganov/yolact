import sys
import zmq
import json
import time

import cv2
import numpy as np
from matplotlib import pyplot as plt

def demogrify(topicmsg):
    json0 = topicmsg.find('{')
    topic = topicmsg[0:json0].strip()
    msg = json.loads(topicmsg[json0:])
    return topic, msg

ports = []
for i in range(10):
    ports.append(f'800{i}')

# Socket to talk to server
context = zmq.Context()
socket = context.socket(zmq.SUB)

print ("Waiting for data...")
for port in ports:
    socket.connect ("tcp://localhost:%s" % port)

# Subscribe to zipcode, default is NYC, 10001
topicfilter = "10001"
socket.setsockopt_string(zmq.SUBSCRIBE, topicfilter)

# Process 5 updates
total_value = 0
delays = {}
while True:
    start = time.time()
    string = socket.recv_string()
    wait = time.time() - start
    topic, msg = demogrify(string)
    delay = time.time() - 1e-6*msg['frame_id']
    print (delay)
    try:
        delays[msg['sensor_id']].append(delay)
    except:
        delays[msg['sensor_id']] = [delay]

    """
    detections = msg['detections']
    for d in detections:
        for m in d['mask']:
            img = np.array(m, dtype=np.uint8)*255
            cv2.imshow('img', img)
            k = cv2.waitKey(1)
            if k == ord('q'):
                break
    """
