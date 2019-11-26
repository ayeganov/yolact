import sys
import zmq

import cv2

port = "8000"

# Socket to talk to server
context = zmq.Context()
socket = context.socket(zmq.SUB)

print ("Collecting updates...")
socket.connect ("tcp://localhost:%s" % port)

if len(sys.argv) > 2:
    socket.connect ("tcp://localhost:%s" % port1)

# Subscribe to zipcode, default is NYC, 10001
topicfilter = "10001"
socket.setsockopt_string(zmq.SUBSCRIBE, topicfilter)

# Process 5 updates
total_value = 0
while True:
    string = socket.recv().decode('utf-8')
    topic, message = string.split(':\t')
    data = message.strip().split('\n')
    try:
        data.remove('none')
    except:
        pass

    print ('{}\t-\tdetections:\t{}'.format(topic, len(data)))
