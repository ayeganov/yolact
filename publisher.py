import zmq
import random
import sys
import time

port = "8000"

context = zmq.Context()
socket = context.socket(zmq.PUB)
socket.bind("tcp://*:%s" % port)

while True:
    topic = random.randrange(9999,10005)
    messagedata = random.randrange(1,215) - 80
    print ("%d %d" % (topic, messagedata))
    socket.send_string("%d:\t%d" % (topic, messagedata))
    time.sleep(1)
