import zmq
import zmq.eventloop
import zmq.eventloop.zmqstream


def pub_to_zmq(endpoint):
    '''
    Create a publisher socket for communicating with the Conductor before ROS
    is running.

    @param endpoint - zmq address to publish on
    '''
    context = zmq.Context.instance()

    socket = context.socket(zmq.PUB)
    socket.bind(endpoint)
    return socket


def sub_to_zmq(address, callback, loop):
    '''
    Subscribe to zmq to receive the stop command
    '''
    ctx = zmq.Context.instance()
    socket = ctx.socket(zmq.SUB)
    socket.setsockopt(zmq.SUBSCRIBE, b"")
    socket.connect(address)
    stream = zmq.eventloop.zmqstream.ZMQStream(socket, loop)
    stream.on_recv(callback)
    return stream


def pair_to_zmq(address, callback, loop, is_server=False):
    '''
    Create a pair stream socket

    @param address - zmq address string. ie: ipc:///some_string, or tcp://localhost:5555
    @param callback - callback function to be invoked when a message is received
    @param loop - instance of IOLoop to use
    @param is_server - boolean indicating whether this is a server socket
    @return ZMQStream instance for a pair socket
    '''
    ctx = zmq.Context.instance()
    socket = ctx.socket(zmq.PAIR)
    if is_server:
        socket.bind(address)
    else:
        socket.connect(address)
    stream = zmq.eventloop.zmqstream.ZMQStream(socket, loop)
    stream.on_recv(callback)
    return stream
