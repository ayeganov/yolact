'''
This module defines the process pool for the neural networks that can be used
for inference
'''
from collections import namedtuple
from ctypes import c_uint8
from multiprocessing import Process
from multiprocessing.sharedctypes import Array
import abc
import enum
import functools
import logging
import operator
import queue
import resource
import sys

from logzero import logger
from tornado import ioloop, gen
import bunch
import logzero
import numpy as np
import torch
import torch.nn.functional as F

from eval import CustomDataParallel
from image_pb2 import Image
from layers.box_utils import crop, sanitize_coordinates
from utils.augmentations import FastBaseTransform
from yolact import Yolact
import comm
import wraploop


class NetworkType(enum.Enum):
    '''
    List of supported NN types
    '''
    Yolact = 0


class DeserializerType(enum.Enum):
    '''
    List of supported network input types
    '''
    Image = 0


class SerializerType(enum.Enum):
    '''
    List of supported output types
    '''
    DetectionBox = 0


class Command(enum.Enum):
    '''
    List of supported NN process commands
    '''
    DoWork = 0  # There is work to do
    Shutdown = 1  # We're done, shutdown cleanly


class Response(enum.Enum):
    '''
    List of supported response types from NN processes
    '''
    READY = 0  # expected to be sent once the process initializes the weights
    RESULT = 1  # work finished, result available


class ProcessState(enum.Enum):
    '''
    List of possible states of a NN process
    '''
    INITIALIZING = 0  # initial state of the process, no jobs can be processed in this state
    AVAILABLE = 1  # the only state in which the process can receive and process data
    BUSY = 2  # process is busy processing some data
    BROKEN = 3  # something went wrong and the process broke


class FactoryException(Exception):
    '''
    Raise when trying to create an unknown entity
    '''


class UnknownDeserializer(Exception):
    '''
    Raise when trying to create an unknown deserializer function
    '''


class UnknownSerializer(Exception):
    '''
    Raise when trying to create an unknown serializer function
    '''


class InvalidConfigError(Exception):
    '''
    Raise when the supplied config file contains invalid settings
    '''


def get_max_proc_children():
    '''
    Return the maximum number of children processes this process can create
    '''
    soft, _ = resource.getrlimit(resource.RLIMIT_NPROC)
    return soft


def get_input_byte_size(input_size_str, dtype):
    '''
    Given the input size string convert it to a single number of bytes

    @param input_size_str - string representing the input size of the data. IE: 34x34x3
    @param dtype - string representing the data type of the individual entry
    @returns integer value representing total number of bytes
    '''
    dtype_size = np.dtype(dtype).itemsize
    return functools.reduce(operator.mul, map(int, input_size_str.split('x'))) * dtype_size


def to_image(img):
    '''
    Parse image message to the protobuf Image

    @param img - serialized bytes representing input image
    @returns instance of Image
    '''
    proto = Image()
    proto.ParseFromString(img)
    return proto


def to_numpy(images):
    '''
    Convert list of protobuf Image instances to numpy arrays
    '''
    def from_image(proto):
        '''convert from Image protobuf to numpy array'''
        dtype = np.dtype(proto.dtype)
        return np.frombuffer(proto.image_data, dtype=dtype).reshape(proto.width, proto.height, proto.depth)

    return [from_image(img) for img in images]


def create_yolact_instance(weights_path):
    '''
    Creates an instance of Yolact network with the given weights

    @param weights_path - path to the weights file
    '''
    yolact_net = Yolact()
    yolact_net.load_weights(weights_path)
    yolact_net.eval()
    yolact_net = yolact_net.cuda()
    yolact_net = CustomDataParallel(yolact_net).cuda()
    return yolact_net


def create_network(network_conf):
    '''
    Given the type of the network create an appropriate network instance

    @param network_type - string representing the network to be created
    '''
    network_type = network_conf.type
    weights_path = network_conf.weights_path
    logger.info("Creating network: %s", network_type)

    if network_type == NetworkType.Yolact.name:
        return create_yolact_instance(weights_path)

    raise FactoryException("Unknown network type: {}".format(network_type))


def get_input_deserializer(input_type):
    '''
    Return the tuple of functions that will unserialize the message and convert
    it to appropriate numpy array
    '''
    if input_type == DeserializerType.Image.name:
        return to_image, to_numpy

    raise UnknownDeserializer("Network input type is not recognized: {}".format(input_type))


def get_output_serializer(output_type):
    '''
    Return the function that will serialize the network output to the
    appropriate protocol buffer message
    '''
    if output_type == SerializerType.DetectionBox.name:
        return to_detection_box

    raise UnknownSerializer("Network output type is not recognized: {}".format(output_type))


class NetworkProcess(Process):
    '''
    Neural network process - communicates with the pool through zmq and
    receives input data through shared memory
    '''
    def __init__(self, proc_name, network_conf, input_output, comm_address):
        '''
        Initialize a new instance of NetworkProcess

        @param proc_name - name of this process as string
        @param network_conf - neural network configuration
        @param input_output - shared memory array to receive input data and write output
        @param comm_address - zmq address used for communicating with the process pool
        '''
        super().__init__(name=proc_name)
        self._network_conf = network_conf
        self._input_output = input_output
        self._score_threshold = network_conf.score_threshold
        self._comm_address = comm_address
        self._deserializer, self._num_py_converter = get_input_deserializer(network_conf.input_type)
        self._serializer = get_output_serializer(network_conf.output_type)
        self._loop = None
        self._net = None
        self._transform = None
        self._stream = None

    def _command_cb(self, msg):
        '''
        Receive commands from the pool and execute

        @param msg - command msg
        '''

    def run(self):
        '''
        Override the default run method to initialize the specific network and ioloop
        '''
        with torch.no_grad():
            self._loop = ioloop.IOLoop.instance()
            self._net = create_network(self._network_conf)
            self._transform = torch.nn.DataParallel(FastBaseTransform()).cuda()
            self._stream = comm.pair_to_zmq(self._comm_address, self._command_cb, self._loop)

            logzero.logfile(self.name, maxBytes=1e7)
            logzero.loglevel(logging.INFO)

            @wraploop.eventloop
            @gen.coroutine
            def run_process(_):
                '''
                Run the server
                '''
                logger.info("Started NetworkProcess %s", self.name)
                return 0

            ret_code = run_process(self._loop)
            sys.exit(ret_code)

    def infer(self, images):
        '''
        Run inference on the given images
        '''
        numpy_images = to_numpy(images)
        frames = [torch.from_numpy(img).cuda().float() for img in numpy_images]
        height, width, _ = numpy_images[0].shape
        frames = self._transform(torch.stack(frames, 0))

        results = self._execute_network(frames, height, width)
        return results

    def _execute_network(self, frames, height, width):
        '''
        Executes the network
        '''
        predictions = self._net(frames)
        for pred in predictions:
            cur_image = {}
            # Scores
            scores = pred['score']
            keep = scores > self._score_threshold
            cur_image['score'] = scores[keep]

            # Categories
            cur_image['class'] = pred['class'][keep]

            # Mask
            boxes = pred['box'][keep]
            masks = pred['mask'][keep]
            proto = pred['proto']
            masks = torch.matmul(proto, masks.t())
            masks = torch.sigmoid(masks)
            masks = crop(masks, boxes)
            masks = masks.permute(2, 0, 1).contiguous()
            masks = F.interpolate(masks.unsqueeze(0), (height, width), mode='bilinear', align_corners=False).squeeze(0)
            masks.gt_(0.5)

            # Bounding boxes
            boxes[:, 0], boxes[:, 2] = sanitize_coordinates(boxes[:, 0], boxes[:, 2], width, cast=False)
            boxes[:, 1], boxes[:, 3] = sanitize_coordinates(boxes[:, 1], boxes[:, 3], height, cast=False)
            boxes = boxes.long()

            # Then crop masks
            # NOTE: this is BY FAR the slowest part of the full inference.
            outmasks = []
            for mask, box in zip(masks, boxes):
                x1, y1, x2, y2 = box
                outmask = mask.cpu().bool().numpy()[y1:y2, x1:x2]
                outmasks.append(outmask)
            cur_image['mask'] = outmasks


class ProcessPool(metaclass=abc.ABCMeta):
    '''
    Interface for process pool executors
    '''
    @abc.abstractmethod
    def submit(self, data):
        '''
        Submit data to be processed by this process pool

        @param data - a list of serialized protocol buffer messages to be processed by this pool
        '''

    @abc.abstractmethod
    def shutdown(self):
        '''
        Shutdown this process pool.
        '''


InputOutput = namedtuple("InputOutput", ["input", "output"])


def create_command(cmd, **kwargs):
    '''
    Create a json serialized command to be sent to the pool process

    @param cmd - Command enumeration
    @param **kwargs - free floating parameters made specific for this command
    @returns json serialized command string
    '''
    cmd_msg = {"command": cmd.name}
    cmd_msg.update(kwargs)
    return bunch.toJSON(cmd_msg)


class NetworkProcessPool(ProcessPool):
    '''
    Concrete implementation of a process pool executor made specifically for
    running neural networks with large input data sizes. The exchange of the
    data happens through the use of the shared memory to minimize the
    communication overhead.
    '''
    def __init__(self, network_conf, loop):
        '''
        Initialize a new instance of NetworkProcessPool

        @param network_conf - configuration of the pool and its underlying NN processes
        @param loop - instance of IOLoop
        '''
        self._network_conf = network_conf
        self._num_procs = network_conf.num_procs
        self._loop = loop

        max_procs = get_max_proc_children()
        if not 1 <= self._num_procs <= max_procs:
            raise InvalidConfigError("Num of procs must be a positive int between 1 and {}.".format(max_procs))

        self._procs = []
        self._zmq_socks = []
        self._proc_state = []
        self._input_output = []
        self._next_proc = 0
        self._queue = queue.deque()

    @property
    def _available_idx(self):
        '''
        Returns idx of the process that is available to do work.
        In case no available process is available returns None
        '''
        for idx, state in enumerate(self._proc_state):
            if state == ProcessState.AVAILABLE:
                return idx

        return None

    def _handle_response(self, idx, msg):
        '''
        Handle the response from the NN worker

        @param idx - idx of the process from whom this message originated
        @param msg - actual message payload as a json string
        '''

    def _create_procs(self):
        '''
        Helper method to create the processes and their associated connections
        for communication
        '''
        def response_handler(idx, msg):
            '''handle response from worker process'''
            try:
                bunch_msg = bunch.bunchify(msg)
                self._loop.add_callback(self._handle_response, idx, bunch_msg)
            except Exception as err:  # pylint: disable=broad-except
                logger.exception(err)

        for idx in range(len(self._num_procs)):
            # next line is a dirty trick - I don't know exact size of the
            # serialized protobof that will come in. I know the image size, but
            # not all other fields. Thus, I assume that extra "fudge" factor
            # can take care of this.
            input_size = self._network_conf.input_size
            dtype = self._network_conf.dtype

            input_size = get_input_byte_size(input_size, dtype) + self._network_conf.fudge
            output_size = input_size * 2
            input_array = Array(c_uint8, [0] * input_size, lock=False)
            output_array = Array(c_uint8, [0] * output_size, lock=False)

            input_output = InputOutput(input_array, output_array)
            self._input_output.append(input_output)

            proc_name = "nn_proc_{}".format(idx)
            comm_address = "ipc:///{}".format(proc_name)

            callback = functools.partial(response_handler, idx)

            zmq_sock = comm.pair_to_zmq(comm_address, callback, self._loop, is_server=True)
            self._zmq_socks.append(zmq_sock)

            process = NetworkProcess(proc_name, self._network_conf, input_output, comm_address)
            self._procs.append(process)

            self._proc_state.append(ProcessState.INITIALIZING)

    def submit(self, data):
        '''
        Submit work to be processed by this process pool

        @param data - a list of serialized protobuf messages
        @returns a list of serialized protobuf responses
        '''
        for datum in data:
            if self._available_idx is not None:
                idx = self._available_idx
                msg_length = len(datum)
                command = create_command(Command.DoWork, msg_length=msg_length)
                self._input_output[idx].input[:msg_length] = datum
                zmq_sock = self._zmq_socks[idx]
                zmq_sock.send(command)
            else:
                self._queue.append(datum)

    def shutdown(self):
        '''
        Cleanly shutdown the process pool
        '''
