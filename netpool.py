'''
This module defines the process pool for the neural networks that can be used
for inference
'''
from collections import namedtuple
from multiprocessing.sharedctypes import RawArray
from itertools import islice
import abc
import enum
import functools
import logging
import operator
import queue
import resource
import sys
import time

from logzero import logger
from torch.multiprocessing import Process, set_start_method
from tornado import ioloop, gen
from tornado.concurrent import Future
import logzero
import munch
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

from data import set_cfg
from eval import CustomDataParallel
from image_pb2 import Image, Detection, NetOutput, Box
from layers.box_utils import crop, sanitize_coordinates
from utils.augmentations import FastBaseTransform
from utils.functions import SavePath
from yolact import Yolact
import comm
import wraploop


try:
    set_start_method("spawn")
except RuntimeError:
    pass


MAX_BATCH_SIZE = 8


def batchify(sequence, size=2):
    """
    Breaks a sequence into a set of batches of given size.

    @param sequence - an iterable sequence: list, generator
    """
    it = iter(sequence)
    batch = tuple(islice(it, size))
    while batch:
        yield batch
        batch = tuple(islice(it, size))


class NetworkType(enum.Enum):
    '''
    List of supported NN types
    '''
    Yolact = 0


class Command(enum.Enum):
    '''
    List of supported NN process commands
    '''
    DoWork = 0    # There is work to do
    Next = 1      # Send more results back
    Shutdown = 2  # We're done, shutdown cleanly


class Response(enum.Enum):
    '''
    List of supported response types from NN processes
    '''
    Ready = 0   # expected to be sent once the process initializes the weights
    Result = 1  # work finished, part of the result available
    Done = 2    # work finished, last part of the result made available


class ProcessState(enum.Enum):
    '''
    List of possible states of a NN process
    '''
    Initializing = 0  # initial state of the process, no jobs can be processed in this state
    Available = 1  # the only state in which the process can receive and process data
    Busy = 2  # process is busy processing some data
    Broken = 3  # something went wrong and the process broke


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


class PoolNotReadyException(Exception):
    '''
    Raise when work is being submitted to the pool when it is not ready
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
    @param dtype - numpy data type of the individual entry
    @returns integer value representing total number of bytes
    '''
    dtype_size = dtype.itemsize
    return functools.reduce(operator.mul, map(int, input_size_str.split('x'))) * dtype_size


def to_image(img_payload):
    '''
    Parse bytes payload into the protobuf Image

    @param img_payload - serialized bytes representing input image
    @returns instance of Image
    '''
    proto = Image()
    proto.ParseFromString(img_payload)
    return proto


def image_to_numpy(proto):
    '''
    Convert Image protobuf to numpy array

    @param proto - Image protobuf instance
    @return numpy array
    '''
    dtype = np.dtype(proto.dtype)
    return np.frombuffer(proto.image_data, dtype=dtype).reshape(proto.width, proto.height, proto.depth)


def create_yolact_instance(weights_path):
    '''
    Creates an instance of Yolact network with the given weights

    @param weights_path - path to the weights file
    @return instance of Yolact neural network
    '''
    # Next 3 lines of code are a magic mystery, unfortunately Yolact is
    # implemented with global variables, which need to be set correctly in
    # order for the network to be able to load up the saved weights. Without
    # these lines the network fails to load.
    model_path = SavePath.from_str(weights_path)
    config = model_path.model_name + '_config'
    set_cfg(config)

    cudnn.fastest = True
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    yolact_net = Yolact()
    yolact_net.load_weights(weights_path)
    yolact_net.eval()
    yolact_net = yolact_net.cuda()
    yolact_net = CustomDataParallel(yolact_net).cuda()
    return yolact_net


class NetworkProcess(Process, metaclass=abc.ABCMeta):
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
        self._loop = None
        self._stream = None
        self._dtype = np.dtype(network_conf.dtype)
        self._output_future = None

    @abc.abstractmethod
    def _initialize(self):
        '''
        Code for initalizing this process - setup the network etc.
        '''

    @abc.abstractmethod
    def _parse_payload(self, payload):
        '''
        Parse the supplied payload containig the input data to be run through the neural network

        @param payload - instances of bytes containing the payload
        @return serialized message
        '''

    @abc.abstractmethod
    def _serialize_output(self, net_output, height, width, depth, source, sequence_num, inference_time):
        '''
        Accept raw network output and serialize it to the appropriate format expected by beyond

        @param net_output - raw network output
        @param height - height of the input image
        @param width - width of the input image
        @param depth - depth of the input image
        @param source - string representing the source of the input image
        @param sequence_num - sequence number of the input image
        @param inference_time - duration of the inference calculation
        @returns network specific output type
        '''

    @abc.abstractmethod
    def _execute_network(self, net_input):
        '''
        Executes the network with the given input

        @param net_input - serialized message this network expects. In case of Yolact - Image
        '''

    async def _transfer_output_to_pool(self, net_output):
        '''
        Transfers over all of the output of the network processing to the pool

        @param net_output - serialized output of the network
        '''
        for output in net_output:
            response_length = len(output)

            input_array = np.ctypeslib.as_array(self._input_output.output)
            input_array[:response_length] = np.frombuffer(output, self._dtype)

            # wait for the pool to ask for more
            self._output_future = Future()
            self._respond_to_pool(Response.Result, response_length=response_length)
            await self._output_future

        self._respond_to_pool(Response.Done)

    def _read_input_payload(self, msg_lengths):
        '''
        Given the lengths of each individual message read off all of them from the shared array.
        '''
        start = 0
        input_array = np.ctypeslib.as_array(self._input_output.input)
        payloads = []
        for msg_length in msg_lengths:
            end = start + msg_length
            payloads.append(input_array[start:end].tobytes())
            start = end
        return payloads

    async def _execute_command(self, cmd_msg):
        '''
        Execute the parsed command coming from the process pool

        @param cmd_msg - a Munch instance containing the command and all
                         required parameters to execute it
        '''
        command = Command[cmd_msg.command]

        if command == Command.Shutdown:
            logger.info("Process %s is shutting down", self.name)
            self._loop.stop()
            return

        elif command == Command.DoWork:
            try:
                logger.debug("%s got payload", self.name)
                payloads = self._read_input_payload(cmd_msg.msg_lengths)

                net_input = self._parse_payload(payloads)
                sources = [ni.source for ni in net_input]
                seq_nums = [ni.sequence_num for ni in net_input]
                net_output, height, depth, width, inference_time = self._execute_network(net_input)
                serialized_output = self._serialize_output(net_output,
                                                           height,
                                                           width,
                                                           depth,
                                                           sources,
                                                           seq_nums,
                                                           inference_time)

                await self._transfer_output_to_pool(serialized_output)
            except Exception as err:  # pylint: disable=broad-except
                self._respond_to_pool(Response.Done, error=str(err))
                logger.exception(err)

        elif command == Command.Next:
            # signal to result writer to write more results
            self._output_future.set_result(True)

    def _command_cb(self, msg):
        '''
        Receive commands from the pool and execute them

        @param msg - command msg
        '''
        logger.debug("Received the command: %s", msg)
        msg = munch.Munch.fromYAML(msg[0])
        self._loop.add_callback(self._execute_command, msg)

    def _respond_to_pool(self, response, **kwargs):
        '''
        Send availability response back to the pool

        @param response - response type to the network pool
        @param **kwargs - additional arguments specific to this response
        @return void
        '''
        msg = create_response(response, **kwargs)
        self._stream.send_json(msg)

    def run(self):
        '''
        Override the default run method to initialize the specific network and ioloop
        '''
        with torch.no_grad():
            logzero.logfile(self.name, maxBytes=1e7)
            logzero.loglevel(logging.INFO)

            self._initialize()
            self._loop = ioloop.IOLoop.instance()
            self._stream = comm.pair_to_zmq(self._comm_address, self._command_cb, self._loop)

            @wraploop.eventloop
            @gen.coroutine
            def run_process(_):
                '''
                Run the server
                '''
                logger.info("Started NetworkProcess %s", self.name)
                self._respond_to_pool(Response.Ready)
                return 0

            ret_code = run_process(self._loop)
            sys.exit(ret_code)


class YolactNetwork(NetworkProcess):
    '''
    Concrete implementation of the yolact network process
    '''
    def __init__(self, proc_name, network_conf, input_output, comm_address):
        super().__init__(proc_name, network_conf, input_output, comm_address)
        self._net = None
        self._transform = None

    def _initialize(self):
        '''
        Code for initalizing this process - setup the network etc. It will be
        run once the process starts running.
        '''
        logger.info("Creating the Yolact Network")
        weights_path = self._network_conf.weights_path
        self._net = create_yolact_instance(weights_path)
        self._transform = torch.nn.DataParallel(FastBaseTransform()).cuda()

    def _parse_payload(self, payloads):  # pylint: disable=arguments-differ
        '''
        Parse the serialized messages representing images

        @param payloads - a list of bytes of the protobuf messages
        @return serialized list of Image protobuf messages
        '''
        return [to_image(payload) for payload in payloads]

    def _preprocess_input(self, images):
        '''
        Convert the images to cuda torch tensor to run through the network

        @param images - a list of instances of Image protobuf
        @returns torch tensor
        '''
#        logger.info("image: %s", image.sequence_num)
        images = [image_to_numpy(image) for image in images]
        for image in images:
            logger.debug("numpy images shape: %s", image.shape)

        numpy_images = [torch.from_numpy(image).cuda().float() for image in images]
        height, width, _ = numpy_images[0].shape

        logger.debug("net ready images shape: %s", numpy_images[0].shape)
        net_ready_imgs = self._transform(torch.stack(numpy_images, 0))
        return net_ready_imgs, height, width

    def _seraialize_to_detection(self, mask, box, score, klass, source, sequence_num, inference_time):
        '''
        Convert the detected predection to the detection message

        @param mask - bounding box around the detected segment
        @param box - box coordinates within the original image
        @param score - network confidence in the inferred classification
        @param klass - integer value representing a partcular class of detection
        @param source - source of the image where this detection was found
        @param sequence_num - sequence number of the origin image
        @returns Detection protobuf instance
        '''
        x_left, y_top, x_right, y_bottom = (v.item() for v in box)
        box_proto = Box(x_left=x_left, y_top=y_top, x_right=x_right, y_bottom=y_bottom)

        max_val = np.iinfo(self._dtype).max
        outmask = mask.cpu().bool().numpy()[y_top:y_bottom, x_left:x_right]
        outmask = outmask.astype(self._dtype) * max_val

        width, height = outmask.shape
        det_proto = Detection(timestamp_sec=self._loop.time(),
                              sequence_num=sequence_num,
                              source=source,
                              width=width,
                              height=height,
                              mask=outmask.tobytes(),
                              dtype=self._dtype.name,
                              box=box_proto,
                              score=score.item(),
                              klass=klass.item(),
                              inference_time=inference_time)
        return det_proto

    def _serialize_output(self,  # pylint: disable=arguments-differ
                          predictions,
                          height,
                          width,
                          depth,
                          sources,
                          sequence_nums,
                          inference_time):
        '''
        Serialize network output to detection boxes

        @param predictions - output of yolact network as a list of predictions
        @param height - height of the input image
        @param width - width of the input image
        @param depth - depth of the input image
        @param sources - string representation of the images origins
        @param sequence_nums - sequence numbers of the input images
        @param inference_time - duration of the inference calculation
        @returns a list of DetectionBox instances
        '''
        detections = []
        start = time.perf_counter()
        for pred, source, sequence_num in zip(predictions, sources, sequence_nums):
            # Scores
            scores = pred['score']
            keep = scores > self._score_threshold
            scores = scores[keep]
            if len(scores) < 1:
                continue

            # Categories
            classes = pred['class'][keep]

            # Mask
            boxes = pred['box'][keep]
            masks = pred['mask'][keep]
            proto = pred['proto']
            masks = torch.matmul(proto, masks.t())
            masks = torch.sigmoid(masks)
            masks = crop(masks, boxes)
            masks = masks.permute(2, 0, 1).contiguous()
            masks = F.interpolate(masks.unsqueeze(0),
                                  (height, width),
                                  mode='bilinear',
                                  align_corners=False).squeeze(0)
            masks.gt_(0.5)

            # Bounding boxes
            boxes[:, 0], boxes[:, 2] = sanitize_coordinates(boxes[:, 0], boxes[:, 2], width, cast=False)
            boxes[:, 1], boxes[:, 3] = sanitize_coordinates(boxes[:, 1], boxes[:, 3], height, cast=False)
            boxes = boxes.long()

            # Then crop masks
            # NOTE: this is BY FAR the slowest part of the full inference.
            for mask, box, score, klass in zip(masks, boxes, scores, classes):
                det_proto = self._seraialize_to_detection(mask, box, score, klass, source, sequence_num, inference_time)
                detections.append(det_proto.SerializeToString())

        end = time.perf_counter()
        logger.debug("serialization took: %s seconds", (end - start))

        return detections

    def _execute_network(self, images):  # pylint: disable=arguments-differ
        '''
        Executes the network with the provided image

        @param images - images sent to this network for processing
        @returns network output, input height, depth, width and inference time
        '''
        inference_start = time.perf_counter()
        net_ready_img, height, width = self._preprocess_input(images)

        predictions = self._net(net_ready_img)
        inference_time = time.perf_counter() - inference_start
        logger.debug("Inference time: %s", inference_time)
        return predictions, height, None, width, inference_time


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

    @abc.abstractproperty
    def is_ready(self):
        '''
        Return True if this process pool is initialized and ready to accept work
        '''


InputOutput = namedtuple("InputOutput", ["input", "output"])


def create_message(msg_type, value, **kwargs):
    '''
    Create a json serialized message of given type: command, or response. Value
    is the specific kind of msg: DoWork, or Available

    @param msg_type - type of message, either command or response
    @param value - particular instance of message type

    '''
    msg = {msg_type: value.name}
    msg.update(kwargs)
    return msg


def create_command(cmd, **kwargs):
    '''
    Create a json serialized command message to be sent to the pool process

    @param cmd - Command enumeration
    @param **kwargs - free floating parameters made specific for this command
    @returns json serialized command string
    '''
    return create_message("command", cmd, **kwargs)


def create_response(response, **kwargs):
    '''
    Create a json serialized response message to be sent to the process pool

    @param response - type of the response message
    @param kwargs - specifics of this particular message
    @returns json serialized response string
    '''
    return create_message("response", response, **kwargs)


def create_network_process(network_conf, proc_name, input_output, comm_address):
    '''
    Create an appropriate network process instance based on the network type

    @param network_type - string representing the network type. IE: Yolact
    @return new instance of NetworkProcess
    '''
    try:
        network_type = network_conf.type
        known_type = NetworkType[network_type]
    except KeyError:
        raise FactoryException("Unknown network type: {}".format(network_type))

    if known_type == NetworkType.Yolact:
        return YolactNetwork(proc_name, network_conf, input_output, comm_address)

    raise FactoryException("Unknown network type: {}".format(network_type))


class NetworkProcessPool(ProcessPool):
    '''
    Concrete implementation of a process pool executor made specifically for
    running neural networks with large input data sizes. The exchange of the
    data happens through the use of the shared memory to minimize the
    communication overhead.
    '''
    def __init__(self, network_conf, queue_size=9, loop=ioloop.IOLoop.instance()):
        '''
        Initialize a new instance of NetworkProcessPool

        @param network_conf - configuration of the pool and its underlying NN processes
        @param queue_size - number of work items this pool will hold before dropping
        @param loop - instance of IOLoop
        '''
        self._network_conf = network_conf
        self._num_procs = network_conf.num_procs
        self._batch_size = network_conf.batch_size
        self._queue_size = queue_size
        self._loop = loop
        self._dtype = np.dtype(network_conf.dtype)

        max_procs = get_max_proc_children()
        if not 1 <= self._num_procs <= max_procs:
            raise InvalidConfigError("Num of procs must be a positive int between 1 and {}.".format(max_procs))

        if not 1 <= self._batch_size <= MAX_BATCH_SIZE:
            raise InvalidConfigError("Batch size must be a positive int between 1 and {}.".format(MAX_BATCH_SIZE))

        self._procs = []
        self._zmq_socks = []
        self._proc_state = []
        self._input_output = [None for _ in range(self._num_procs)]
        self._queue = queue.deque()
        self._proc_futures = None
        self._proc_results = []
        self._create_procs()

    def _available_idx(self):
        '''
        Returns idx of the process that is available to do work.
        In case no available process is available returns None
        '''
        for idx, state in enumerate(self._proc_state):
            if state == ProcessState.Available:
                return idx

        return None

    def _set_proc_state(self, proc_idx, state):
        '''
        Set process state to the given state for the given proc_idx

        @param proc_idx - idx of the process for which to set the state
        @param state - instance of ProcessState
        @return void
        '''
        self._proc_state[proc_idx] = state

    def _run_next_work(self):
        '''
        Picks up first available work in the queue and schedule it be run by
        the first available process
        '''
        more_work = len(self._queue) > 0
        proc_idx = self._available_idx()
        if more_work and proc_idx is not None:
            work = self._queue.popleft()
            self._start_work(proc_idx, work)

    def _handle_response(self, proc_idx, msg):
        '''
        Handle the response from the NN worker

        @param proc_idx - idx of the process from whom this message originated
        @param msg - actual message payload as a json string
        '''
        logger.debug("Got response from %s", self._procs[proc_idx].name)
        msg = munch.Munch.fromYAML(msg[0])

        response_type = Response[msg.response]
        if response_type == Response.Ready:
            logger.info("Process %s is Ready", self._procs[proc_idx].name)
            self._set_proc_state(proc_idx, ProcessState.Available)

        elif response_type == Response.Result:
            logger.debug("Process %s returned result", self._procs[proc_idx].name)

            # TODO: Read the output result and store it for this process
            output_array = np.ctypeslib.as_array(self._input_output[proc_idx].output)
            response = output_array[:msg.response_length].tobytes()
            self._proc_results.append(response)

            self._send_command(proc_idx, Command.Next)

        elif response_type == Response.Done:
            self._set_proc_state(proc_idx, ProcessState.Available)

            self._proc_futures[proc_idx][0].set_result(True)

            self._run_next_work()

    def _allocate_shared_arrays(self, proc_idx):
        '''
        Allocates the shared arrays for the input and output of the worker process

        @param proc_idx - index of the process
        '''
        input_size = self._network_conf.input_size

        # next line is a dirty trick - I don't know exact size of the
        # serialized protobof that will come in. I know the image size, but
        # not all other fields. Thus, I assume that extra "fudge" factor
        # can take care of this.
        input_size = get_input_byte_size(input_size, self._dtype) + self._network_conf.fudge
        output_size = input_size

        # scale input by the batch size
        input_size *= self._batch_size

        ctype = np.ctypeslib.as_ctypes_type(self._dtype)
        input_array = RawArray(ctype, input_size)
        output_array = RawArray(ctype, output_size)

        input_output = InputOutput(input_array, output_array)
        self._input_output[proc_idx] = input_output

    def _create_procs(self):
        '''
        Helper method to create the processes and their associated connections
        for communication
        '''
        def response_handler(idx, msg):
            '''handle response from worker process'''
            try:
                self._loop.add_callback(self._handle_response, idx, msg)
            except Exception as err:  # pylint: disable=broad-except
                logger.exception(err)

        for idx in range(self._num_procs):

            self._allocate_shared_arrays(idx)

            proc_name = "nn_proc_{}".format(idx)
            comm_address = "ipc:///tmp/{}".format(proc_name)

            callback = functools.partial(response_handler, idx)

            zmq_sock = comm.pair_to_zmq(comm_address, callback, self._loop, is_server=True)
            self._zmq_socks.append(zmq_sock)

            input_output = self._input_output[idx]
            process = create_network_process(self._network_conf, proc_name, input_output, comm_address)
            self._procs.append(process)

            self._proc_state.append(ProcessState.Initializing)
            process.start()

    def _push_to_queue(self, work):
        '''
        Push to the queue respecting its size.

        @param work - data to be processed by the this pool
        '''
        while len(self._queue) > self._queue_size:
            # drop old data
            logger.warning("Droping old data")
            self._queue.popleft()
        self._queue.append(work)

    def _send_command(self, proc_idx, command, **kwargs):
        '''
        Send a command to a specific process

        @param proc_idx - index of the process to be commanded
        @param command - instance of Command enum, DoWork, Next etc
        @param **kwargs - additional command specific arguments
        '''
        command = create_command(command, **kwargs)
        zmq_sock = self._zmq_socks[proc_idx]
        zmq_sock.send_json(command)

    def _transfer_to_network_proces(self, proc_idx, batch):
        '''
        Write out the batch of data to the networks process shared memory

        @param proc_idx - index of the target process
        @param batch - batch of the input data
        '''
        input_array = np.ctypeslib.as_array(self._input_output[proc_idx].input)
        start = 0

        for item in batch:
            end = start + len(item)
            input_array[start:end] = np.frombuffer(item, self._dtype)
            start = end

    def _start_work(self, proc_idx, work):
        '''
        Push new work item to the process specified by its index.

        @param proc_idx - index of the process to do the work
        @param work - tuple batch of data to be processed by the NN
        '''
        msg_lengths = [len(w) for w in work]

        self._transfer_to_network_proces(proc_idx, work)

        self._set_proc_state(proc_idx, ProcessState.Busy)
        self._proc_futures[proc_idx].append(Future())
        self._send_command(proc_idx, Command.DoWork, msg_lengths=msg_lengths)

    @property
    def is_ready(self):
        '''
        True if work can be submitted, False otherwise
        '''
        all_procs_available = all(state == ProcessState.Available for state in self._proc_state)
        return all_procs_available

    def _reset_futures(self):
        '''
        Helper method to reset futures list to avoid using the old ones
        '''
        self._proc_futures = [[] for _ in range(self._num_procs)]

    def _clean_done_futures(self, proc_idx):
        '''
        Removes all futures for a particular process that have been completed

        @param proc_idx - index of the process
        '''
        self._proc_futures[proc_idx] = [future for future in self._proc_futures[proc_idx] if not future.done()]

    def _get_next_future(self):
        '''
        Generator that will continusly produce futures while the children
        processes still have something to do.
        '''
        def still_processing(proc_idx):
            '''returns true while the process still has futures'''
            self._clean_done_futures(proc_idx)
            return bool(self._proc_futures[proc_idx])

        for idx in range(self._num_procs):
            while still_processing(idx):
                yield self._proc_futures[idx][0]

    async def submit(self, data):
        '''
        Submit work to be processed by this process pool. A strong expectation
        is that this method won't be called from multiple threads - calling
        this method before it is done will result in raising PoolNotReadyException

        @param data - a list of serialized protobuf messages
        @returns a list of serialized protobuf responses
        @raises PoolNotReadyException if the pool is not initialized or is processing data already
        '''
        if not self.is_ready:
            raise PoolNotReadyException("This network pool is still processing data, or initializing")

        self._reset_futures()
        self._proc_results = []

        for batch in batchify(data, self._batch_size):
            proc_idx = self._available_idx()
            if proc_idx is not None:
                self._start_work(proc_idx, batch)
            else:
                self._push_to_queue(batch)

        for future in self._get_next_future():
            await future

        logger.debug("Processed all of the data")
        net_output = NetOutput(output=self._proc_results)
        return net_output.SerializeToString()

    def shutdown(self):
        '''
        Cleanly shutdown the process pool
        '''
        for zmq_con in self._zmq_socks:
            zmq_con.close()

        for proc in self._procs:
            logger.info("Terminating process %s", proc.name)
            proc.terminate()

        if self._proc_futures:
            for proc_idx in range(self._num_procs):
                for future in self._proc_futures[proc_idx]:
                    if not future.done():
                        future.cancel()
