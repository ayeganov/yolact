import errno
import functools
import logging
import sys

from tornado.concurrent import Future, is_future
import zmq

log = logging.getLogger(__name__)


def eventloop(func):
    """
    Wraps a function and invokes a ZMQ eventloop, or tornado loop.
    The intention of this decorator is to provide a standard way of starting up
    code that runs with the ZMQ eventloop. The main issue here is that ZMQ
    occassionally receives a EINTR error when creating a socket connection or
    communicating via a socket and this will not be handled by tornado.
    The following shows a simple example of how to use the decorator.
        def foo():
            print('foo')
        @eventloop
        def frobnicator(loop):
            pub = zmq.eventloop.PeriodicCallback(
                    foo,
                    1000,
                    io_loop=loop)
            pub.start()
        loop = zmq.eventloop.ioloop.IOLoop.instance()
        frobnicator(loop)
    This will result in 'foo' being written to stdout once per second.
    """
    @functools.wraps(func)
    def impl(loop):
        future_cell = [None]

        def run():
            try:
                result = func(loop)
                if result is not None:
                    from tornado.gen import convert_yielded
                    result = convert_yielded(result)
            except Exception:
                future_cell[0] = Future()
                future_cell[0].set_exc_info(sys.exc_info())
            else:
                if is_future(result):
                    future_cell[0] = result
                else:
                    future_cell[0] = Future()
                    future_cell[0].set_result(result)
            loop.add_future(future_cell[0], lambda future: loop.stop())

        loop.add_callback(run)
        try:
            loop.start()
        except (SystemExit, KeyboardInterrupt):
            log.info('Exiting due to interrupt')
            print('Exiting due to interrupt')
            return 0

        # If loop was stopped before the future was done then the user
        # requested an explicit stoppage, which is fine
        if not future_cell[0].done():
            return 0

        while True:
            try:
                if future_cell:
                    future_cell[0].result()
                    future_cell = []
                loop.start()
                return 0
            except zmq.ZMQError as e:
                if e.errno == errno.EINTR:
                    continue
                log.exception(e)
                return 1
            except Exception as e:
                log.exception(e)
                return 1
            except (SystemExit, KeyboardInterrupt):
                log.info('Exiting due to interrupt')
                print('Exiting due to interrupt')
                return 0

    return impl
