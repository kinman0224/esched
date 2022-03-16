import zmq

from utils import misc
import logging

L = logging.getLogger("EventLoop")

class EventLoop:
    def __init__(self, port: int):
        self.conn: zmq.Socket = zmq.Context.instance().socket(zmq.PULL)
        if port == 0:
            port = self.conn.bind_to_random_port("tcp://*")
        else:
            self.conn.bind("tcp://*:{port}".format(port=port))

        self.endpoint = "tcp://{ip}:{port}".format(ip=misc.getLocalIP(), port=port)

        self.poller = zmq.Poller()
        self.poller.register(self.conn, zmq.POLLIN)

        self.handler = {}
        self.toExit = False

    def register(self, mType: str, handler):
        self.handler[mType] = handler

    def unregister(self, mType: str):
        del self.handler[mType]

    def shutdown(self):
        self.toExit = True

    def run(self):
        while not self.toExit:
            if self.poller.poll(1000):
                msg = self.conn.recv_json(zmq.NOBLOCK)

                handler = self.handler.get(msg['type'], None)
                L.info("Recv {msg} handled: {handled}".format(msg=msg, handled=(handler is not None)))
                
                if handler:
                    handler(msg)
    
    def recv(self):
        while True:
            if self.poller.poll(1000):
                msg = self.conn.recv_json(zmq.NOBLOCK)
                L.info("Recv {msg}".format(msg=msg))

                return msg