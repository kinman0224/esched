import zmq

class WorkerConn:
    def __init__(self, endpoint: str):
        self.endpoint = endpoint

        self.conn: zmq.Socket = zmq.Context.instance().socket(zmq.PUSH)
        self.conn.connect(self.endpoint)

        self.job = None

        self.available = False
        self.status = None


    def __repr__(self):
        return "Worker(wid={wid}, host={host}, gpu={gpuid}, job={jid}, avail={avail})".format(
            wid=self.wid, host=self.host, gpuid=self.gpuid, 
            jid=(self.job.jid if self.job else None), avail=self.available)

    def __eq__(self, other):
        return isinstance(other, WorkerConn) and self.wid == other.wid

    def __hash__(self):
        return hash(self.wid)
