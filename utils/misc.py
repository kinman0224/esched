import socket
import json
from datetime import datetime

def getLocalIP():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(('8.8.8.8', 1))  # connect() for UDP doesn't send packets
    return s.getsockname()[0]

def now():
    return datetime.now().strftime('%m/%d/%Y %I:%M:%S')

def getHostname():
    return socket.gethostname()

def indexOf(l, x):
    try:
        return l.index(x)
    except:
        return None