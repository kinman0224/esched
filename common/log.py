from datetime import datetime
from io import TextIOBase
import logging
import sys
import time

log_level_map = {
    'fatal': logging.FATAL,
    'error': logging.ERROR,
    'warning': logging.WARNING,
    'info': logging.INFO,
    'debug': logging.DEBUG
}

_time_format = '%m/%d/%Y %I:%M:%S %p'

class _LoggerFileWrapper(TextIOBase):
    def __init__(self, logger_file):
        self.stdout = sys.stdout
        self.file = logger_file

    def write(self, s):
        if s != '\n':
            cur_time = datetime.now().strftime(_time_format)
            self.file.write('[{}] PRINT '.format(cur_time) + s + '\n')
            self.file.flush()
        return len(s)

def init_logger(logger_file_path, log_level_name='info'):
    """Initialize root logger.
    This will redirect anything from logging.getLogger() as well as stdout to specified file.
    logger_file_path: path of logger file (path-like object).
    """
    log_level = log_level_map.get(log_level_name, logging.INFO)
    logger_file = open(logger_file_path, 'w')
    fmt = '[%(asctime)s] %(levelname)s (%(name)s/%(threadName)s) %(message)s'
    logging.Formatter.converter = time.localtime
    formatter = logging.Formatter(fmt, _time_format)
    handler = logging.StreamHandler(logger_file)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.addHandler(handler)
    root_logger.setLevel(log_level)

    sys.stdout = _LoggerFileWrapper(logger_file)