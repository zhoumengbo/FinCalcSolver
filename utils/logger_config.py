import logging


# ANSI escape codes for colors
class CustomFormatter(logging.Formatter):
    grey = "\x1b[30;1m"
    white = "\x1b[97m"
    red = "\x1b[31m"
    reset = "\x1b[0m"
    format = "%(asctime)s - %(levelname)s - %(message)s"

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: white + format + reset,
        logging.WARNING: grey + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: red + format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


class LoggerConfig:
    def __init__(self, log_file='app.log', level=logging.INFO):
        self.logger = logging.getLogger(log_file)
        self.logger.setLevel(level)
        self.logger.propagate = False

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(level)
        stream_handler.setFormatter(CustomFormatter())

        self.logger.addHandler(file_handler)
        self.logger.addHandler(stream_handler)
