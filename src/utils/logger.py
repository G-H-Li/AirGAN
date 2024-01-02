import logging

import colorlog


class TrainLogger:
    """
    Write logs to file and console at the same time
    """
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.DEBUG)

        log_colors_config = {
            'DEBUG': 'white',  # cyan white
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'bold_red',
        }

        console_handler = logging.StreamHandler()
        file_handler = logging.FileHandler(self.file_path, encoding='utf-8')
        formatter = logging.Formatter(
            fmt='[%(asctime)s.%(msecs)03d] %(filename)s -> %(funcName)s line:%(lineno)d [%(levelname)s] : %(message)s',
            datefmt='%Y-%m-%d  %H:%M:%S'
        )
        console_formatter = colorlog.ColoredFormatter(
            fmt='[%(asctime)s.%(msecs)03d] %(filename)s -> %(funcName)s line:%(lineno)d [%(levelname)s] : %(message)s',
            datefmt='%Y-%m-%d  %H:%M:%S',
            log_colors=log_colors_config
        )

        console_handler.setFormatter(console_formatter)
        file_handler.setFormatter(formatter)

        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
