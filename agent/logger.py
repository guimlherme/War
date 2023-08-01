import logging

class CustomLogger:
    def __init__(self, log_file="training_log.log"):
        self.logger = logging.getLogger("training_logger")
        self.logger.setLevel(logging.INFO)

        # Create a formatter for the log messages
        formatter = logging.Formatter("%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

        # Create a console handler to display log messages on the console
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        # Create a file handler to save log messages to a file
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

    def info(self, message):
        self.logger.info(message)

    def error(self, message):
        self.logger.error(message)