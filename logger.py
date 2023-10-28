import logging

class Logger():
    def __init__(self, log_file, print_log=False):
        self.logger = logging.getLogger("logger")
        self.handler = logging.FileHandler(log_file)
        self.handler.setLevel(logging.INFO)
        self.formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.handler.setFormatter(self.formatter)
        self.logger.addHandler(self.handler)
        self.print_log = print_log

    def output(self, msg):
        self.logger.info(msg)
        if self.print_log:
            print(msg)
            
    def debug(self, msg):
        print(msg)