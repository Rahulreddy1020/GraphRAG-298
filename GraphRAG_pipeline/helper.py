import os
import logging


class Helper():

    def __init__(self):
        self.file_name = 'logs.log'
        self.lexim_log = logging.getLogger()
        if not self.lexim_log.handlers:
            self.lexim_log.setLevel(logging.INFO)
            file_handler = logging.FileHandler(self.file_name)
            file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s'))
            self.lexim_log.addHandler(file_handler)
        self.lexim_log.info("Logger initialized")
            
   
    def get_environ_key(self, key):        
        val = os.environ.get(key)        
        if val is None or len(val.strip()) <= 0:
            logging.error(f"{key} is missing from the environment.")
                  
        return val


    def get_logger(self):
        return self.lexim_log
    
    def clear_log_file(self):
        with open(self.file_name, 'w'):
            pass
